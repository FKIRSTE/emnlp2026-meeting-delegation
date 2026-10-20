"""
> Supports multiple LLM providers:
    - Azure OpenAI (gpt-4o, gpt-5-mini)
    - Direct OpenAI (gpt-4o-openai, gpt-5-mini-openai, gpt-5-openai)
    - Google GenAI (gemini-2.5-pro)

> The client automatically detects the provider from the config and routes
calls to the appropriate backend.
"""

import time
import random
import logging
import re
import sys
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import Config
from llm_telemetry import (
    LLMTelemetryRecorder,
    estimate_tokens,
    extract_response_metadata,
    request_fingerprint,
)


def _sanitize_identity_aliases(payload: Dict[str, Any], schema_name: str) -> None:
    """Drop unresolved or malformed alias entries before EIM schema validation."""
    aliases = payload.get("identity_aliases")
    if aliases is None or not isinstance(aliases, dict):
        return

    cleaned_aliases: Dict[str, str] = {}
    dropped_aliases: List[str] = []

    for raw_alias, raw_target in aliases.items():
        alias = raw_alias.strip() if isinstance(raw_alias, str) else str(raw_alias).strip()
        if not alias:
            dropped_aliases.append("<empty>")
            continue
        if not isinstance(raw_target, str):
            dropped_aliases.append(alias)
            continue
        target = raw_target.strip()
        if not target:
            dropped_aliases.append(alias)
            continue
        cleaned_aliases[alias] = target

    if dropped_aliases:
        payload["identity_aliases"] = cleaned_aliases
        dropped_preview = ", ".join(dropped_aliases[:5])
        if len(dropped_aliases) > 5:
            dropped_preview = f"{dropped_preview}, ..."
        print(
            f"[WARN] Dropped {len(dropped_aliases)} invalid identity_aliases entr"
            f"{'y' if len(dropped_aliases) == 1 else 'ies'} in {schema_name}: {dropped_preview}"
        )
    elif cleaned_aliases != aliases:
        payload["identity_aliases"] = cleaned_aliases


class LLMClient:
    """Extended LLM client supporting multiple providers.
    
    Drop-in replacement for the original LLMClient class.
    Automatically routes calls to Azure OpenAI or Google GenAI based on config.
    """
    
    def __init__(self, config: Config):
        """Initialize the LLM client with provider-specific configuration."""
        self.config = config
        self.provider = getattr(config, 'provider', 'azure_openai')
        self.model = config.model
        self.telemetry: Optional[LLMTelemetryRecorder] = None
        self._last_response_metadata: Dict[str, Any] = {}

        telemetry_path = os.environ.get("MD_LLM_TELEMETRY_PATH")
        if telemetry_path:
            self.configure_telemetry(
                telemetry_path,
                base_context={"provider": self.provider, "configured_model": self.model},
            )
        
        # Initialize the appropriate client
        if self.provider == "azure_openai":
            self._init_azure(config)
        elif self.provider in {"openai", "groq"}:
            self._init_openai(config)
        elif self.provider == "google_genai":
            self._init_gemini(config)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def configure_telemetry(
        self,
        path: Union[str, Path],
        *,
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Enable append-only JSONL telemetry for this client."""
        context = {
            "provider": self.provider,
            "configured_model": self.model,
            **(base_context or {}),
        }
        self.telemetry = LLMTelemetryRecorder(Path(path), base_context=context)
        return self.telemetry.path

    def set_telemetry_context(self, **context: Any) -> None:
        if self.telemetry:
            self.telemetry.set_context(**context)

    def clear_telemetry_context(self, *keys: str) -> None:
        if self.telemetry:
            self.telemetry.clear_context(*keys)

    def record_telemetry_span(
        self,
        name: str,
        elapsed_seconds: float,
        *,
        phase: Optional[str] = None,
        status: str = "success",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.telemetry:
            return
        self.telemetry.record(
            {
                "event_type": "span",
                "span_name": name,
                "phase": phase,
                "status": status,
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                **(extra or {}),
            }
        )

    def _capture_response_metadata(self, response: Any) -> None:
        self._last_response_metadata = extract_response_metadata(response, self.provider)
    
    def _init_azure(self, config: Config):
        """Initialize Azure OpenAI client."""
        from openai import AzureOpenAI
        import httpx
        
        # Set 5-minute timeout to prevent infinite hangs
        # The retry mechanism will handle timeout exceptions
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            timeout=httpx.Timeout(300.0, connect=60.0)  # 5 min total, 1 min connect
        )
        self._call_impl = self._call_azure
        self._is_gpt5 = "gpt-5" in config.model.lower()

    def _init_openai(self, config: Config):
        """Initialize direct OpenAI client."""
        from openai import OpenAI
        import httpx
        import ssl

        if config.api_mode not in {"chat_completions", "chat"}:
            raise ValueError(f"Unsupported OpenAI api_mode: {config.api_mode}")

        client_kwargs = {
            "api_key": config.api_key,
            "timeout": httpx.Timeout(300.0, connect=60.0),
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.use_system_ca:
            import truststore

            ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client_kwargs["http_client"] = httpx.Client(
                verify=ssl_context,
                timeout=httpx.Timeout(300.0, connect=60.0),
            )

        self.client = OpenAI(**client_kwargs)
        self._call_impl = self._call_openai
        self._is_gpt5 = "gpt-5" in config.model.lower()

    def _init_gemini(self, config: Config):
        """Initialize Google GenAI client."""
        from google import genai
        from google.genai.types import HttpOptions
        
        self.client = genai.Client(
            http_options=HttpOptions(base_url=config.base_url),
            api_key=config.api_key,
        )
        self.thinking_budget = config.thinking_budget
        self._call_impl = self._call_gemini
    
    def _prepare_openai_schema(self, response_model) -> dict:
        """Prepare a Pydantic schema for OpenAI strict structured outputs.
        
        OpenAI strict mode requires:
        1. All properties must be in the 'required' array
        2. additionalProperties must be false
        
        This recursively fixes nested schemas.
        """
        schema = response_model.model_json_schema()
        fixed = self._fix_openai_schema(schema)
        return fixed
    
    def _fix_openai_schema(self, schema: dict) -> dict:
        """Recursively fix schema for OpenAI strict mode.
        
        OpenAI's strict structured output has several requirements:
        1. All properties must be in the 'required' array
        2. additionalProperties must be false
        3. $ref CANNOT be combined with other keywords (like 'description', 'default')
        """
        if not isinstance(schema, dict):
            return schema
        
        # Handle $defs (definitions for nested models)
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                schema["$defs"][def_name] = self._fix_openai_schema(def_schema)
        
        # CRITICAL FIX: OpenAI strict mode doesn't allow $ref with other keywords
        # If we have $ref, strip all other keys except $ref and allOf/anyOf/oneOf
        if "$ref" in schema:
            allowed_with_ref = {"$ref", "allOf", "anyOf", "oneOf"}
            keys_to_remove = [k for k in schema.keys() if k not in allowed_with_ref]
            for k in keys_to_remove:
                del schema[k]
            return schema  # No further processing needed for $ref schemas
        
        # If this is an object type with properties, fix required and additionalProperties
        if schema.get("type") == "object" and "properties" in schema:
            # Make all properties required
            schema["required"] = list(schema["properties"].keys())
            # Disallow additional properties
            schema["additionalProperties"] = False
            
            # Recursively fix nested object properties
            for prop_name, prop_schema in schema["properties"].items():
                schema["properties"][prop_name] = self._fix_openai_schema(prop_schema)
        
        # Handle arrays with items
        if schema.get("type") == "array" and "items" in schema:
            schema["items"] = self._fix_openai_schema(schema["items"])
        
        # Handle anyOf (used for Optional types)
        if "anyOf" in schema:
            schema["anyOf"] = [self._fix_openai_schema(s) for s in schema["anyOf"]]
        
        # Handle allOf 
        if "allOf" in schema:
            schema["allOf"] = [self._fix_openai_schema(s) for s in schema["allOf"]]
        
        return schema

    
    def validate_json_response(self, json_response: str, schema_class):
        """
        Parse JSON response and validate against a Pydantic schema.
        
        This method is provider-agnostic - works with any JSON string.
        
        Args:
            json_response: JSON string from LLM
            schema_class: Pydantic model class for validation
            
        Returns:
            Validated Pydantic model instance or None if validation failed
        """
        if not json_response or not schema_class:
            return None
            
        try:
            # Parse JSON from response
            parsed_json = json.loads(json_response)

            # Only dict payloads are valid for schema validation
            if not isinstance(parsed_json, dict):
                print(f"[WARN] Parsed JSON is not an object; skipping validation for {schema_class.__name__}")
                return None

            # Coerce known mapping fields that occasionally arrive as lists
            def _ensure_dict(d, key):
                val = d.get(key)
                if isinstance(val, list):
                    d[key] = {}
                    print(f"[WARN] Coerced list to empty dict for '{key}' in {schema_class.__name__}")

            if schema_class.__name__ in ['EIMMasterState', 'EIMMeetingStateVector']:
                _ensure_dict(parsed_json, 'topic_summary_log')
                _ensure_dict(parsed_json, 'participant_models')
                _ensure_dict(parsed_json, 'identity_aliases')
                _sanitize_identity_aliases(parsed_json, schema_class.__name__)
            
            # Special handling for EIM schemas to auto-fix missing fields
            if schema_class.__name__ in ['EIMMasterState', 'EIMMeetingStateVector'] and 'topic_summary_log' in parsed_json:
                for topic_key, topic_data in parsed_json['topic_summary_log'].items():
                    if isinstance(topic_data, dict):
                        if 'label' not in topic_data:
                            topic_data['label'] = topic_key
                            print(f"  [Auto-fix] Added missing label '{topic_key}' to topic")
                        if 'status' not in topic_data:
                            topic_data['status'] = 'Ongoing'
                            print(f"  [Auto-fix] Added missing status 'Ongoing' to topic '{topic_key}'")
                        if 'key_points' not in topic_data:
                            topic_data['key_points'] = []
                            print(f"  [Auto-fix] Added missing key_points [] to topic '{topic_key}'")
            
            # Auto-fix expressed_stance_on_topics format
            if schema_class.__name__ in ['EIMMasterState', 'EIMMeetingStateVector'] and 'participant_models' in parsed_json:
                for participant_id, participant_data in parsed_json['participant_models'].items():
                    if isinstance(participant_data, dict) and 'expressed_stance_on_topics' in participant_data:
                        stance_topics = participant_data['expressed_stance_on_topics']
                        if isinstance(stance_topics, dict):
                            for topic_key, stance_data in stance_topics.items():
                                if isinstance(stance_data, dict) and 'stance_summary' in stance_data:
                                    stance_topics[topic_key] = [stance_data]
                                    print(f"  [Auto-fix] Wrapped stance for {participant_id} on '{topic_key}' in list")

            # A semantic non-match has the canonical discrete strength "none".
            # JSON-object backends sometimes emit null despite the other fields
            # unambiguously identifying a non-match. Keep this deliberately
            # narrow: partial and same-proposition cases still fail validation.
            if (
                schema_class.__name__ == 'GTMatchVerifierReport'
                and parsed_json.get('proposition_relation') == 'different'
                and parsed_json.get('matched_utt_id') is None
                and parsed_json.get('same_proposition') in (None, False)
                and parsed_json.get('match_strength') is None
            ):
                parsed_json['same_proposition'] = False
                parsed_json['match_strength'] = 'none'
                print("  [Auto-fix] Canonicalized GT non-match fields")
            
            # Validate against Pydantic schema
            validated_instance = schema_class(**parsed_json)
            print(f"[OK] JSON validation successful for {schema_class.__name__}")
            return validated_instance
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            print(f"Raw response: {json_response[:200]}...")
            return None
            
        except ValidationError as e:
            print(f"[ERROR] Schema validation failed for {schema_class.__name__}: {e}")
            print(f"Raw JSON: {json.dumps(parsed_json, indent=2) if 'parsed_json' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected validation error: {e}")
            return None
        
    def _build_openai_chat_params(
        self,
        messages: List[Dict],
        max_tokens: int,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Build provider-agnostic chat-completion params for OpenAI-family backends."""
        model_name = model or self.model
        model_params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        if self._is_gpt5:
            model_params["max_completion_tokens"] = max_tokens
            model_params["reasoning_effort"] = "low"
        else:
            model_params["max_tokens"] = max_tokens
            if temperature is not None:
                model_params["temperature"] = temperature
            if top_p is not None:
                model_params["top_p"] = top_p
            if frequency_penalty is not None:
                model_params["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                model_params["presence_penalty"] = presence_penalty

        model_params.update(self.config.default_request_params)
        model_params.update(kwargs)
        return model_params

    @staticmethod
    def _extract_chat_content(response: Any) -> Optional[str]:
        if hasattr(response, 'choices'):
            return response.choices[0].message.content
        return str(response) if response is not None else None

    @staticmethod
    def _schema_error_should_fallback(error: Exception) -> bool:
        error_str = str(error).lower()
        fallback_signals = [
            "json_schema",
            "structured output",
            "structured outputs",
            "response_format",
            "unsupported",
            "not supported",
            "invalid parameter",
            "invalid value",
            "schema",
        ]
        return any(signal in error_str for signal in fallback_signals)

    def _build_json_schema_format(self, response_model) -> Dict[str, Any]:
        schema = self._prepare_openai_schema(response_model)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    def _parse_structured_content(self, content: Optional[str], response_model, provider_label: str) -> Any:
        if not content:
            print(f"[WARN] {provider_label} returned empty structured content")
            return None
        try:
            parsed = json.loads(content)
            return response_model(**parsed)
        except Exception as exc:
            print(f"[WARN] {provider_label} structured output parsing failed: {exc}")
            print(f"[DEBUG] Raw content: {content[:500]}..." if len(content) > 500 else f"[DEBUG] Raw content: {content}")
            return content

    def _call_openai_json_mode(self, model_params: Dict[str, Any], response_model=None) -> Any:
        response = self.client.chat.completions.create(**model_params)
        self._capture_response_metadata(response)
        print(f"\n[DEBUG_MODEL_EXTRACT] API resolved this JSON call to exact model: {getattr(response, 'model', 'Unknown')}")
        content = self._extract_chat_content(response)
        if not content:
            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown') if hasattr(response, 'choices') else 'unknown'
            refusal = getattr(response.choices[0].message, 'refusal', None) if hasattr(response, 'choices') else None
            print(f"[WARN] JSON-required call returned empty content. finish_reason={finish_reason}, refusal={refusal}")
            return None
        if response_model:
            validated = self.validate_json_response(content, response_model)
            return validated or content
        return content

    def call_with_retry(
        self, 
        messages: List[Dict], 
        max_tokens: int = 1000, 
        max_attempts: int = 6, 
        base_delay: float = 3.0, 
        json_required: bool = False, 
        response_model=None,
        use_strict_json: bool = True,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.1,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        telemetry_label: Optional[str] = None,
        telemetry_phase: Optional[str] = None,
        telemetry_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Call model with retry logic for rate limiting and transient errors.
        
        Args:
            messages: List of message dictionaries for the conversation
            max_tokens: Maximum tokens in response
            max_attempts: Maximum retry attempts
            base_delay: Base delay between attempts
            json_required: Whether to force JSON response format
            response_model: Optional Pydantic model for structured output (Azure only)
            model: Override model name (uses default if None)
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0) - Azure only
            presence_penalty: Presence penalty (-2.0 to 2.0) - Azure only
            telemetry_label: Stable module name written to the telemetry event
            telemetry_phase: High-level phase such as pre_turn, floor, or post_turn
            telemetry_context: Optional call-local identifiers/metadata
            **kwargs: Additional parameters to pass to the model
        """
        telemetry_enabled = self.telemetry is not None
        call_id = str(uuid.uuid4()) if telemetry_enabled else None
        logical_start = time.perf_counter()
        response_schema = None
        if telemetry_enabled and response_model is not None:
            try:
                response_schema = response_model.model_json_schema()
            except Exception:
                response_schema = str(response_model)
        request_meta = request_fingerprint(messages, response_schema) if telemetry_enabled else {}
        requested_model = model or self.model

        for attempt in range(max_attempts):
            attempt_start = time.perf_counter()
            provider_start = attempt_start
            provider_elapsed = 0.0
            retry_sleep_seconds = 0.0
            post_attempt_delay_seconds = 0.0
            error: Optional[Exception] = None
            retry_planned = False
            response = None
            self._last_response_metadata = {}
            try:
                response = self.call_model(
                    messages=messages, 
                    max_tokens=max_tokens, 
                    json_required=json_required, 
                    response_model=response_model,
                    use_strict_json=use_strict_json,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    **kwargs
                )
                provider_elapsed = time.perf_counter() - provider_start
            except Exception as e:
                provider_elapsed = time.perf_counter() - provider_start
                error = e
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000)
                    logging.warning(f"Rate limit hit, backing off for {sleep_time} seconds.")
                    retry_planned = attempt < max_attempts - 1
                    retry_sleep_seconds = sleep_time
                    time.sleep(sleep_time)
                elif attempt < max_attempts - 1:
                    print(f"[Attempt {attempt+1}/{max_attempts}] Error: {error_str[:200]}", flush=True)
                    retry_planned = True
                    retry_sleep_seconds = base_delay
                    time.sleep(base_delay)
                else:
                    print(f"[FINAL] Error after {max_attempts} attempts: {error_str}", flush=True)
            finally:
                # Preserve the existing inter-attempt throttle. It is logged
                # separately so reports can distinguish provider latency from it.
                delay_start = time.perf_counter()
                time.sleep(base_delay)
                post_attempt_delay_seconds = time.perf_counter() - delay_start

                if self.telemetry:
                    provider_meta = dict(self._last_response_metadata)
                    input_tokens = provider_meta.get("input_tokens")
                    output_tokens = provider_meta.get("output_tokens")
                    usage_source = provider_meta.get("usage_source")
                    if input_tokens is None:
                        input_tokens = request_meta["estimated_input_tokens"]
                        usage_source = "estimated" if not usage_source else f"{usage_source}+estimated_input"
                    if output_tokens is None and error is None and response is not None:
                        if hasattr(response, "model_dump"):
                            output_payload = response.model_dump()
                        else:
                            output_payload = response
                        output_tokens = estimate_tokens(output_payload)
                        usage_source = "estimated" if not usage_source else f"{usage_source}+estimated_output"
                    total_tokens = provider_meta.get("total_tokens")
                    if total_tokens is None and input_tokens is not None and output_tokens is not None:
                        total_tokens = int(input_tokens) + int(output_tokens)

                    status = "error" if error is not None else ("success" if response is not None else "empty_response")
                    self.telemetry.record(
                        {
                            "event_type": "llm_call_attempt",
                            "call_id": call_id,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "module": telemetry_label or "unlabeled",
                            "phase": telemetry_phase or "unspecified",
                            "status": status,
                            "retry_planned": retry_planned,
                            "requested_model": requested_model,
                            "resolved_model": provider_meta.get("resolved_model"),
                            "response_id": provider_meta.get("response_id"),
                            "system_fingerprint": provider_meta.get("system_fingerprint"),
                            "service_tier": provider_meta.get("service_tier"),
                            "finish_reason": provider_meta.get("finish_reason"),
                            "response_schema": getattr(response_model, "__name__", None),
                            "json_required": bool(json_required),
                            "strict_json_requested": bool(use_strict_json),
                            "max_output_tokens": max_tokens,
                            "temperature": temperature,
                            "provider_elapsed_seconds": round(provider_elapsed, 6),
                            "attempt_elapsed_seconds": round(time.perf_counter() - attempt_start, 6),
                            "logical_elapsed_seconds": round(time.perf_counter() - logical_start, 6),
                            "retry_sleep_seconds": round(retry_sleep_seconds, 6),
                            "post_attempt_delay_seconds": round(post_attempt_delay_seconds, 6),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cached_input_tokens": provider_meta.get("cached_input_tokens", 0),
                            "reasoning_output_tokens": provider_meta.get("reasoning_output_tokens", 0),
                            "usage_source": usage_source or "unavailable",
                            "error_type": error.__class__.__name__ if error else None,
                            "error_message": str(error)[:500] if error else None,
                            **request_meta,
                            **(telemetry_context or {}),
                        }
                    )

            if error is None:
                return response
            if not retry_planned:
                break
        
        return None
    
    def call_model(
        self, 
        messages: List[Dict], 
        max_tokens: int = 1200,
        json_required: bool = False, 
        response_model=None,
        use_strict_json: bool = True,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Core model calling logic - routes to provider-specific implementation.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            json_required: Whether to force JSON response format
            response_model: Optional Pydantic model for structured output
            model: Override model name (uses default if None)
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            **kwargs: Additional parameters to pass to the model
        """
        return self._call_impl(
            messages=messages,
            max_tokens=max_tokens,
            json_required=json_required,
            response_model=response_model,
            use_strict_json=use_strict_json,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs
        )
    
    def _call_azure(
        self,
        messages: List[Dict],
        max_tokens: int,
        json_required: bool,
        response_model,
        use_strict_json: bool,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        **kwargs
    ) -> Any:
        """Azure OpenAI implementation.
        
        GPT-4o: JSON mode only in this repo.
        GPT-5-mini: Supports structured outputs via json_schema.
        """
        model_params = self._build_openai_chat_params(
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

        use_json_schema = self.config.structured_output_mode == "json_schema"
        if response_model and use_strict_json and use_json_schema:
            schema_params = dict(model_params)
            schema_params["response_format"] = self._build_json_schema_format(response_model)
            print(f"\n[DEBUG_RESPONSE_FORMAT] Azure path: Attempting strict json_schema")
            try:
                response = self.client.chat.completions.create(**schema_params)
                self._capture_response_metadata(response)
                print(f"\n[DEBUG_MODEL_EXTRACT] Azure API resolved json_schema call to exact model: {getattr(response, 'model', 'Unknown')}")
                content = self._extract_chat_content(response)
                if not content:
                    finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown') if hasattr(response, 'choices') else 'unknown'
                    refusal = getattr(response.choices[0].message, 'refusal', None) if hasattr(response, 'choices') else None
                    print(f"[WARN] Azure model returned empty content. finish_reason={finish_reason}, refusal={refusal}")
                    return None
                return self._parse_structured_content(content, response_model, "Azure OpenAI")
            except Exception as exc:
                if not self._schema_error_should_fallback(exc):
                    raise
                print(f"\n[DEBUG_RESPONSE_FORMAT] Azure json_schema rejected. Safely falling back to json_object... Error: {str(exc)[:150]}")
                # Fall through to the json_object block below

        if json_required or response_model:
            fallback_params = dict(model_params)
            fallback_params["response_format"] = {"type": "json_object"}
            print(f"\n[DEBUG_RESPONSE_FORMAT] Azure path: Using basic '{fallback_params['response_format']['type']}'")
            return self._call_openai_json_mode(fallback_params, response_model=response_model)

        response = self.client.chat.completions.create(**model_params)
        self._capture_response_metadata(response)
        print(f"\n[DEBUG_MODEL_EXTRACT] Azure API resolved this call to exact model: {getattr(response, 'model', 'Unknown')}")
        content = self._extract_chat_content(response)
        return content

    def _call_openai(
        self,
        messages: List[Dict],
        max_tokens: int,
        json_required: bool,
        response_model,
        use_strict_json: bool,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        **kwargs
    ) -> Any:
        """Direct OpenAI implementation via Chat Completions.

        Preferred behavior:
        - response_model => json_schema first, then json_object fallback once
        - json_required => json_object
        - plain text => standard completion
        """
        model_params = self._build_openai_chat_params(
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

        use_json_schema = self.config.structured_output_mode == "json_schema"
        if response_model and use_strict_json and use_json_schema:
            schema_params = dict(model_params)
            schema_params["response_format"] = self._build_json_schema_format(response_model)
            try:
                response = self.client.chat.completions.create(**schema_params)
                self._capture_response_metadata(response)
                content = self._extract_chat_content(response)
                return self._parse_structured_content(content, response_model, "Direct OpenAI")
            except Exception as exc:
                if not self._schema_error_should_fallback(exc):
                    raise
                logging.warning(
                    "OpenAI-compatible json_schema rejected; retrying once with json_object fallback: %s",
                    exc,
                )
                fallback_params = dict(model_params)
                fallback_params["response_format"] = {"type": "json_object"}
                return self._call_openai_json_mode(fallback_params, response_model=response_model)

        if json_required or response_model:
            model_params["response_format"] = {"type": "json_object"}
            return self._call_openai_json_mode(model_params, response_model=response_model)

        response = self.client.chat.completions.create(**model_params)
        self._capture_response_metadata(response)
        print(f"\n[DEBUG_MODEL_EXTRACT] OpenAI-compatible API resolved this call to exact model: {getattr(response, 'model', 'Unknown')}")
        return self._extract_chat_content(response)

    def _call_gemini(
        self,
        messages: List[Dict],
        max_tokens: int,
        json_required: bool,
        response_model,
        model: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],  # Not used for Gemini
        presence_penalty: Optional[float],   # Not used for Gemini
        **kwargs
    ) -> Any:
        """Google GenAI (Gemini) implementation.
        
        Gemini 2.5 Pro supports:
        - thinking_config for internal reasoning
        - response_mime_type="application/json" for JSON output
        - response_schema for structured output with Pydantic schema
        All three can be used together.
        """
        from google.genai.types import GenerateContentConfig, ThinkingConfig
        
        model_name = model or self.model
        
        # Convert OpenAI-style messages to Gemini format
        # Returns (contents, system_instruction) tuple
        contents, system_instruction = self._convert_messages_to_gemini(messages)
        
        # Build generation config
        config_params = {
            "max_output_tokens": max_tokens,
        }
        
        if temperature is not None:
            config_params["temperature"] = temperature
        if top_p is not None:
            config_params["top_p"] = top_p
        
        # Add system instruction if present (proper Gemini API usage)
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        
        # Add thinking config for Gemini 2.5 Pro (min budget is 128)
        # This works WITH response_mime_type and response_schema
        config_params["thinking_config"] = ThinkingConfig(
            include_thoughts=False,  # Don't include thoughts in response - they consume output and cause empty answers
            thinking_budget=self.thinking_budget
        )
        
        # Handle JSON mode / structured output
        # Logic:
        # - json_required=True: Use response_mime_type for JSON output (prompt-based schema)
        # - response_model provided: Use response_schema for native structured output
        
        if response_model:
            # Native structured output with schema enforcement
            # Pass Pydantic model directly - SDK handles schema conversion
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_model
        elif json_required:
            # JSON mode only (like GPT-4o) - no schema enforcement
            config_params["response_mime_type"] = "application/json"
        
        try:
            gen_config = GenerateContentConfig(**config_params)
            
            # Make the API call
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config
            )
        except Exception as e:
            # Fallback for models with lower token limits (e.g., 8192) if requested max_tokens was higher
            error_str = str(e)
            # Check for common error signatures related to invalid arguments / max output tokens
            is_invalid_arg = "InvalidArgument" in error_str or "400" in error_str or "max_output_tokens" in error_str
            current_max = config_params.get("max_output_tokens", 0)
            
            if is_invalid_arg and current_max > 8192:
                print(f"[WARN] Gemini max_output_tokens={current_max} failed. Retrying with 8192 output limit.")
                config_params["max_output_tokens"] = 8192
                gen_config = GenerateContentConfig(**config_params)
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )
            else:
                raise e

        self._capture_response_metadata(response)
        
        # Check for truncation (finish_reason = LENGTH or MAX_TOKENS)
        # This helps diagnose why JSON might be incomplete
        for candidate in response.candidates:
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                finish_str = str(finish_reason).upper()
                if finish_str in ('LENGTH', 'MAX_TOKENS', '2'):
                    print(f"[WARN] Gemini response truncated (finish_reason={finish_reason}). Consider increasing max_tokens.")
        
        # Extract text from response (skip thought parts, get answer only)
        answer_content = []
        thought_content = []
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not candidate.content or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                print(f"[DEBUG] Gemini candidate has no content/parts: {candidate}")
                continue
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    is_thought = getattr(part, 'thought', False)
                    if is_thought:
                        thought_content.append(part.text)
                    else:
                        answer_content.append(part.text)
        
        # Debug: log what we got
        if not answer_content and thought_content:
            print(f"[DEBUG] Gemini returned {len(thought_content)} thought parts but NO answer content!")
        
        result_text = "".join(answer_content) if answer_content else getattr(response, 'text', '')
        
        # Parse and validate against schema if provided
        if response_model:
            try:
                parsed = json.loads(result_text)
                return response_model(**parsed)
            except json.JSONDecodeError as e:
                # More informative error for truncation diagnosis
                print(f"[WARN] Gemini structured output validation failed: {e}")
                preview = result_text[:200] + "..." if len(result_text) > 200 else result_text
                print(f"[DEBUG] Response preview (truncated?): {preview}")
                return result_text
            except Exception as e:
                print(f"[WARN] Gemini structured output validation failed: {e}")
                return result_text
        else:
            return result_text
    
    def _convert_messages_to_gemini(self, messages: List[Dict]) -> tuple:
        """Convert OpenAI-style messages to Gemini format.
        
        OpenAI format: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Gemini format: [{"role": "user"|"model", "parts": [{"text": "..."}]}]
        
        Returns:
            Tuple of (contents, system_instruction)
            - contents: List of Content objects for the conversation
            - system_instruction: Combined system prompts as string (or None)
        """
        from google.genai.types import Content, Part
        
        gemini_contents = []
        system_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            
            if role == "system":
                # Collect all system messages
                if content.strip():
                    system_parts.append(content)
            elif role == "user":
                gemini_contents.append(
                    Content(role="user", parts=[Part(text=content)])
                )
            elif role == "assistant":
                gemini_contents.append(
                    Content(role="model", parts=[Part(text=content)])
                )
        
        # Combine all system prompts into one instruction
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        
        return gemini_contents, system_instruction

