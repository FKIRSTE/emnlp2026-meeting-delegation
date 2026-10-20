"""
> Supports multiple LLM providers.
> Reads from config_extended.json and selects active model based on:
    1. MD_ACTIVE_MODEL environment variable (highest priority)
    2. "active_model" field in config file
    3. Default: "gpt-4o"
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Extended configuration loader supporting multiple LLM providers.
    
    Drop-in replacement for the original Config class.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize config from JSON file.
        
        Args:
            config_path: Path to config JSON file. If None, auto-detects location.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        config_path = Path(config_path)
        self.loaded_dotenv_path: Optional[str] = None

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._load_dotenv_file(config_path)

        with open(config_path, 'r') as f:
            self.data = json.load(f)
        
        # Determine active model from env var or config default
        self.active_model_name = os.environ.get(
            "MD_ACTIVE_MODEL", 
            self.data.get("active_model", "gpt-4o")
        )
        
        # Check if this is an extended config (has "models" key) or legacy config
        if "models" in self.data:
            # Extended config format
            if self.active_model_name not in self.data["models"]:
                available = list(self.data["models"].keys())
                raise ValueError(
                    f"Model '{self.active_model_name}' not found in config. "
                    f"Available: {available}"
                )
            self._model_config = self.data["models"][self.active_model_name]
            self._is_extended = True
        else:
            # Legacy config format (single model) - backwards compatible
            self._model_config = self.data
            self._is_extended = False
    
    def _load_dotenv_file(self, config_path: Path) -> None:
        """Load the first discovered .env file without overriding real env vars."""
        try:
            from dotenv import load_dotenv
        except ImportError:
            return

        candidates = [
            Path.cwd() / ".env",
            config_path.parent / ".env",
            config_path.parent.parent / ".env",
        ]

        seen = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                load_dotenv(candidate, override=False)
                self.loaded_dotenv_path = str(candidate.resolve())
                return

    def _find_config_file(self) -> Path:
        """
        Find the config file by looking in common locations.
        Prefers config_extended.json if it exists.
        
        Returns:
            Path to the config file
        """
        current_dir = Path(__file__).parent
        
        # Try extended config first, then fall back to legacy
        possible_locations = [
            current_dir / "config_extended.json",
            current_dir.parent / "config_extended.json",
            current_dir / "config_gpt.json",
            current_dir.parent / "config_gpt.json",
            Path.cwd() / "config_extended.json",
            Path.cwd() / "src" / "config_extended.json",
            Path.cwd() / "config_gpt.json",
            Path.cwd() / "src" / "config_gpt.json",
        ]
        
        for location in possible_locations:
            if location.exists():
                return location
        
        # Default fallback
        return current_dir / "config_extended.json"
    
    @property
    def provider(self) -> str:
        """LLM provider: 'azure_openai', 'openai', or 'google_genai'."""
        return self._model_config.get("provider", "azure_openai")

    def _resolve_model_value(self, key: str, default: str = "") -> str:
        """Resolve a model config value with optional env-var indirection."""
        env_key = self._model_config.get(f"{key}_env")
        if env_key:
            env_value = os.environ.get(env_key)
            if env_value:
                return env_value
        return self._model_config.get(key, default)
    
    @property
    def api_key(self) -> str:
        """API key for the active model."""
        return self._resolve_model_value("api_key")
    
    @property
    def api_version(self) -> str:
        """Azure OpenAI API version (only for azure_openai provider)."""
        return self._resolve_model_value("api_version", "2024-02-01")
    
    @property
    def endpoint(self) -> str:
        """Azure OpenAI endpoint (only for azure_openai provider)."""
        return self._resolve_model_value("endpoint")
    
    @property
    def base_url(self) -> str:
        """Base URL for direct OpenAI or Google GenAI if configured."""
        return self._resolve_model_value("base_url")

    @property
    def api_mode(self) -> str:
        """Provider transport mode. Direct OpenAI defaults to chat completions."""
        return self._model_config.get("api_mode", "chat_completions")

    @property
    def structured_output_mode(self) -> str:
        """Preferred structured-output mode for OpenAI-compatible providers."""
        return self._model_config.get("structured_output_mode", "json_schema")

    @property
    def default_request_params(self) -> Dict[str, Any]:
        """Provider/model-specific request defaults merged into every call."""
        value = self._model_config.get("default_request_params", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def use_system_ca(self) -> bool:
        """Use the operating-system certificate store for this provider."""
        return bool(self._model_config.get("use_system_ca", False))
    
    @property
    def model(self) -> str:
        """Model name (e.g., gpt-4o, gpt-5, gemini-2.5-pro)."""
        return self._model_config["model"]
    
    @property
    def thinking_budget(self) -> int:
        """Gemini thinking budget (only for google_genai provider)"""
        return self._model_config.get("thinking_budget", 128)
    
    def get(self, key: str, default=None):
        """Get any config value with optional default.
        
        Checks model-specific config first, then top-level config.
        """
        # Check model config first
        if key in self._model_config:
            return self._model_config.get(key, default)
        # Then check top-level config
        return self.data.get(key, default)

    def get_section(self, key: str) -> Dict[str, Any]:
        """Get a nested config section as a dict (returns {} if missing/non-dict)."""
        val = self.data.get(key, {})
        return val if isinstance(val, dict) else {}

    @property
    def capa(self) -> Dict[str, Any]:
        """CAPA-specific configuration section (optional)."""
        return self.get_section("capa")
    
    def __repr__(self):
        provider_info = f"provider={self.provider}" if self._is_extended else ""
        return f"Config(model={self.model}, {provider_info}, active={self.active_model_name})"
