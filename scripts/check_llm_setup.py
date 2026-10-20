"""
LLM Connection Test Script

This script tests the basic connectivity to the LLM service and helps diagnose
configuration issues before running the full data preparation pipeline.

Usage:
    python scripts/check_llm_setup.py
"""

import sys
import json
from pathlib import Path
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

# Add src to path for imports - works from any directory
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from config import Config
from llm_client import LLMClient


def test_config_loading():
    """Test if configuration loads correctly."""
    print("🔍 TESTING CONFIGURATION LOADING")
    print("=" * 50)
    
    try:
        # Test config loading through Config class (auto-discovers file)
        config = Config()
        print(f"✅ Config loaded successfully (Active model: {config.active_model_name})")
        
        # Check required fields
        print(f"\n📋 Configuration Details:")
        provider = getattr(config, 'provider', 'Not specified')
        print(f"   Provider: {provider}")
        print(f"   Model: {config.model}")
        
        if hasattr(config, 'api_version') and config.api_version:
            print(f"   API Version: {config.api_version}")
            
        if hasattr(config, 'endpoint') and config.endpoint:
            print(f"   Endpoint: '{config.endpoint}'")
            
        if hasattr(config, 'base_url') and config.base_url:
            print(f"   Base URL: '{config.base_url}'")
            
        print(f"   API Key: {'*' * (len(config.api_key) - 4) + config.api_key[-4:] if config.api_key and config.api_key != 'xxx' else 'MISSING'}")
        
        # Check for common issues
        issues = []
        if provider == "azure_openai" and (not config.endpoint or config.endpoint.strip() == ""):
            issues.append("❌ Endpoint is empty or missing (required for Azure)")
        
        if not config.api_key or config.api_key.strip() == "" or config.api_key == "xxx":
            issues.append("❌ API key is empty, missing, or placeholder ('xxx')")
        elif len(config.api_key) < 10:
            issues.append("⚠️ API key seems too short")
        
        if not config.model:
            issues.append("❌ Model is not specified")
            
        if issues:
            print(f"\n🚨 Configuration Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            return False
        else:
            print(f"\n✅ Configuration appears valid")
            return True
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_llm_client_initialization():
    """Test LLM client initialization."""
    print("\n🔍 TESTING LLM CLIENT INITIALIZATION")
    print("=" * 50)
    
    try:
        config = Config()
        
        # Try to initialize the client
        llm_client = LLMClient(config)
        print(f"✅ LLM Client initialized successfully")
        print(f"   Provider: {llm_client.provider}")
        print(f"   Using model: {llm_client.model}")
        return llm_client
        
    except Exception as e:
        print(f"❌ LLM Client initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return None


def test_simple_llm_call(llm_client):
    """Test a simple LLM API call."""
    print("\n🔍 TESTING SIMPLE LLM API CALL")
    print("=" * 50)
    
    if not llm_client:
        print("❌ Cannot test API call - LLM client not available")
        return False
    
    try:
        # Simple test message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, world!' and nothing else."}
        ]
        
        print("📤 Sending test message...")
        print("   System: You are a helpful assistant.")
        print("   User: Say 'Hello, world!' and nothing else.")
        
        # Make the API call with minimal parameters
        response = llm_client.call_with_retry(
            messages=test_messages,
            max_tokens=50,
            max_attempts=1,  # Only try once for testing
            temperature=0.1
        )
        
        print(f"📥 Raw response type: {type(response)}")
        print(f"📥 Raw response: {response}")
        
        if response:
            # The LLM client returns just the content string for standard calls
            if isinstance(response, str):
                response_text = response
                print(f"📥 Response content: '{response_text}'")
                
                if "hello" in response_text.lower():
                    print(f"✅ API call successful - response contains expected content")
                    return True
                else:
                    print(f"✅ API call successful - got response (content different than expected)")
                    return True
            else:
                print(f"⚠️ Unexpected response type: {type(response)}")
                return False
        else:
            print(f"❌ API call returned None or empty response")
            return False
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # More detailed exception information
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()
        
        # Provide specific guidance based on error type
        error_str = str(e).lower()
        if "connection" in error_str:
            print(f"\n💡 Connection Error Diagnosis:")
            print(f"   - Check if endpoint URL is correct and accessible")
            print(f"   - Verify network connectivity")
            print(f"   - Ensure firewall/proxy allows HTTPS connections")
        elif "authentication" in error_str or "401" in error_str:
            print(f"\n💡 Authentication Error Diagnosis:")
            print(f"   - Verify API key is correct and active")
            print(f"   - Check if API key has proper permissions")
        elif "not found" in error_str or "404" in error_str:
            print(f"\n💡 Resource Not Found Diagnosis:")
            print(f"   - Check if model name is correct")
            print(f"   - Verify endpoint URL points to correct resource")
        
        return False


class TestActionType(str, Enum):
    SPEAK = "Speak"
    SILENT = "Remain_Silent"
    ASK_QUESTION = "Ask_Question"

class TestConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class Insight(BaseModel):
    insight_text: str = Field(description="Brief text of the insight")
    relevance_score: int = Field(description="Score between 1 and 100")

class CandidateEvalTest(BaseModel):
    candidate_id: str = Field(description="The ID of the candidate evaluated")
    confidence_level: TestConfidenceLevel = Field(description="The level of confidence in this evaluation")
    rationale: str = Field(description="Why this candidate was assigned this confidence")
    key_insights: List[Insight] = Field(description="A list of 1 to 3 key insights about this candidate")

class ActionDetailsTest(BaseModel):
    action_name: TestActionType = Field(description="The final chosen action")
    scope: str = Field(description="The intended scope, e.g., 'Concise' or 'Comprehensive'")
    justification_points: List[str] = Field(description="A list of bullet points justifying the chosen action")

class TestMetadata(BaseModel):
    source_system: str = Field(description="The system creating this output")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    tags: List[str] = Field(description="Optional tags about the meeting")

class ComplexStructuredOutputTest(BaseModel):
    meeting_id: str = Field(description="The ID of the fake meeting")
    is_urgent: bool = Field(description="Whether the response is flagged as urgent")
    action_details: ActionDetailsTest = Field(description="Decision details")
    active_candidate_id: Optional[str] = Field(default=None, description="The chosen candidate ID if speaking")
    evaluations: List[CandidateEvalTest] = Field(description="List of candidate evaluations")
    metadata: TestMetadata = Field(description="Metadata about the execution")

def test_structured_llm_call(llm_client):
    """Test a complex structured LLM API call."""
    print("\n🔍 TESTING CONCRETE STRUCTURED OUTPUTS (JSON_SCHEMA)")
    print("=" * 50)
    
    if not llm_client:
        print("❌ Cannot test API call - LLM client not available")
        return False
    
    try:
        test_messages = [
            {
                "role": "system", 
                "content": "You are a test agent evaluating candidates for a hypothetical meeting called 'M-1234'. You must strictly output the data in the requested JSON structure."
            },
            {
                "role": "user", 
                "content": (
                    "Evaluate two hypothetical candidates ('C-Alpha' and 'C-Beta').\n"
                    "C-Alpha is highly relevant with a score of 95, so we should Speak and choose C-Alpha. "
                    "C-Beta is irrelevant with a score of 12. "
                    "Make up exactly two bullet points of justification for the action. "
                    "Set processing_time_ms to 42 and system to 'Pytest-Agent'."
                )
            }
        ]
        
        print("📤 Sending deep-nested structured test prompt...")
        
        response = llm_client.call_with_retry(
            messages=test_messages,
            max_tokens=600,
            max_attempts=1,
            temperature=0.1,
            response_model=ComplexStructuredOutputTest
        )
        
        if response:
            print(f"📥 Raw response type: {type(response)}")
            if isinstance(response, ComplexStructuredOutputTest):
                print(f"✅ API call successful - Pydantic model returned natively!")
                print("📥 Parsed deep properties:")
                print(f"   Meeting ID: {response.meeting_id}")
                print(f"   Action Name (Enum): {response.action_details.action_name.value}")
                print(f"   Active Candidate: {response.active_candidate_id}")
                print(f"   Justifications: {len(response.action_details.justification_points)} bullets")
                print(f"   Evaluations Count: {len(response.evaluations)}")
                print(f"   Meta processing time: {response.metadata.processing_time_ms} ms")
                print("   [Candidate Dump]:")
                for e in response.evaluations:
                    print(f"     - ID: {e.candidate_id} | Confidence: {e.confidence_level.value} | Insights: {len(e.key_insights)}")
                return True
            else:
                print(f"⚠️ Returned response was not parsed into the Pydantic model natively.")
                print(f"Response dump: {response}")
                return False
        else:
            print(f"❌ Structured API call returned None or empty response")
            return False
            
    except Exception as e:
        print(f"❌ Structured API call failed: {e}")
        return False


def test_llm_connection():
    """Run complete LLM connection test suite."""
    print("🧪 LLM CONNECTION TEST SUITE")
    print("=" * 60)
    print("This test will help diagnose LLM connectivity issues")
    print("=" * 60)
    
    # Test 1: Configuration Loading
    config_ok = test_config_loading()
    
    if not config_ok:
        print(f"\n❌ CONFIGURATION TEST FAILED")
        print(f"Please fix configuration issues before proceeding.")
        return False
    
    # Test 2: LLM Client Initialization  
    llm_client = test_llm_client_initialization()
    
    if not llm_client:
        print(f"\n❌ CLIENT INITIALIZATION TEST FAILED")
        return False
    
    # Test 3: Simple API Call
    api_ok = test_simple_llm_call(llm_client)
    
    # Test 4: Structured API Call
    structured_ok = test_structured_llm_call(llm_client)
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"🏁 TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    if config_ok and llm_client and api_ok and structured_ok:
        print(f"✅ All tests passed! LLM connection and structured outputs are working correctly.")
        print(f"📋 Your setup is ready for the data preparation pipeline.")
    elif config_ok and llm_client:
        print(f"⚠️ Configuration and client work, but some API calls failed.")
        print(f"📋 Check the logs above for specific test failures.")
    else:
        print(f"❌ Multiple issues detected. Please fix configuration first.")
    
    print(f"{'='*60}")
    
    return config_ok and llm_client and api_ok and structured_ok


if __name__ == "__main__":
    test_llm_connection()
