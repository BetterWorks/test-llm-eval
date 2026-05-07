"""
Endpoint client for calling the LLM API.
Supports OpenAI-compatible endpoints with retry logic.
"""
import logging
import time
from typing import Any, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import config

logger = logging.getLogger(__name__)


class EndpointError(Exception):
    """Exception raised for endpoint errors."""
    pass


class EndpointClient:
    """Client for OpenAI-compatible LLM endpoints."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self.url = url or config.endpoint.url
        self.model = model or config.endpoint.model
        self.timeout = timeout or config.endpoint.timeout
        
    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        stop=stop_after_attempt(config.endpoint.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def call(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Call the LLM endpoint.
        
        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Response dict containing the completion
            
        Raises:
            EndpointError: If the API call fails
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        start_time = time.time()
        
        try:
            logger.debug(f"Calling endpoint: {self.url} with model: {self.model}")
            
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Endpoint response received in {elapsed:.2f}s")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content from OpenAI-compatible response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                
                return {
                    "content": content,
                    "raw_response": result,
                    "elapsed_time": elapsed,
                    "model": self.model,
                }
            else:
                raise EndpointError(f"Unexpected response format: {result}")
                
        except requests.Timeout as e:
            logger.error(f"Endpoint timeout after {self.timeout}s")
            raise EndpointError(f"Request timeout: {e}")
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise EndpointError(f"HTTP {e.response.status_code}: {e.response.text}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise EndpointError(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise EndpointError(f"Unexpected error: {e}")
    
    def test_connection(self) -> bool:
        """
        Test if the endpoint is reachable.
        
        Returns:
            True if endpoint responds successfully
        """
        try:
            response = self.call(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                max_tokens=10,
            )
            return "content" in response
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
