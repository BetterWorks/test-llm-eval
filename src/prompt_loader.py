"""
Prompt loader for fetching prompts from Betterworks/llm-engine repository.
Dynamically loads prompt files based on feature and model type.
"""
import ast
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


# Map our feature names to llm-engine prompt directory names
FEATURE_MAPPING = {
    "writing_assistant": "writing_assistant",
    "feedback_summary": "feedback_summary",
    "goal_assist": "goals_assistant",  # Note: different name in llm-engine
    "performance_summary": "performance_summary",
    "meetings_summary": "meetings_summary",
    "skills_discovery": "skill",  # Note: different name in llm-engine
}


class PromptLoader:
    """Loads prompts from llm-engine GitHub repository."""
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_dir = ensure_dir(Path(".cache/prompts"))
        
    def _get_prompt_file_for_model(self, model: str) -> str:
        """Determine which prompt file to use based on model."""
        model_lower = model.lower()
        
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai.py"
        elif "mistral" in model_lower:
            return "mistral.py"
        elif "meta" in model_lower or "llama" in model_lower or "gemma" in model_lower:
            # Gemma is Google's model but follows Meta's format
            return "meta.py"
        else:
            # Default to meta for unknown models
            logger.warning(f"Unknown model type {model}, defaulting to meta.py")
            return "meta.py"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _fetch_from_github(self, feature: str, prompt_file: str) -> str:
        """Fetch prompt file content from GitHub."""
        # Map feature name to llm-engine directory name
        llm_engine_feature = FEATURE_MAPPING.get(feature, feature)
        
        url = f"https://api.github.com/repos/{config.github.org}/{config.github.llm_engine_repo}/contents/app/prompt/{llm_engine_feature}/{prompt_file}"
        
        logger.info(f"Fetching prompt from GitHub: {llm_engine_feature}/{prompt_file}")
        
        response = requests.get(
            url,
            headers=config.github.headers,
            timeout=30
        )
        response.raise_for_status()
        
        # GitHub API returns content as base64
        content_b64 = response.json()["content"]
        content = base64.b64decode(content_b64).decode("utf-8")
        
        return content
    
    def _load_from_cache(self, feature: str, prompt_file: str) -> Optional[str]:
        """Load prompt file from cache if available."""
        llm_engine_feature = FEATURE_MAPPING.get(feature, feature)
        cache_file = self.cache_dir / llm_engine_feature / prompt_file
        
        if not cache_file.exists():
            return None
        
        logger.info(f"Loading prompt from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _save_to_cache(self, feature: str, prompt_file: str, content: str) -> None:
        """Save prompt file content to cache."""
        llm_engine_feature = FEATURE_MAPPING.get(feature, feature)
        cache_dir = ensure_dir(self.cache_dir / llm_engine_feature)
        cache_file = cache_dir / prompt_file
        
        logger.info(f"Saving prompt to cache: {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _extract_prompt_constants(self, content: str) -> Dict[str, str]:
        """Extract prompt string constants from Python file using AST parsing."""
        prompts = {}
        
        try:
            # Parse the Python file content
            tree = ast.parse(content)
            
            # Find all top-level assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Check if it's assigning to a variable (not tuple unpacking, etc.)
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        
                        # Only process uppercase constant names
                        if var_name.isupper():
                            # Get the value - it could be a string or a joined string
                            value = self._get_string_value(node.value)
                            if value is not None:
                                prompts[var_name] = value
        
        except SyntaxError as e:
            logger.error(f"Syntax error parsing prompts file: {e}")
            # Fall back to the old method if AST parsing fails
            return self._extract_prompt_constants_fallback(content)
        
        logger.info(f"Extracted {len(prompts)} prompt constants: {list(prompts.keys())}")
        return prompts
    
    def _get_string_value(self, node) -> Optional[str]:
        """Extract string value from AST node, handling string concatenation."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Simple string constant
            return node.value
        elif isinstance(node, ast.JoinedStr):
            # f-string - concatenate parts
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
            return ''.join(parts) if parts else None
        elif hasattr(ast, 'Str') and isinstance(node, ast.Str):
            # Python <3.8 compatibility
            return node.s
        else:
            # Not a string
            return None
    
    def _extract_prompt_constants_fallback(self, content: str) -> Dict[str, str]:
        """Fallback method using regex - handles simpler cases."""
        prompts = {}
        
        # Match patterns like: CONSTANT_NAME = '''...'''  or CONSTANT_NAME = ('''...''')
        # This regex captures multi-line strings
        pattern = r"^([A-Z_]+)\s*=\s*(?:\()?\s*'''(.*?)'''\s*(?:\))?"
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for var_name, value in matches:
            prompts[var_name] = value
        
        logger.info(f"Extracted {len(prompts)} prompt constants using fallback method")
        return prompts
    
    def load_prompts(self, feature: str, model: Optional[str] = None, force_refresh: bool = False) -> Dict[str, str]:
        """
        Load prompts for a specific feature from llm-engine repository.
        
        Args:
            feature: Feature name (writing_assistant, feedback_summary, goal_assist, etc.)
            model: Model name to determine which prompt file to use (defaults to config)
            force_refresh: Skip cache and fetch from GitHub
            
        Returns:
            Dictionary of prompt constant names to their values
        """
        if model is None:
            model = config.endpoint.model
        
        prompt_file = self._get_prompt_file_for_model(model)
        
        # Special handling for skills_discovery - load from all subdirectories
        if feature == "skills_discovery":
            return self._load_skill_prompts(prompt_file, force_refresh)
        
        # Try cache first
        if self.cache_enabled and not force_refresh:
            cached = self._load_from_cache(feature, prompt_file)
            if cached:
                return self._extract_prompt_constants(cached)
        
        # Fetch from GitHub
        content = self._fetch_from_github(feature, prompt_file)
        
        # Save to cache
        if self.cache_enabled:
            self._save_to_cache(feature, prompt_file, content)
        
        return self._extract_prompt_constants(content)
    
    def _load_skill_prompts(self, prompt_file: str, force_refresh: bool = False) -> Dict[str, str]:
        """Load prompts from skill subdirectories (conversation, feedback, job_description, job_title)."""
        all_prompts = {}
        subdirs = ["conversation", "feedback", "job_description", "job_title"]
        
        for subdir in subdirs:
            # Try cache first
            cache_key = f"skill/{subdir}"
            if self.cache_enabled and not force_refresh:
                cached = self._load_from_cache(cache_key, prompt_file)
                if cached:
                    all_prompts.update(self._extract_prompt_constants(cached))
                    continue
            
            # Fetch from GitHub
            llm_engine_feature = "skill"
            url = f"https://api.github.com/repos/{config.github.org}/{config.github.llm_engine_repo}/contents/app/prompt/{llm_engine_feature}/{subdir}/{prompt_file}"
            
            logger.info(f"Fetching prompt from GitHub: skill/{subdir}/{prompt_file}")
            
            try:
                response = requests.get(url, headers=config.github.headers, timeout=30)
                response.raise_for_status()
                content_b64 = response.json()["content"]
                content = base64.b64decode(content_b64).decode("utf-8")
                
                # Save to cache
                if self.cache_enabled:
                    cache_dir = ensure_dir(self.cache_dir / "skill" / subdir)
                    cache_file = cache_dir / prompt_file
                    logger.info(f"Saving prompt to cache: {cache_file}")
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                all_prompts.update(self._extract_prompt_constants(content))
            except Exception as e:
                logger.warning(f"Failed to load prompts from skill/{subdir}: {e}")
                continue
        
        return all_prompts


def get_prompt_loader() -> PromptLoader:
    """Get a cached prompt loader instance."""
    if not hasattr(get_prompt_loader, "_instance"):
        get_prompt_loader._instance = PromptLoader(cache_enabled=config.benchmark.cache_enabled)
    return get_prompt_loader._instance
