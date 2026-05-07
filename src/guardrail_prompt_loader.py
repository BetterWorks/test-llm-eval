"""
Prompt loader for guardrail prompts from Betterworks/llm-proxy repository.
Loads safety (selfcheck), PII, and bias detection prompts.
"""
import base64
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# llm-proxy file paths for each guardrail type
GUARDRAIL_PROMPT_FILES: Dict[str, str] = {
    "safety": "src/api/prompts/selfcheck.py",
    "pii": "src/api/prompts/pii.py",
    "bias": "src/api/prompts/bias.py",
}


class GuardrailPromptLoader:
    """Loads safety, PII, and bias prompts from Betterworks/llm-proxy."""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache_dir = ensure_dir(Path(".cache/prompts/guardrails"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _fetch_from_github(self, file_path: str) -> str:
        """Fetch a file from the llm-proxy repository via GitHub API."""
        url = (
            f"https://api.github.com/repos/{config.github.org}/"
            f"{config.github.llm_proxy_repo}/contents/{file_path}"
        )
        logger.info(f"Fetching guardrail prompt: {file_path}")

        response = requests.get(url, headers=config.github.headers, timeout=30)
        response.raise_for_status()

        content_b64 = response.json()["content"]
        return base64.b64decode(content_b64).decode("utf-8")

    def _load_from_cache(self, guardrail_type: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{guardrail_type}.py"
        if not cache_file.exists():
            return None
        logger.info(f"Loading guardrail prompt from cache: {cache_file}")
        return cache_file.read_text(encoding="utf-8")

    def _save_to_cache(self, guardrail_type: str, content: str) -> None:
        cache_file = self.cache_dir / f"{guardrail_type}.py"
        logger.info(f"Caching guardrail prompt: {cache_file}")
        cache_file.write_text(content, encoding="utf-8")

    def _extract_prompt_constants(self, content: str) -> Dict[str, str]:
        """Extract uppercase string constants from a Python source file."""
        prompts: Dict[str, str] = {}
        parts = re.split(r"('''|\"\"\")", content)

        current_var: Optional[str] = None
        in_string = False
        string_content: list = []

        for i, part in enumerate(parts):
            if part in ("'''", '"""'):
                if not in_string:
                    if i > 0:
                        match = re.search(
                            r"([A-Z_]+)\s*=\s*(?:\(\s*)?$", parts[i - 1].strip()
                        )
                        if match:
                            current_var = match.group(1)
                            in_string = True
                            string_content = []
                else:
                    if current_var:
                        prompts[current_var] = "".join(string_content)
                        current_var = None
                    in_string = False
                    string_content = []
            elif in_string:
                string_content.append(part)

        logger.info(f"Extracted {len(prompts)} constants: {list(prompts.keys())}")
        return prompts

    def load_all_prompts(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Load all guardrail prompts from llm-proxy.

        Returns a flat dict of prompt constant names → prompt text, e.g.:
            INPUT_CHECK_PROMPT, OUTPUT_CHECK_PROMPT  (safety)
            PII_DETECTION_PROMPT                     (pii)
            BIAS_DETECTION_PROMPT                    (bias)
        """
        all_prompts: Dict[str, str] = {}

        for guardrail_type, file_path in GUARDRAIL_PROMPT_FILES.items():
            if self.cache_enabled and not force_refresh:
                cached = self._load_from_cache(guardrail_type)
                if cached:
                    all_prompts.update(self._extract_prompt_constants(cached))
                    continue

            content = self._fetch_from_github(file_path)

            if self.cache_enabled:
                self._save_to_cache(guardrail_type, content)

            all_prompts.update(self._extract_prompt_constants(content))

        return all_prompts
