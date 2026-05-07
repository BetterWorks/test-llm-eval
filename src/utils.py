"""
Utility functions for the evaluation framework.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional


def extract_json_from_response(text: str) -> tuple[Optional[dict], bool, bool]:
    """
    Extract JSON from LLM response text.
    
    Returns:
        tuple of (parsed_json, has_markdown_wrapper, is_valid_json)
    """
    if not text:
        return None, False, False
    
    # Check for markdown code block wrapper
    has_markdown = bool(re.search(r'```(?:json)?\s*\n', text))
    
    # Try to extract JSON from markdown blocks
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        json_text = text.strip()
    
    # Try to parse JSON
    try:
        parsed = json.loads(json_text)
        return parsed, has_markdown, True
    except json.JSONDecodeError:
        return None, has_markdown, False


def cache_key_for_url(url: str) -> str:
    """Generate a cache key for a URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    # Replace spaces and special chars with underscores
    safe = re.sub(r'[^\w\-.]', '_', name)
    # Remove duplicate underscores
    safe = re.sub(r'_+', '_', safe)
    return safe.lower()


def format_score(score: Optional[float], precision: int = 3) -> str:
    """Format a score for display."""
    if score is None:
        return "N/A"
    return f"{score:.{precision}f}"


def calculate_pass_rate(passed: int, total: int, eval_errors: int = 0) -> float:
    """Calculate pass rate excluding eval errors."""
    denominator = total - eval_errors
    return passed / denominator if denominator > 0 else 0.0
