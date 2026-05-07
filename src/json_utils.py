"""
JSON extraction utilities from LLM responses.
Based on llm-engine benchmark framework's json_utils.py.
"""
import re
from typing import Tuple


def extract_json_with_flags(content: str) -> Tuple[str, bool, bool]:
    """Extract JSON from an LLM response string.

    Returns:
        (json_string, has_markdown_wrapper, has_closing_backticks)
    
    Extraction patterns (in order of precedence):
    1. Fully fenced: ```json ... ```
    2. Opening ```json without closing fence
    3. Any fenced code block with closing fence
    4. Any fenced code block without closing fence
    5. First JSON object in text
    """
    # Fully fenced ```json ... ```
    m = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
    if m:
        return m.group(1).strip(), True, True

    # Opening ```json without closing fence (rest of document)
    m = re.search(r'```json\s*\n(.*)$', content, re.DOTALL)
    if m:
        extracted = m.group(1).strip()
        if extracted.startswith(('{', '[')):
            return extracted, True, False

    # Any fenced code block with closing fence (``` or ```lang)
    m = re.search(r'```\w*\n(.*?)\n```', content, re.DOTALL)
    if m:
        potential = m.group(1).strip()
        if potential.startswith(('{', '[')):
            return potential, True, True

    # Any fenced code block without closing fence
    m = re.search(r'```\w*\n(.*)$', content, re.DOTALL)
    if m:
        extracted = m.group(1).strip()
        if extracted.startswith(('{', '[')):
            return extracted, True, False

    # First JSON object in text
    m = re.search(r'(\{[\s\S]*\})', content, re.DOTALL)
    if m:
        return m.group(1).strip(), False, False

    return content.strip(), False, False
