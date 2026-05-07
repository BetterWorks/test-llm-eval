"""
Dataset loader for golden datasets from Betterworks/llm-golden-datasets.
Fetches datasets from GitHub and caches them locally.
"""
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import requests
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from src.utils import cache_key_for_url, ensure_dir

logger = logging.getLogger(__name__)


class TestCase(BaseModel):
    """Base test case model matching golden dataset schema."""
    
    test_id: str
    timestamps: dict
    metadata: dict
    evaluation_criteria: str
    min_pass_score: float
    prompt: str
    response: str
    system_prompt: Optional[str] = None
    task_type: str
    difficulty: Optional[str] = None


class WritingAssistantTestCase(TestCase):
    """Writing assistant specific test case."""
    
    @property
    def context(self) -> Optional[str]:
        return self.metadata.get("context")
    
    @property
    def tone(self) -> Optional[str]:
        return self.metadata.get("tone")
    
    @property
    def is_richtext(self) -> bool:
        return self.metadata.get("is_richtext", False)
    
    @property
    def language(self) -> Optional[str]:
        return self.metadata.get("language", "en")
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")


class GoalAssistTestCase(TestCase):
    """Goal assist specific test case."""
    
    @property
    def conversation_type(self) -> Optional[str]:
        return self.metadata.get("conversation_type")
    
    @property
    def num_feedback_items(self) -> int:
        return self.metadata.get("num_feedback_items", 0)
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")


class FeedbackSummaryTestCase(TestCase):
    """Feedback summary specific test case."""
    
    @property
    def sentiment(self) -> Optional[str]:
        return self.metadata.get("sentiment")
    
    @property
    def num_items(self) -> int:
        return self.metadata.get("num_items", 0)
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")


class PerformanceSummaryTestCase(TestCase):
    """Performance summary specific test case."""
    
    @property
    def mode(self) -> Optional[str]:
        return self.metadata.get("mode")
    
    @property
    def performance_level(self) -> Optional[str]:
        return self.metadata.get("performance_level")
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")
    
    @property
    def user_name(self) -> Optional[str]:
        return self.metadata.get("user_name")


class MeetingsSummaryTestCase(TestCase):
    """Meetings summary specific test case."""
    
    @property
    def relationship_type(self) -> Optional[str]:
        return self.metadata.get("relationship_type")
    
    @property
    def n_meetings(self) -> int:
        return self.metadata.get("n_meetings", 0)
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")
    
    @property
    def participant_1(self) -> Optional[str]:
        return self.metadata.get("participant_1")
    
    @property
    def participant_2(self) -> Optional[str]:
        return self.metadata.get("participant_2")


class SkillsDiscoveryTestCase(TestCase):
    """Skills discovery specific test case."""
    
    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source")
    
    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")
    
    @property
    def department(self) -> Optional[str]:
        return self.metadata.get("department")
    
    @property
    def skills(self) -> list:
        return self.metadata.get("skills", [])
    
    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")


class GuardrailTestCase(BaseModel):
    """Guardrail test case with parsed prompt fields."""

    test_id: str
    timestamps: dict
    metadata: dict
    prompt: str  # JSON-serialized {text, guardrails, direction, context_type}
    response: Optional[Any] = None  # null | {"error": {...}} | {"warning": {...}}
    system_prompt: Optional[str] = None
    task_type: str = "classify"
    difficulty: Optional[str] = None
    evaluation_criteria: str = ""
    min_pass_score: float = 0.0

    # Parsed from prompt JSON
    input_text: str = ""
    guardrails_requested: List[str] = Field(default_factory=list)
    direction: str = "input"
    context_type: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _parse_prompt_json(self) -> "GuardrailTestCase":
        try:
            data = json.loads(self.prompt)
            self.input_text = data.get("text", "")
            self.guardrails_requested = data.get("guardrails", [])
            self.direction = data.get("direction", "input")
            self.context_type = data.get("context_type", "")
        except (json.JSONDecodeError, TypeError):
            pass
        return self

    @property
    def expected_outcome(self) -> str:
        return self.metadata.get("outcome", "clean")

    @property
    def expected_triggered_categories(self) -> List[str]:
        return self.metadata.get("triggered_categories", [])

    @property
    def locale(self) -> str:
        return self.metadata.get("locale", "en")

    @property
    def edge_case_type(self) -> Optional[str]:
        return self.metadata.get("edge_case_type")

    @property
    def category(self) -> str:
        return self.metadata.get("category", "")


class DatasetLoader:
    """Loads golden datasets from GitHub with caching."""
    
    def __init__(
        self,
        dataset_name: str,
        version: Optional[str] = None,
        split: str = "test",
    ):
        self.dataset_name = dataset_name
        # Get feature-specific dataset version from config if not specified
        self.version = version or config.benchmark.get_dataset_version(dataset_name)
        self.split = split
        self.cache_dir = ensure_dir(config.benchmark.cache_dir / "datasets" / dataset_name / f"v{self.version}")
        
    def _build_github_url(self) -> str:
        """Build GitHub raw content URL for dataset file."""
        base = f"https://raw.githubusercontent.com/{config.github.org}/{config.github.golden_datasets_repo}"
        path = f"main/datasets/{self.dataset_name}/v{self.version}/{self.split}.jsonl"
        return f"{base}/{path}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _fetch_from_github(self, url: str) -> str:
        """Fetch dataset content from GitHub."""
        logger.info(f"Fetching dataset from GitHub: {url}")
        
        response = requests.get(
            url,
            headers=config.github.headers,
            timeout=30
        )
        response.raise_for_status()
        return response.text
    
    def _load_from_cache(self) -> Optional[list[dict]]:
        """Load dataset from cache if available."""
        cache_file = self.cache_dir / f"{self.split}.jsonl"
        
        if not cache_file.exists():
            return None
        
        logger.info(f"Loading dataset from cache: {cache_file}")
        records = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        return records
    
    def _save_to_cache(self, content: str) -> None:
        """Save dataset content to cache."""
        cache_file = self.cache_dir / f"{self.split}.jsonl"
        logger.info(f"Saving dataset to cache: {cache_file}")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def load(self, force_refresh: bool = False) -> list[TestCase]:
        """
        Load dataset from cache or GitHub.
        
        Args:
            force_refresh: Skip cache and fetch from GitHub
            
        Returns:
            List of test cases
        """
        # Try cache first
        if config.benchmark.cache_enabled and not force_refresh:
            cached = self._load_from_cache()
            if cached:
                logger.info(f"Loaded {len(cached)} test cases from cache")
                return self._parse_records(cached)
        
        # Fetch from GitHub
        url = self._build_github_url()
        content = self._fetch_from_github(url)
        
        # Parse records
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        
        # Save to cache
        if config.benchmark.cache_enabled:
            self._save_to_cache(content)
        
        logger.info(f"Loaded {len(records)} test cases from GitHub")
        return self._parse_records(records)
    
    def _parse_records(self, records: list[dict]) -> list[TestCase]:
        """Parse raw records into TestCase objects."""
        test_cases = []
        for record in records:
            try:
                # Use specific test case class if available
                if self.dataset_name == "writing_assistant":
                    test_cases.append(WritingAssistantTestCase(**record))
                elif self.dataset_name == "goal_assist":
                    test_cases.append(GoalAssistTestCase(**record))
                elif self.dataset_name == "feedback_summary":
                    test_cases.append(FeedbackSummaryTestCase(**record))
                elif self.dataset_name == "performance_summary":
                    test_cases.append(PerformanceSummaryTestCase(**record))
                elif self.dataset_name == "meetings_summary":
                    test_cases.append(MeetingsSummaryTestCase(**record))
                elif self.dataset_name == "skills_discovery":
                    test_cases.append(SkillsDiscoveryTestCase(**record))
                elif self.dataset_name == "guardrails":
                    test_cases.append(GuardrailTestCase(**record))
                else:
                    test_cases.append(TestCase(**record))
            except Exception as e:
                logger.warning(f"Failed to parse test case {record.get('test_id')}: {e}")
                continue
        
        return test_cases
    
    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        test_cases = self.load()
        
        stats = {
            "total": len(test_cases),
            "by_difficulty": {},
            "by_task_type": {},
        }
        
        for tc in test_cases:
            # Count by difficulty
            difficulty = tc.difficulty or "unknown"
            stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1
            
            # Count by task type
            stats["by_task_type"][tc.task_type] = stats["by_task_type"].get(tc.task_type, 0) + 1
        
        # Writing assistant specific stats
        if self.dataset_name == "writing_assistant":
            stats["by_context"] = {}
            stats["by_tone"] = {}
            
            for tc in test_cases:
                if isinstance(tc, WritingAssistantTestCase):
                    context = tc.context or "unknown"
                    stats["by_context"][context] = stats["by_context"].get(context, 0) + 1
                    
                    tone = tc.tone or "unknown"
                    stats["by_tone"][tone] = stats["by_tone"].get(tone, 0) + 1
        
        return stats


def load_writing_assistant_dataset(
    version: str = "1.2",
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[WritingAssistantTestCase]:
    """
    Convenience function to load writing assistant dataset.
    
    Args:
        version: Dataset version (default: 1.2)
        split: Dataset split (train/val/test)
        max_cases: Maximum number of cases to load (for testing)
        
    Returns:
        List of writing assistant test cases
    """
    loader = DatasetLoader("writing_assistant", version=version, split=split)
    test_cases = loader.load()
    
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    return test_cases


def load_goal_assist_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[GoalAssistTestCase]:
    """Convenience function to load goal assist dataset."""
    loader = DatasetLoader("goal_assist", version=version, split=split)
    test_cases = loader.load()
    
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    return test_cases


def load_feedback_summary_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[FeedbackSummaryTestCase]:
    """Convenience function to load feedback summary dataset."""
    loader = DatasetLoader("feedback_summary", version=version, split=split)
    test_cases = loader.load()
    
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    return test_cases


def load_performance_summary_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[PerformanceSummaryTestCase]:
    """Convenience function to load performance summary dataset."""
    loader = DatasetLoader("performance_summary", version=version, split=split)
    test_cases = loader.load()
    
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    return test_cases


def load_meetings_summary_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[MeetingsSummaryTestCase]:
    """Convenience function to load meetings summary dataset."""
    loader = DatasetLoader("meetings_summary", version=version, split=split)
    test_cases = loader.load()
    
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    return test_cases


def load_skills_discovery_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[SkillsDiscoveryTestCase]:
    """Convenience function to load skills discovery dataset."""
    loader = DatasetLoader("skills_discovery", version=version, split=split)
    test_cases = loader.load()

    if max_cases:
        test_cases = test_cases[:max_cases]

    return test_cases


def load_guardrails_dataset(
    version: Optional[str] = None,
    split: str = "test",
    max_cases: Optional[int] = None,
) -> list[GuardrailTestCase]:
    """Convenience function to load guardrails dataset."""
    loader = DatasetLoader("guardrails", version=version, split=split)
    test_cases = loader.load()

    if max_cases:
        test_cases = test_cases[:max_cases]

    return test_cases
