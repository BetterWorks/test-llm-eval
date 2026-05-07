"""
Configuration management for LLM evaluation framework.
Loads configuration from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file
load_dotenv()


class GitHubConfig(BaseModel):
    """GitHub repository configuration."""
    
    token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    org: str = Field(default_factory=lambda: os.getenv("GITHUB_ORG", "Betterworks"))
    llm_engine_repo: str = Field(default_factory=lambda: os.getenv("LLMENGINE_REPO", "llm-engine"))
    llm_proxy_repo: str = Field(default_factory=lambda: os.getenv("LLM_PROXY_REPO", "llm-proxy"))
    golden_datasets_repo: str = Field(default_factory=lambda: os.getenv("GOLDEN_DATASETS_REPO", "llm-golden-datasets"))
    
    @property
    def llm_engine_url(self) -> str:
        return f"https://api.github.com/repos/{self.org}/{self.llm_engine_repo}"
    
    @property
    def golden_datasets_url(self) -> str:
        return f"https://api.github.com/repos/{self.org}/{self.golden_datasets_repo}"
    
    @property
    def headers(self) -> dict:
        """GitHub API headers with authentication."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers


class EndpointConfig(BaseModel):
    """LLM endpoint configuration."""
    
    url: str = Field(default_factory=lambda: os.getenv("ENDPOINT_URL", "http://10.50.21.45:8000/v1/chat/completions"))
    model: str = Field(default_factory=lambda: os.getenv("ENDPOINT_MODEL", "google/gemma-4-31B-it"))
    timeout: int = Field(default=120)
    max_retries: int = Field(default=3)


# Removed JudgeConfig - no evaluation needed


class BenchmarkConfig(BaseModel):
    """Benchmark execution configuration."""
    
    max_concurrent: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT", "3")))
    # Per-feature dataset versions
    writing_assistant_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_WRITING_ASSISTANT", "1.2"))
    goal_assist_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_GOAL_ASSIST", "1.0"))
    feedback_summary_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_FEEDBACK_SUMMARY", "1.0"))
    performance_summary_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_PERFORMANCE_SUMMARY", "1.1"))
    meetings_summary_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_MEETINGS_SUMMARY", "1.0"))
    skills_discovery_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_SKILLS_DISCOVERY", "1.1"))
    guardrails_version: str = Field(default_factory=lambda: os.getenv("DATASET_VERSION_GUARDRAILS", "1.1"))
    cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    cache_enabled: bool = Field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "artifacts")))
    
    def get_dataset_version(self, feature: str) -> str:
        """Get dataset version for a specific feature."""
        version_map = {
            "writing_assistant": self.writing_assistant_version,
            "goal_assist": self.goal_assist_version,
            "feedback_summary": self.feedback_summary_version,
            "performance_summary": self.performance_summary_version,
            "meetings_summary": self.meetings_summary_version,
            "skills_discovery": self.skills_discovery_version,
            "guardrails": self.guardrails_version,
        }
        return version_map.get(feature, "1.0")


class Config(BaseModel):
    """Main configuration container."""
    
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.github.token:
            issues.append("GITHUB_TOKEN is not set - GitHub API rate limits will apply")
        
        if not self.endpoint.url:
            issues.append("ENDPOINT_URL is not set")
        
        if not self.endpoint.model:
            issues.append("ENDPOINT_MODEL is not set")
        
        return issues


# Global config instance
config = Config()
