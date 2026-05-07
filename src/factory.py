"""
Factory pattern for creating prompt builders for different features.
All builders now fetch prompts dynamically from GitHub.
"""
from src.prompt_builder import (
    WritingAssistantPromptBuilder,
    FeedbackSummaryPromptBuilder,
    GoalAssistPromptBuilder,
    PerformanceSummaryPromptBuilder,
    MeetingsSummaryPromptBuilder,
    SkillsDiscoveryPromptBuilder
)


class PromptBuilderFactory:
    """Factory for creating prompt builders based on feature."""
    
    @staticmethod
    def create(feature: str):
        """Create appropriate prompt builder for the feature."""
        if feature == "writing_assistant":
            return WritingAssistantPromptBuilder()
        elif feature == "feedback_summary":
            return FeedbackSummaryPromptBuilder()
        elif feature == "goal_assist":
            return GoalAssistPromptBuilder()
        elif feature == "performance_summary":
            return PerformanceSummaryPromptBuilder()
        elif feature == "meetings_summary":
            return MeetingsSummaryPromptBuilder()
        elif feature == "skills_discovery":
            return SkillsDiscoveryPromptBuilder()
        else:
            raise ValueError(f"Unknown feature: {feature}")
