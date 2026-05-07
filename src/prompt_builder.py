"""
Prompt builders for constructing LLM prompts dynamically from GitHub.
Fetches prompts from llm-engine repository instead of hardcoding them.
"""
import logging
from typing import Optional

from src.prompt_loader import get_prompt_loader

logger = logging.getLogger(__name__)


class WritingAssistantPromptBuilder:
    """Build prompts for writing assistant feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("writing_assistant")
        logger.info(f"Loaded {len(self.prompts)} writing_assistant prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for writing assistant feature.
        
        Args:
            test_case: WritingAssistantTestCase with metadata (tone, is_richtext, etc.)
            user_locale: Language locale for output
            
        Returns:
            List of message dictionaries for LLM API
        """
        # Extract metadata from test case
        tone = test_case.tone or "base"
        is_richtext = test_case.is_richtext
        text_input = test_case.prompt
        module_name = test_case.metadata.get("module_name", "text")
        
        # Select tone-specific prompt
        tone_prompts = {
            "professional": self.prompts.get("PROFESSIONAL_REPHRASE_PROMPT", ""),
            "casual": self.prompts.get("CASUAL_REPHRASE_PROMPT", ""),
            "longer": self.prompts.get("LONGER_REPHRASE_PROMPT", ""),
            "shorter": self.prompts.get("SHORTER_REPHRASE_PROMPT", ""),
            "base": self.prompts.get("BASE_REPHRASE_PROMPT", ""),
        }
        tone_prompt = tone_prompts.get(tone.lower(), tone_prompts["base"])
        
        # Select HTML handling prompt
        if is_richtext:
            text_handling_prompt = self.prompts.get("HTML_HANDLING_PROMPT", "")
        else:
            text_handling_prompt = self.prompts.get("NORMAL_TEXT_HANDLING_PROMPT", "")
        
        # Build system prompt
        base_prompt = self.prompts.get("BASE_PROMPT", "")
        instruction_prompt = self.prompts.get("INSTRUCTION_PROMPT", "")
        
        # Replace placeholders
        base_prompt = base_prompt.replace("{module_name}", module_name)
        base_prompt = base_prompt.replace("{user_locale}", user_locale)
        
        instruction_prompt = instruction_prompt.replace("{text_or_html_handling_prompt}", text_handling_prompt)
        instruction_prompt = instruction_prompt.replace("{tone_prompt}", tone_prompt)
        instruction_prompt = instruction_prompt.replace("{user_locale}", user_locale)
        
        system_message = base_prompt + "\n" + instruction_prompt
        
        # Build messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text_input}
        ]
        
        return messages


class FeedbackSummaryPromptBuilder:
    """Build prompts for feedback summary feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("feedback_summary")
        logger.info(f"Loaded {len(self.prompts)} feedback_summary prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for feedback summary feature.
        
        Uses FS_INITIAL_AGGREGATION_PROMPT as the system prompt.
        """
        # Use initial aggregation prompt as default
        system_prompt = self.prompts.get("FS_INITIAL_AGGREGATION_PROMPT", "")
        
        # Replace user name placeholder if available
        user_name = test_case.metadata.get("user_name", "the employee")
        user_name_prompt = f"'{user_name}'"
        system_prompt = system_prompt.replace("{user_name_prompt}", user_name_prompt)
        system_prompt = system_prompt.replace("{user_locale}", user_locale)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.prompt}
        ]
        
        return messages


class GoalAssistPromptBuilder:
    """Build prompts for goal assist feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("goal_assist")
        logger.info(f"Loaded {len(self.prompts)} goal_assist prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for goal assist feature.
        
        Uses appropriate context based on goal type in metadata.
        """
        # Determine which context to use based on metadata
        goal_type = test_case.metadata.get("goal_type", "mix").lower()
        
        if goal_type == "development":
            context_prompt = self.prompts.get("GOAL_ASSIST_DEVELOPMENT_CONTEXT", "")
        elif goal_type == "business":
            context_prompt = self.prompts.get("GOAL_ASSIST_BUSINESS_CONTEXT", "")
        else:  # mix or default
            context_prompt = self.prompts.get("GOAL_ASSIST_MIX_CONTEXT", "")
        
        # Add common context
        common_context = self.prompts.get("GOAL_ASSIST_COMMON_CONTEXT", "")
        
        # Combine context and common instructions
        system_prompt = context_prompt + "\n\n" + common_context
        
        # Replace placeholders
        goal_focus_nudge = test_case.metadata.get("goal_focus_nudge", "business or developmental")
        input_sources = test_case.metadata.get("input_sources", "user input, job title, department")
        
        system_prompt = system_prompt.replace("{goal_focus_nudge}", goal_focus_nudge)
        system_prompt = system_prompt.replace("{input_sources}", input_sources)
        system_prompt = system_prompt.replace("{user_locale}", user_locale)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.prompt}
        ]
        
        return messages


class PerformanceSummaryPromptBuilder:
    """Build prompts for performance summary feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("performance_summary")
        logger.info(f"Loaded {len(self.prompts)} performance_summary prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for performance summary feature.
        
        Uses PS_INITIAL_AGGREGATION_PROMPT as the system prompt.
        """
        # Check mode to determine which prompt to use
        mode = test_case.metadata.get("mode", "full")
        
        if mode == "only_strengths":
            system_prompt = self.prompts.get("PS_INITIAL_AGGREGATION_PROMPT_ONLY_STRENGTHS", "")
        else:
            system_prompt = self.prompts.get("PS_INITIAL_AGGREGATION_PROMPT", "")
        
        # Replace user name placeholder if available
        user_name = test_case.metadata.get("user_name", "the employee")
        user_name_prompt = f"'{user_name}'"
        system_prompt = system_prompt.replace("{user_name_prompt}", user_name_prompt)
        system_prompt = system_prompt.replace("{user_locale}", user_locale)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.prompt}
        ]
        
        return messages


class MeetingsSummaryPromptBuilder:
    """Build prompts for meetings summary feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("meetings_summary")
        logger.info(f"Loaded {len(self.prompts)} meetings_summary prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for meetings summary feature.
        
        Uses MS_INITIAL_AGGREGATION_PROMPT as the system prompt.
        """
        # Use initial aggregation prompt as default
        system_prompt = self.prompts.get("MS_INITIAL_AGGREGATION_PROMPT", "")
        
        # Replace participant placeholders if available
        participant_1 = test_case.metadata.get("participant_1", "Participant 1")
        participant_2 = test_case.metadata.get("participant_2", "Participant 2")
        
        system_prompt = system_prompt.replace("{participant_1}", participant_1)
        system_prompt = system_prompt.replace("{participant_2}", participant_2)
        system_prompt = system_prompt.replace("{user_locale}", user_locale)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.prompt}
        ]
        
        return messages


class SkillsDiscoveryPromptBuilder:
    """Build prompts for skills discovery feature using llm-engine prompts."""
    
    def __init__(self):
        """Initialize prompt builder and load prompts from GitHub."""
        self.prompts = get_prompt_loader().load_prompts("skills_discovery")
        logger.info(f"Loaded {len(self.prompts)} skills_discovery prompts from GitHub")
    
    def build_messages(
        self,
        test_case,
        user_locale: str = "en",
    ) -> list[dict]:
        """
        Build messages for skills discovery feature.
        
        Uses appropriate prompt based on source in metadata.
        """
        # Determine which prompt to use based on source
        source = test_case.metadata.get("source", "conversation").lower()
        
        prompt_map = {
            "conversation": "SKILL_DISCOVERY_CONVERSATION_CONTEXT_PROMPT",
            "feedback": "SKILL_DISCOVERY_FEEDBACK_CONTEXT_PROMPT",
            "job_description": "SKILL_DISCOVERY_JOB_DESCRIPTION_CONTEXT_PROMPT",
            "job_title": "SKILL_DISCOVERY_JOB_TITLE_CONTEXT_PROMPT",
        }
        
        prompt_key = prompt_map.get(source, "SKILL_DISCOVERY_CONVERSATION_CONTEXT_PROMPT")
        system_prompt = self.prompts.get(prompt_key, "")
        
        # Replace placeholders - skills discovery uses {locale} instead of {user_locale}
        system_prompt = system_prompt.replace("{locale}", user_locale)
        system_prompt = system_prompt.replace("{user_locale}", user_locale)  # fallback
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_case.prompt}
        ]
        
        return messages
