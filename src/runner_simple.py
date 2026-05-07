"""
Simple runner for collecting LLM responses without evaluation.
Outputs responses in CSV, Markdown, and JSONL formats.
"""
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from config import config
from src.dataset_loader import (
    TestCase,
    load_writing_assistant_dataset,
    load_goal_assist_dataset,
    load_feedback_summary_dataset,
    load_performance_summary_dataset,
    load_meetings_summary_dataset,
    load_skills_discovery_dataset,
)
from src.endpoint_client import EndpointClient, EndpointError
from src.factory import PromptBuilderFactory
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


class ResponseArtifact(BaseModel):
    """Artifact for a single LLM response."""
    
    test_id: str
    feature: str
    input_text: str
    llm_output: str
    model: str
    elapsed_time: float
    timestamp: str
    metadata: dict = {}
    error: Optional[str] = None


class ResponseCollector:
    """Collects LLM responses and saves them in multiple formats."""
    
    def __init__(
        self,
        feature: str = "writing_assistant",
        dataset_version: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.feature = feature
        self.dataset_version = dataset_version or config.benchmark.get_dataset_version(feature)
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.benchmark.output_dir / feature / f"run_{timestamp}"
        
        self.output_dir = ensure_dir(output_dir)
        
        # Initialize components
        self.endpoint_client = EndpointClient()
        self.prompt_builder = PromptBuilderFactory.create(feature)
        
        # Runtime state
        self.artifacts: list[ResponseArtifact] = []
        self.test_cases: list[TestCase] = []
        
    async def run_single_test(
        self,
        test_case: TestCase,
    ) -> ResponseArtifact:
        """
        Execute a single test case and collect response.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            Response artifact
        """
        logger.info(f"Running test case: {test_case.test_id}")
        
        error = None
        llm_output = ""
        elapsed_time = 0.0
        model = config.endpoint.model
        
        try:
            # Extract locale from test case (default to "en" if not present)
            user_locale = "en"
            if hasattr(test_case, 'locale'):
                user_locale = test_case.locale
            
            # Build messages with locale
            messages = self.prompt_builder.build_messages(test_case, user_locale=user_locale)
            
            # Call LLM endpoint
            llm_response = self.endpoint_client.call(
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
            
            llm_output = llm_response.get("content", "")
            elapsed_time = llm_response.get("elapsed_time", 0.0)
            model = llm_response.get("model", model)
            
            logger.info(f"{test_case.test_id}: Response received ({elapsed_time:.2f}s)")
            
        except EndpointError as e:
            error = str(e)
            logger.error(f"{test_case.test_id}: Endpoint error - {error}")
        except Exception as e:
            error = str(e)
            logger.error(f"{test_case.test_id}: Unexpected error - {error}")
        
        # Create artifact
        artifact = ResponseArtifact(
            test_id=test_case.test_id,
            feature=self.feature,
            input_text=test_case.prompt,
            llm_output=llm_output,
            model=model,
            elapsed_time=elapsed_time,
            timestamp=datetime.now().isoformat(),
            metadata=test_case.metadata if hasattr(test_case, 'metadata') else {},
            error=error,
        )
        
        self.artifacts.append(artifact)
        
        return artifact
    
    async def run_all_tests(
        self,
        max_cases: Optional[int] = None,
        max_concurrent: int = 3,
    ) -> dict:
        """
        Run all test cases with concurrency control.
        
        Args:
            max_cases: Maximum number of cases to run (for testing)
            max_concurrent: Maximum concurrent executions
            
        Returns:
            Summary statistics
        """
        # Load dataset
        logger.info(f"Loading {self.feature} dataset v{self.dataset_version}")
        
        # Map features to their loader functions
        loaders = {
            "writing_assistant": load_writing_assistant_dataset,
            "goal_assist": load_goal_assist_dataset,
            "feedback_summary": load_feedback_summary_dataset,
            "performance_summary": load_performance_summary_dataset,
            "meetings_summary": load_meetings_summary_dataset,
            "skills_discovery": load_skills_discovery_dataset,
        }
        
        loader_fn = loaders.get(self.feature)
        if not loader_fn:
            raise ValueError(f"Unknown feature: {self.feature}")
        
        self.test_cases = loader_fn(
            version=self.dataset_version,
            split="test",
            max_cases=max_cases,
        )
        
        logger.info(f"Loaded {len(self.test_cases)} test cases")
        
        # Run tests with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(tc):
            async with semaphore:
                return await self.run_single_test(tc)
        
        logger.info(f"Starting collection with max_concurrent={max_concurrent}")
        
        await asyncio.gather(*[run_with_semaphore(tc) for tc in self.test_cases])
        
        # Save outputs
        self._save_jsonl()
        self._save_csv()
        self._save_markdown()
        
        # Calculate summary
        total = len(self.artifacts)
        errors = sum(1 for a in self.artifacts if a.error is not None)
        success = total - errors
        avg_time = sum(a.elapsed_time for a in self.artifacts) / total if total > 0 else 0.0
        
        summary = {
            "total": total,
            "success": success,
            "errors": errors,
            "avg_elapsed_time": avg_time,
        }
        
        logger.info(
            f"Collection complete: {success}/{total} successful "
            f"(avg time: {avg_time:.2f}s)"
        )
        
        return summary
    
    def _save_jsonl(self) -> None:
        """Save artifacts to JSONL file."""
        output_file = self.output_dir / "responses.jsonl"
        
        logger.info(f"Saving {len(self.artifacts)} responses to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for artifact in self.artifacts:
                f.write(artifact.model_dump_json() + '\n')
    
    def _save_csv(self) -> None:
        """Save artifacts to CSV file without truncation."""
        output_file = self.output_dir / "responses.csv"
        
        logger.info(f"Saving {len(self.artifacts)} responses to {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_number', 'test_id', 'feature', 'category', 
                'input_text', 'input_text_length',
                'llm_output', 'output_length',
                'model', 'execution_time_seconds', 
                'ai_response_time_seconds', 'timestamp', 
                'success', 'error_message'
            ])
            writer.writeheader()
            
            for idx, artifact in enumerate(self.artifacts, 1):
                # Extract category from metadata
                category = artifact.metadata.get('category', artifact.metadata.get('description', ''))
                
                writer.writerow({
                    'test_number': idx,
                    'test_id': artifact.test_id,
                    'feature': artifact.feature,
                    'category': category,
                    'input_text': artifact.input_text,  # FULL TEXT - NO TRUNCATION
                    'input_text_length': len(artifact.input_text),
                    'llm_output': artifact.llm_output,  # FULL TEXT - NO TRUNCATION
                    'output_length': len(artifact.llm_output) if artifact.llm_output else 0,
                    'model': artifact.model,
                    'execution_time_seconds': f"{artifact.elapsed_time:.2f}",
                    'ai_response_time_seconds': f"{artifact.elapsed_time:.2f}",  # Same as execution for now
                    'timestamp': artifact.timestamp,
                    'success': not bool(artifact.error),
                    'error_message': artifact.error or '',
                })

    
    def _save_markdown(self) -> None:
        """Generate markdown report."""
        output_file = self.output_dir / "responses.md"
        
        logger.info(f"Generating markdown report: {output_file}")
        
        total = len(self.artifacts)
        errors = sum(1 for a in self.artifacts if a.error is not None)
        success = total - errors
        avg_time = sum(a.elapsed_time for a in self.artifacts) / total if total > 0 else 0.0
        
        report = f"""# {self.feature.title().replace('_', ' ')} - LLM Responses

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** {config.endpoint.model}  
**Endpoint:** {config.endpoint.url}  
**Dataset Version:** v{self.dataset_version}  
**Total Cases:** {total}  
**Successful:** {success}  
**Errors:** {errors}  
**Avg Response Time:** {avg_time:.2f}s

---

## Responses

| Test ID | Input (truncated) | Output (truncated) | Time (s) | Status |
|---------|-------------------|-------------------|----------|--------|
"""
        
        for artifact in self.artifacts:
            input_preview = artifact.input_text[:50].replace('\n', ' ').replace('|', '\\|')
            if len(artifact.input_text) > 50:
                input_preview += '...'
            
            output_preview = artifact.llm_output[:80].replace('\n', ' ').replace('|', '\\|')
            if len(artifact.llm_output) > 80:
                output_preview += '...'
            
            status = '❌ Error' if artifact.error else '✅ Success'
            
            report += f"| {artifact.test_id} | {input_preview} | {output_preview} | {artifact.elapsed_time:.2f} | {status} |\n"
        
        # Add error details if any
        errors_list = [a for a in self.artifacts if a.error]
        if errors_list:
            report += "\n---\n\n## Errors\n\n"
            for artifact in errors_list:
                report += f"### {artifact.test_id}\n\n"
                report += f"**Error:** {artifact.error}\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Markdown report saved to {output_file}")
