"""
Command-line interface for LLM evaluation framework.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from config import config
from src.runner_simple import ResponseCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)

logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LLM Evaluation Framework - Test AI features against golden datasets."""
    pass


@cli.command()
@click.option(
    "--feature",
    type=click.Choice(["writing_assistant", "goal_assist", "feedback_summary", "performance_summary", "meetings_summary", "skills_discovery"], case_sensitive=False),
    default="writing_assistant",
    help="Feature to evaluate",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config per feature)",
)
@click.option(
    "--max-cases",
    type=int,
    default=None,
    help="Maximum number of test cases to run (for testing)",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Maximum concurrent requests (default: 3)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def run(
    feature: str,
    dataset_version: Optional[str],
    max_cases: int,
    max_concurrent: int,
    output_dir: Path,
    verbose: bool,
):
    """Collect LLM responses for a feature."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    console.print("\n[bold blue]🔧 Configuration Validation[/bold blue]")
    issues = config.validate()
    
    if issues:
        console.print("[yellow]⚠️  Configuration warnings:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✓ Configuration valid[/green]")
    
    # Display configuration
    table = Table(title="Response Collection Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Feature", feature)
    table.add_row("Dataset Version", dataset_version or config.benchmark.get_dataset_version(feature))
    table.add_row("Max Cases", str(max_cases) if max_cases else "All")
    table.add_row("Max Concurrent", str(max_concurrent or config.benchmark.max_concurrent))
    table.add_row("Endpoint", config.endpoint.url)
    table.add_row("Model", config.endpoint.model)
    
    console.print(table)
    console.print()
    
    # Run collection
    try:
        collector = ResponseCollector(
            feature=feature,
            dataset_version=dataset_version,
            output_dir=output_dir,
        )
        
        max_concurrent = max_concurrent or config.benchmark.max_concurrent
        
        console.print(f"[bold green]🚀 Starting response collection...[/bold green]\n")
        
        summary = asyncio.run(
            collector.run_all_tests(
                max_cases=max_cases,
                max_concurrent=max_concurrent,
            )
        )
        
        # Display results
        console.print(f"\n[bold green]✅ Collection Complete![/bold green]\n")
        
        results_table = Table(title="Results Summary", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Total Cases", str(summary["total"]))
        results_table.add_row("Successful", f"{summary['success']} ({summary['success']/summary['total']*100:.1f}%)" if summary["total"] > 0 else "0")
        results_table.add_row("Errors", str(summary["errors"]))
        results_table.add_row("Avg Response Time", f"{summary['avg_elapsed_time']:.2f}s")
        
        console.print(results_table)
        
        console.print(f"\n📁 Results saved to: {collector.output_dir}")
        console.print(f"  • [cyan]responses.csv[/cyan] - CSV format")
        console.print(f"  • [cyan]responses.md[/cyan] - Markdown report")
        console.print(f"  • [cyan]responses.jsonl[/cyan] - JSONL with full details\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Collection interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
def test_connection():
    """Test connection to the LLM endpoint."""
    from src.endpoint_client import EndpointClient
    
    console.print("\n[bold blue]🔌 Testing endpoint connection...[/bold blue]")
    console.print(f"URL: {config.endpoint.url}")
    console.print(f"Model: {config.endpoint.model}\n")
    
    try:
        client = EndpointClient()
        
        with console.status("[bold green]Sending test request..."):
            success = client.test_connection()
        
        if success:
            console.print("[bold green]✅ Connection successful![/bold green]")
        else:
            console.print("[bold red]❌ Connection failed[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--feature",
    type=click.Choice(["writing_assistant", "goal_assist", "feedback_summary", "performance_summary", "meetings_summary", "skills_discovery"], case_sensitive=False),
    default="writing_assistant",
    help="Feature to show info for",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config per feature)",
)
def dataset_info(feature: str, dataset_version: Optional[str]):
    """Display dataset information and statistics."""
    from src.dataset_loader import DatasetLoader
    
    console.print(f"\n[bold blue]📊 Dataset Information[/bold blue]")
    console.print(f"Feature: {feature}\n")
    
    try:
        loader = DatasetLoader(feature, version=dataset_version, split="test")
        console.print(f"Version: v{loader.version}")
        
        with console.status("[bold green]Loading dataset..."):
            stats = loader.get_statistics()
        
        # Main stats
        table = Table(title="Dataset Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Test Cases", str(stats["total"]))
        
        for task_type, count in stats["by_task_type"].items():
            table.add_row(f"  {task_type}", str(count))
        
        console.print(table)
        
        # Difficulty breakdown
        if "by_difficulty" in stats:
            console.print("\n[bold]By Difficulty:[/bold]")
            for difficulty, count in sorted(stats["by_difficulty"].items()):
                console.print(f"  • {difficulty}: {count}")
        
        # Feature-specific stats
        if "by_context" in stats:
            console.print("\n[bold]By Context:[/bold]")
            for context, count in sorted(stats["by_context"].items()):
                console.print(f"  • {context}: {count}")
        
        if "by_tone" in stats:
            console.print("\n[bold]By Tone:[/bold]")
            for tone, count in sorted(stats["by_tone"].items()):
                console.print(f"  • {tone}: {count}")
        
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def config_check():
    """Check configuration and display current settings."""
    console.print("\n[bold blue]🔧 Configuration Check[/bold blue]\n")
    
    # GitHub config
    console.print("[bold]GitHub Configuration:[/bold]")
    console.print(f"  Organization: {config.github.org}")
    console.print(f"  LLM Engine Repo: {config.github.llm_engine_repo}")
    console.print(f"  Golden Datasets Repo: {config.github.golden_datasets_repo}")
    console.print(f"  Token: {'✓ Set' if config.github.token else '✗ Not set'}")
    
    # Endpoint config
    console.print("\n[bold]Endpoint Configuration:[/bold]")
    console.print(f"  URL: {config.endpoint.url}")
    console.print(f"  Model: {config.endpoint.model}")
    console.print(f"  Timeout: {config.endpoint.timeout}s")
    console.print(f"  Max Retries: {config.endpoint.max_retries}")
    
    # Judge config
    console.print("\n[bold]Judge Configuration:[/bold]")
    console.print(f"  Model: {config.judge.model}")
    console.print(f"  API Key: {'✓ Set' if config.judge.api_key else '✗ Not set'}")
    console.print(f"  Max Tokens: {config.judge.max_tokens}")
    
    # Benchmark config
    console.print("\n[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Max Concurrent: {config.benchmark.max_concurrent}")
    console.print(f"  Dataset Version: {config.benchmark.dataset_version}")
    console.print(f"  Cache Dir: {config.benchmark.cache_dir}")
    console.print(f"  Cache Enabled: {config.benchmark.cache_enabled}")
    console.print(f"  Output Dir: {config.benchmark.output_dir}")
    
    # Validate
    console.print()
    issues = config.validate()
    
    if issues:
        console.print("[yellow]⚠️  Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✅ Configuration is valid[/green]")
    
    console.print()


@cli.command()
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config, currently 1.1)",
)
@click.option(
    "--max-cases",
    type=int,
    default=None,
    help="Maximum number of test cases to run (for testing)",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Maximum concurrent test cases (each spawns up to 3 parallel guardrail calls)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results",
)
@click.option(
    "--force-refresh-prompts",
    is_flag=True,
    help="Bypass cache and re-fetch guardrail prompts from llm-proxy",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run_guardrails(
    dataset_version: Optional[str],
    max_cases: Optional[int],
    max_concurrent: Optional[int],
    output_dir: Optional[Path],
    force_refresh_prompts: bool,
    verbose: bool,
):
    """Evaluate safety, PII, and bias guardrails against the golden dataset."""
    from src.runner_guardrails import GuardrailEvaluator

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate config
    console.print("\n[bold blue]🔧 Configuration Validation[/bold blue]")
    issues = config.validate()
    if issues:
        console.print("[yellow]⚠️  Configuration warnings:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✓ Configuration valid[/green]")

    effective_version = dataset_version or config.benchmark.get_dataset_version("guardrails")
    effective_concurrent = max_concurrent or config.benchmark.max_concurrent

    table = Table(title="Guardrails Evaluation Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Guardrail Types", "safety, pii, bias")
    table.add_row("Dataset Version", effective_version)
    table.add_row("Max Cases", str(max_cases) if max_cases else "All (150)")
    table.add_row("Max Concurrent", str(effective_concurrent))
    table.add_row("Endpoint", config.endpoint.url)
    table.add_row("Model", config.endpoint.model)
    table.add_row("Prompt Source", f"Betterworks/{config.github.llm_proxy_repo}")
    console.print(table)
    console.print()

    try:
        evaluator = GuardrailEvaluator(
            dataset_version=dataset_version,
            output_dir=output_dir,
        )

        console.print("[bold green]🚀 Starting guardrail evaluation...[/bold green]\n")

        summary = asyncio.run(
            evaluator.run_all_tests(
                max_cases=max_cases,
                max_concurrent=effective_concurrent,
                force_refresh_prompts=force_refresh_prompts,
            )
        )

        console.print("\n[bold green]✅ Evaluation Complete![/bold green]\n")

        results_table = Table(title="Results Summary", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_row("Total Cases", str(summary["total"]))
        results_table.add_row(
            "Outcome Accuracy",
            f"{summary['outcome_accuracy']:.4f} ({summary['outcome_accuracy']*100:.1f}%)",
        )
        results_table.add_row(
            "Category Accuracy (non-clean)",
            f"{summary['category_accuracy']:.4f} ({summary['category_accuracy']*100:.1f}%)",
        )
        results_table.add_row("F1", f"{summary['f1']:.4f}")
        results_table.add_row("Errors", str(summary["errors"]))
        console.print(results_table)

        console.print(f"\n📁 Results saved to: {evaluator.output_dir}")
        console.print(f"  • [cyan]responses.csv[/cyan]  — per-case results with raw guardrail outputs")
        console.print(f"  • [cyan]responses.md[/cyan]   — summary table")
        console.print(f"  • [cyan]responses.jsonl[/cyan] — full data")
        console.print(f"  • [cyan]metrics.json[/cyan]   — classification metrics (JSON)")
        console.print(f"  • [cyan]metrics.md[/cyan]     — classification metrics (Markdown)\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    cli()
