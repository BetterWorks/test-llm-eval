# LLM Evaluation Framework

Standalone evaluation framework for testing Betterworks AI features and guardrails against golden datasets. Pulls prompts dynamically from GitHub and evaluates against versioned test datasets.

## Supported Evaluations

### AI Features (generation quality)

Prompts sourced from `Betterworks/llm-engine`. Outputs raw LLM responses for manual or downstream review.

| Feature | Dataset | Version | Test Cases |
|---------|---------|---------|------------|
| Writing Assistant | `writing_assistant` | v1.2 | 150 |
| Goal Assist | `goal_assist` | v1.0 | 1 |
| Feedback Summary | `feedback_summary` | v1.0 | 2 |
| Performance Summary | `performance_summary` | v1.1 | 100 |
| Meetings Summary | `meetings_summary` | v1.0 | 100 |
| Skills Discovery | `skills_discovery` | v1.1 | 100 |

### Guardrails (classification accuracy)

Prompts sourced from `Betterworks/llm-proxy`. Evaluates safety, PII, and bias guardrails and computes classification metrics against ground-truth labels.

| Guardrail | Type | Dataset | Version | Test Cases |
|-----------|------|---------|---------|------------|
| Safety | Blocking (error) | `guardrails` | v1.1 | 150 |
| PII | Blocking (error) / Silent | `guardrails` | v1.1 | 150 |
| Bias | Non-blocking (warning) | `guardrails` | v1.1 | 150 |

---

## Quick Start

### 1. Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Configuration](#configuration) below).

### 3. Verify setup

```bash
python cli.py test-connection
python cli.py config-check
```

### 4. Run a quick smoke test

```bash
# Feature eval (5 writing assistant cases)
python cli.py run --feature writing_assistant --max-cases 5

# Guardrail eval (5 cases)
python cli.py run-guardrails --max-cases 5
```

---

## CLI Commands

### `run` — Feature evaluation

Collect LLM responses for an AI feature.

```bash
python cli.py run [OPTIONS]

Options:
  --feature TEXT          writing_assistant | goal_assist | feedback_summary |
                          performance_summary | meetings_summary | skills_discovery
  --dataset-version TEXT  Override dataset version (default: per-feature from config)
  --max-cases INTEGER     Limit test cases (useful for smoke tests)
  --max-concurrent INT    Parallel requests (default: 3)
  --output-dir PATH       Custom output directory
  -v, --verbose           Debug logging
```

**Examples:**

```bash
# Quick test
python cli.py run --feature writing_assistant --max-cases 5

# Full run with higher concurrency
python cli.py run --feature performance_summary --max-concurrent 5

# Specific dataset version
python cli.py run --feature skills_discovery --dataset-version 1.0
```

---

### `run-guardrails` — Guardrail evaluation

Evaluate safety, PII, and bias guardrails against the golden dataset. Each test case runs all requested guardrail types in parallel, then computes classification metrics.

```bash
python cli.py run-guardrails [OPTIONS]

Options:
  --dataset-version TEXT    Override dataset version (default: 1.1)
  --max-cases INTEGER       Limit test cases
  --max-concurrent INT      Concurrent test cases (each spawns up to 3 parallel
                            guardrail calls internally)
  --output-dir PATH         Custom output directory
  --force-refresh-prompts   Re-fetch guardrail prompts from llm-proxy (bypass cache)
  -v, --verbose             Debug logging
```

**Examples:**

```bash
# Smoke test
python cli.py run-guardrails --max-cases 5

# Full evaluation
python cli.py run-guardrails

# Force fresh prompts from llm-proxy
python cli.py run-guardrails --force-refresh-prompts
```

---

### `dataset-info` — Dataset statistics

```bash
python cli.py dataset-info --feature writing_assistant
python cli.py dataset-info --feature performance_summary --dataset-version 1.1
```

### `test-connection` — Endpoint health check

```bash
python cli.py test-connection
```

### `config-check` — Display current configuration

```bash
python cli.py config-check
```

---

## Output Files

### Feature eval

Each run creates a timestamped directory under `artifacts/<feature>/run_<timestamp>/`:

```
artifacts/writing_assistant/run_20260507_120000/
├── responses.jsonl    # Full response data (one JSON object per line)
├── responses.csv      # Tabular format (14 columns, no text truncation)
└── responses.md       # Markdown summary with response table
```

**CSV columns:** `test_number`, `test_id`, `feature`, `category`, `input_text`, `input_text_length`, `llm_output`, `output_length`, `model`, `execution_time_seconds`, `ai_response_time_seconds`, `timestamp`, `success`, `error_message`

### Guardrail eval

Each run creates a timestamped directory under `artifacts/guardrails/run_<timestamp>/`:

```
artifacts/guardrails/run_20260507_120000/
├── responses.jsonl    # Full per-case data including per-guardrail raw responses
├── responses.csv      # Tabular format with per-type triggered/category columns
├── responses.md       # Markdown summary (predicted vs expected outcome per case)
├── metrics.json       # Full classification metrics (machine-readable)
└── metrics.md         # Classification metrics report (human-readable)
```

**Guardrail CSV columns:** `test_number`, `test_id`, `direction`, `context_type`, `guardrails_requested`, `input_text`, `input_text_length`, `predicted_outcome`, `expected_outcome`, `outcome_correct`, `category_correct`, `expected_triggered_categories`, `safety_raw`, `safety_triggered`, `safety_category`, `pii_raw`, `pii_triggered`, `pii_category`, `bias_raw`, `bias_triggered`, `bias_category`, `difficulty`, `locale`, `edge_case_type`, `model`, `execution_time_seconds`, `timestamp`, `error_message`

**Metrics reported:**
- Overall: outcome accuracy, category accuracy, precision, recall, F1, FPR (over-block rate), FNR (miss rate), TP/FP/TN/FN
- Per guardrail type: same metrics for safety / pii / bias independently
- Breakdowns by: difficulty, locale, context type
- Edge cases: separate accuracy for the 20 boundary-condition cases per split

---

## Configuration

All settings via `.env` file or environment variables.

### GitHub

```bash
GITHUB_TOKEN=<token>                      # Personal access token (recommended to avoid rate limits)
GITHUB_ORG=Betterworks                    # Default: Betterworks
LLMENGINE_REPO=llm-engine                 # Prompt source for AI features
LLM_PROXY_REPO=llm-proxy                  # Prompt source for guardrails
GOLDEN_DATASETS_REPO=llm-golden-datasets  # Test dataset source
```

### Endpoint

```bash
ENDPOINT_URL=http://10.50.21.45:8000/v1/chat/completions
ENDPOINT_MODEL=google/gemma-4-31B-it
```

### Execution

```bash
MAX_CONCURRENT=3       # Parallel test cases (default: 3)
CACHE_DIR=.cache       # Local cache for datasets and prompts (default: .cache)
CACHE_ENABLED=true     # Toggle caching (default: true)
OUTPUT_DIR=artifacts   # Results output directory (default: artifacts)
```

### Dataset versions (per-feature overrides)

```bash
DATASET_VERSION_WRITING_ASSISTANT=1.2
DATASET_VERSION_GOAL_ASSIST=1.0
DATASET_VERSION_FEEDBACK_SUMMARY=1.0
DATASET_VERSION_PERFORMANCE_SUMMARY=1.1
DATASET_VERSION_MEETINGS_SUMMARY=1.0
DATASET_VERSION_SKILLS_DISCOVERY=1.1
DATASET_VERSION_GUARDRAILS=1.1
```

---

## Cache Management

The framework caches datasets and prompts locally in `.cache/` to speed up repeated runs and reduce GitHub API calls.

### Cache Structure

```
.cache/
├── datasets/                        # Cached golden datasets
│   ├── writing_assistant/
│   ├── goal_assist/
│   ├── feedback_summary/
│   ├── performance_summary/
│   ├── meetings_summary/
│   ├── skills_discovery/
│   └── guardrails/
└── prompts/                         # Cached prompts from llm-engine/llm-proxy
    ├── writing_assistant/
    ├── goals_assistant/
    ├── feedback_summary/
    ├── performance_summary/
    ├── meetings_summary/
    ├── skill/
    └── guardrails/
```

### Clear Cache Commands

```bash
# Clear all caches (datasets + prompts)
rm -rf .cache

# Clear only prompts cache (force fresh prompt download)
rm -rf .cache/prompts

# Clear only datasets cache (force fresh dataset download)
rm -rf .cache/datasets

# Clear specific feature prompt cache
rm -rf .cache/prompts/writing_assistant

# Clear guardrail prompts cache
rm -rf .cache/prompts/guardrails
```

### When to Clear Cache

- **Prompts updated in llm-engine/llm-proxy** — Clear prompts cache to get latest versions
- **Dataset version changed** — Cache is version-specific, so switching versions auto-fetches new data
- **Debugging unexpected responses** — Clear prompts cache to ensure you have latest instructions
- **Multi-language support issues** — Clear prompts cache to refresh locale-aware prompt templates
- **Disk space concerns** — Typical cache is ~5MB datasets + ~50KB prompts

### Bypass Cache

```bash
# Force refresh guardrail prompts without clearing cache file
python cli.py run-guardrails --force-refresh-prompts

# Disable caching for a single run (set in .env)
CACHE_ENABLED=false python cli.py run --feature writing_assistant
```

---

## Architecture

```
test-llm-eval/
├── config.py                       # Pydantic configuration (env vars)
├── cli.py                          # Click CLI (run, run-guardrails, dataset-info, ...)
├── src/
│   ├── dataset_loader.py           # Load datasets from llm-golden-datasets (+ GuardrailTestCase)
│   ├── prompt_loader.py            # Fetch feature prompts from llm-engine
│   ├── guardrail_prompt_loader.py  # Fetch guardrail prompts from llm-proxy
│   ├── prompt_builder.py           # Feature-specific prompt builders
│   ├── factory.py                  # PromptBuilderFactory
│   ├── endpoint_client.py          # OpenAI-compatible HTTP client with retry
│   ├── runner_simple.py            # Feature eval runner (response collection)
│   ├── runner_guardrails.py        # Guardrail eval runner (classification + metrics)
│   └── utils.py                    # Shared utilities
├── .cache/                         # Cached datasets and prompts (git-ignored)
│   ├── datasets/                   # Per-feature/version JSONL caches
│   └── prompts/                    # Per-feature prompt file caches
│       └── guardrails/             # Guardrail prompt caches (safety/pii/bias)
└── artifacts/                      # Evaluation outputs (git-ignored)
```

### Prompt sources

**AI features** — fetched from `Betterworks/llm-engine`:
- `app/prompt/writing_assistant/meta.py`
- `app/prompt/goals_assistant/meta.py`
- `app/prompt/feedback_summary/meta.py`
- `app/prompt/performance_summary/meta.py`
- `app/prompt/meetings_summary/meta.py`
- `app/prompt/skill/{conversation,feedback,job_description,job_title}/meta.py`

**Guardrails** — fetched from `Betterworks/llm-proxy`:
- `src/api/prompts/selfcheck.py` → `INPUT_CHECK_PROMPT`, `OUTPUT_CHECK_PROMPT`
- `src/api/prompts/pii.py` → `PII_DETECTION_PROMPT`
- `src/api/prompts/bias.py` → `BIAS_DETECTION_PROMPT`

All prompts are cached in `.cache/prompts/` after the first fetch.

### Guardrail evaluation flow

```
test_case.prompt (JSON)
  → parse: {text, guardrails_requested, direction, context_type}
  → parallel LLM calls (one per requested guardrail, via asyncio.to_thread)
      safety call  →  "unsafe: <Category> - <Reason>"  |  "safe: ..."
      pii call     →  "PII: <Category> - <Explanation>"  |  "Safe: ..."
      bias call    →  "bias: <Category> - <Explanation>"  |  "no-bias: ..."
  → combine: error > warning > clean (silent PII → clean)
  → score vs test_case.response (ground truth)
  → aggregate classification metrics
```

---

## Troubleshooting

**GitHub rate limit errors** — Set `GITHUB_TOKEN` in `.env`.

**Prompt not found** — If a guardrail prompt constant name changed in llm-proxy, clear the guardrail prompts cache:
```bash
rm -rf .cache/prompts/guardrails
python cli.py run-guardrails --force-refresh-prompts --max-cases 1 --verbose
```

**Cache stale or outdated prompts** — See the [Cache Management](#cache-management) section for detailed cache clearing commands.

**Multi-language responses defaulting to English** — Ensure prompts cache is cleared to get latest locale-aware templates:
```bash
rm -rf .cache/prompts
```

**Endpoint unreachable** — Check VPN/network access to the endpoint:
```bash
python cli.py test-connection
```

**Verbose logging** — Add `-v` to any command for debug output:
```bash
python cli.py run-guardrails --max-cases 5 -v
```
