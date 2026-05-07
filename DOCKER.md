# Docker Setup Guide

This guide explains how to run the LLM Evaluation Framework using Docker and Docker Compose.

## Prerequisites

- Docker (v20.10 or later)
- Docker Compose (v1.29 or later)

## Setup

1. **Clone the repository** (if you haven't already):
```bash
git clone <repo-url>
cd test-llm-eval
```

2. **Create .env file**:
```bash
cp .env.example .env
```

3. **Edit .env file** with your credentials:
```bash
# Required:
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_api_key_here

# Pre-configured (verify these match your setup):
ENDPOINT_URL=http://10.50.21.45:8000/v1/chat/completions
ENDPOINT_MODEL=google/gemma-4-31B-it
JUDGE_MODEL=gpt-5
```

## Running with Docker Compose

The framework provides several pre-configured services:

### 1. Validate Setup
Verify all dependencies and configuration before running tests:
```bash
docker-compose run --rm validate
```

### 2. Test Endpoint Connection
Verify connectivity to the LLM endpoint:
```bash
docker-compose run --rm test-connection
```

### 3. Quick Test (5 cases)
Run a quick test with 5 test cases:
```bash
docker-compose run --rm quick-test
```

### 4. Full Benchmark
Run the complete benchmark suite:
```bash
docker-compose run --rm full-benchmark
```

### 5. Custom Commands
Run any CLI command with any of the 6 supported features:

```bash
# Writing Assistant
docker-compose run --rm llm-eval python cli.py run --feature writing_assistant --max-cases 5

# Goal Assist
docker-compose run --rm llm-eval python cli.py run --feature goal_assist --max-cases 5

# Feedback Summary
docker-compose run --rm llm-eval python cli.py run --feature feedback_summary --max-cases 5

# Performance Summary
docker-compose run --rm llm-eval python cli.py run --feature performance_summary --max-cases 5

# Meetings Summary
docker-compose run --rm llm-eval python cli.py run --feature meetings_summary --max-cases 5

# Skills Discovery
docker-compose run --rm llm-eval python cli.py run --feature skills_discovery --max-cases 5

# Dataset info
docker-compose run --rm llm-eval python cli.py dataset-info --feature performance_summary
docker-compose run --rm llm-eval python cli.py config-check
```

## Results and Artifacts

All artifacts are persisted to your host machine in the `./artifacts/` directory. The cache is also persisted in `./.cache/`.

Example output structure for all 6 features:
```
artifacts/
├── writing_assistant/
│   └── run_20260506_092346/
│       ├── responses.jsonl
│       ├── responses.csv
│       └── responses.md
├── goal_assist/
├── feedback_summary/
├── performance_summary/
├── meetings_summary/
└── skills_discovery/
```

## Building the Image

If you need to rebuild the Docker image:
```bash
docker-compose build
```

## Development Mode

The Docker Compose setup mounts your source code as a volume, so changes to Python files are immediately reflected:

```bash
# Edit source files locally
vim src/evaluator.py

# Run without rebuilding
docker-compose run --rm quick-test
```

## Troubleshooting

### Permission Issues
If you encounter permission issues with artifacts or cache directories:
```bash
sudo chown -R $(id -u):$(id -g) artifacts .cache
```

### Network Issues
If the container can't reach the endpoint at `10.50.21.45:8000`:
- Verify the endpoint is accessible from your host machine
- Try using `--network host` mode (Linux only):
  ```bash
  docker run --rm --network host --env-file .env -v $(pwd)/artifacts:/app/artifacts llm-eval python cli.py test-connection
  ```

### View Container Logs
```bash
docker-compose logs quick-test
```

## Advanced Usage

### Run with Different Dataset Version
```bash
docker-compose run --rm llm-eval python cli.py run \
  --feature writing_assistant \
  --max-cases 10 \
  --max-concurrent 5 \
  --verbose
```

### Interactive Shell
Access the container shell for debugging:
```bash
docker-compose run --rm llm-eval /bin/bash
```

## CI/CD Integration

Example GitHub Actions workflow:
```yaml
name: Benchmark Tests
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Create .env file
        run: |
          echo "GITHUB_TOKEN=${{ secrets.GITHUB_TOKEN }}" >> .env
          echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> .env
          cat .env.example | grep -v "GITHUB_TOKEN\|OPENAI_API_KEY" >> .env
      
      - name: Run quick test
        run: docker-compose run --rm quick-test
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: artifacts/
```

## Supported Features

The Docker setup supports all 6 AI features:

1. **Writing Assistant** - Text rephrasing (150 test cases, ~3-5s per case)
2. **Goal Assist** - Goal generation (1 test case, ~30-35s per case)
3. **Feedback Summary** - Feedback summarization (2 test cases, ~8-10s per case)
4. **Performance Summary** - Performance reviews (100 test cases, ~6-8s per case)
5. **Meetings Summary** - Meeting notes (100 test cases, ~30-40s per case)
6. **Skills Discovery** - Skills extraction (100 test cases, ~40-45s per case)

### Running Full Test Suite

Test all features in Docker:
```bash
#!/bin/bash
for feature in writing_assistant goal_assist feedback_summary performance_summary meetings_summary skills_discovery; do
  echo "Testing $feature..."
  docker-compose run --rm llm-eval python cli.py run --feature $feature --max-cases 5
done
```

### Performance Considerations

For features with longer response times (meetings_summary, skills_discovery):
```bash
# Reduce concurrency to avoid timeouts
docker-compose run --rm llm-eval python cli.py run --feature meetings_summary --max-concurrent 2
```
