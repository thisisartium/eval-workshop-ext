# Eval Workshop

A hands-on workshop for building and evaluating LLM-powered agents using a helpdesk routing system as the example.

## What You'll Build

A helpdesk agent that:
1. **Routes requests** to the right department (IT, HR, Facilities, Finance, Legal, Security)
2. **Answers questions** using a knowledge base (HR policies)
3. **Escalates to humans** when it can't help

Along the way, you'll learn evaluation-driven development: running experiments, analyzing failures, and iterating on prompts.

## Get Started

**[Start the Workshop](docs/workshop.md)** (~60 minutes, guided)

## Quick Start

### Option A: GitHub Codespaces (recommended)

Click the button below to open a fully configured environment in your browser — no local setup required.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/<org>/<repo>)

Once it opens, add your API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### Option B: VS Code Dev Container

1. Clone the repo and open in VS Code
2. When prompted, click **"Reopen in Container"**
3. Add your API key to `.env` (same as above)

Python, uv, and all dependencies are installed automatically. Cat Cafe starts on container launch.

### Option C: Local Setup

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd eval-workshop-ext
uv sync

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys)
# Or OPENAI_API_KEY if using OpenAI (https://platform.openai.com/api-keys)

# 3. Start local services (pull latest images first)
docker compose pull && docker compose up -d

# 4. Verify services are running
docker compose ps
# Should show cat-cafe as running/healthy
```

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and Docker.

## Try the Agent

```bash
# Baseline (routing only, escalation only)
uv run helpdesk-agent -c configs/baseline.yaml "My laptop won't turn on"

# With HR specialist (tuned concierge + specialist)
uv run helpdesk-agent -c configs/tuned.yaml "How many vacation days do I get?"
```

## Project Structure

```
eval-workshop-ext/
├── configs/                 # Agent configurations + models.yaml
├── data/                    # Datasets (JSONL format)
├── docs/
│   └── workshop.md          # Workshop guide
├── experiments/             # Experiment definitions
├── kb/                      # Knowledge base documents
├── prompts/                 # Prompt files (versioned)
├── scripts/                 # CLI tools
└── src/
    ├── agent_platform/      # Custom agent runtime
    └── helpdesk/            # Helpdesk domain
```

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| CAT Cafe | http://localhost:8000 | Experiment tracking, traces, datasets |

## Prerequisites

- **Anthropic API key** ([get one here](https://console.anthropic.com/settings/keys)) or **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))
- **One of:** GitHub Codespaces (browser only), VS Code with Dev Containers extension, or local Python 3.13+ with Docker
