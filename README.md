# Tuilink Project - AI Auto Reply

An intelligent system for generating contextually appropriate replies in professional conversations, particularly focused on job referral scenarios.

## Overview

This project implements an automated reply generation system that:

1. Analyzes conversation context
2. Classifies conversation categories
3. Suggests relevant topics for replies
4. Generates professional and contextually appropriate responses

## Project Structure

```
.
├── charts/                     # Mermaid charts
├── input/                      # Input data directory
│   ├── categories.json         # Category definitions
│   └── convo_2454_rows.xlsx    # Conversation dataset
├── models/                     # Core data models
├── nodes/                      # Processing nodes
├── utils/                      # Utility functions
├── .env.example                # Example environment config
├── build.sh                    # Build script for dist/
├── handler.py                  # Web / Quick Lambda: generate reply messages
├── requirements-compact.txt    # Compact dependencies for deployment
├── requirements.txt            # Full dependencies for local development
└── *.ipynb                     # Local notebooks to play around
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file (or export in your deployment) with the following credentials and settings.
You can copy `.env.example` as a starting point.

```bash
# DeepSeek (OpenAI-compatible API)
DEEPSEEK_API_KEY=<your-deepseek-api-key>
DEEPSEEK_API_BASE=https://api.deepseek.com/beta

# LLM configuration (use deepseek-chat; deepseek-reasoner does not support structured output)
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0

# Caching, whether to cache the LLM responses locally
LLM_USE_CACHE=false

# Whether to show debug fields in outputs (reason, confidence)
LLM_INCLUDE_DEBUG_FIELDS=true
```

Structured LLM outputs use DeepSeek's beta API with strict tool calling via LangChain's
`with_structured_output(..., method="function_calling", strict=True)`.

## Build for Deployment

Use the build script to create a self-contained `dist/` folder ready for packaging and deployment.

1. Ensure your `.env` is present and set `LLM_USE_CACHE=false` and `LLM_INCLUDE_DEBUG_FIELDS=false` for deployment (to avoid local cache writes and unnecessary debug fields in outputs).

2. Run the build script:

```bash
chmod +x ./build.sh
./build.sh
```

3. The script will create `./dist` containing:

- `handler.py`
- `requirements.txt` (copied from `requirements-compact.txt`)
- `models/`, `nodes/`, `utils/`, `input/` (necessary files only)
- `__init__.py` files for Python package directories (`models/`, `nodes/`, `utils/`)

4. Deployment notes for your infra repo:

- Set your Lambda source path to the `dist` folder of this repo, e.g., `AI_AUTO_REPLY_LAMBDA_SOURCE_PATH=../../course-ai-auto-reply-starter/dist/`
- Use handler for the Lambda (triggered by API Gateway), e.g., `AI_AUTO_REPLY_LAMBDA_HANDLER=handler.handler`
- (Optional) Set your Lambda layer source path to the `dist` folder of this repo, e.g., `AI_AUTO_REPLY_LAMBDA_LAYER_SOURCE_PATH=../../course-ai-auto-reply-starter/dist/`
