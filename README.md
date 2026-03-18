# ZivAI AI Services Engine (MindSpore / MindNLP)

MindSpore-powered AI service for ZivAI. This service is moving beyond standalone ASAG and is being aligned to the shared `core-backend` database contracts for grading, inference tracing, mastery-related workflows, and other AI capabilities.

## Features
- Local LLM smoke test endpoint
- Shared-schema question grading endpoint
- Shared-schema whole-assessment grading endpoint
- MindSpore + MindNLP inference runtime
- Shared PostgreSQL integration with `core-backend`
- Foundation for grading, planning, lesson/resource generation, and related AI workflows

## Tech Stack
- Flask
- SQLAlchemy
- PostgreSQL
- MindSpore + MindNLP
- pdfplumber + python-docx

## Requirements
- Python 3.10+
- PostgreSQL 14+

Install system packages on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3-venv build-essential libpq-dev postgresql postgresql-contrib
```

Create the project virtual environment and install dependencies:

```bash
cd ~/Desktop/Huawei/innovation/application/zivai-asag-engine
python3 -m venv .engine-venv
source .engine-venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Current Recommended Run

This is the current working local setup:
- shared PostgreSQL source with `core-backend`
- MindSpore on CPU
- `Qwen/Qwen2.5-0.5B-Instruct`
- `MAX_NEW_TOKENS=128`

```bash
cd ~/Desktop/Huawei/innovation/application/zivai-asag-engine
source .engine-venv/bin/activate

export ZIVAI_DB_URL="jdbc:postgresql://<host>:5432/zivai"
export ZIVAI_DB_USERNAME="doadmin"
export ZIVAI_DB_PASSWORD="<db_password>"
export AUTO_CREATE_TABLES=false

export MS_DEVICE_TARGET=CPU
export MS_STRICT_DEVICE=true
export MS_MODE=PYNATIVE_MODE
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export MAX_NEW_TOKENS=128

ASAG_VENV="$PWD/.engine-venv" bash run.sh
```

Notes:
- Do not commit real database passwords into the repo or `.env` examples.
- `AUTO_CREATE_TABLES=false` is the correct setting when pointing at the shared LMS database.
- `/api/v1/llm/test` is a `POST` route, not a `GET` route.
- The grading endpoints are also `POST` routes and support optional `dry_run` / `force` flags in the JSON body.
- `run.sh` already applies the local `mindnlp` import patch needed for this project, so you do not need to edit package files manually.

## Shared Core DB Setup

ASAG now understands the same database env contract as `core-backend`:

```bash
export ZIVAI_DB_URL="jdbc:postgresql://<host>:5432/zivai"
export ZIVAI_DB_USERNAME="<db_user>"
export ZIVAI_DB_PASSWORD="<db_password>"
```

Behavior:
- If `ZIVAI_DB_URL` is set, ASAG derives the SQLAlchemy URL from `ZIVAI_DB_URL`, `ZIVAI_DB_USERNAME`, and `ZIVAI_DB_PASSWORD`.
- `DATABASE_URL` is still supported for direct SQLAlchemy usage.
- When both are present, `ZIVAI_DB_URL` takes precedence so ASAG can share the same DB source as `core-backend`.

Important:
- This only points ASAG at the same PostgreSQL instance as `core-backend`.
- The grading service now reads and writes the shared `lms.*` and `ai.*` tables used by `core-backend`.

## Active API Routes
- `GET /api/v1/health`
- `POST /api/v1/dev/bootstrap-grading-scenario`
- `GET /api/v1/dev/grading-targets`
- `POST /api/v1/grade/attempt-answer/<attempt_answer_id>`
- `POST /api/v1/grade/assessment-attempt/<assessment_attempt_id>`
- `POST /api/v1/llm/test`

### Dev Bootstrap Flow

If the shared LMS tables are empty, seed a complete grading scenario first:

`POST /api/v1/dev/bootstrap-grading-scenario`

Optional body:

```json
{
  "scenario": "rubric"
}
```

Supported scenarios:
- `rubric`: creates a short-answer question with a marking scheme and a partial student answer
- `holistic`: creates a short-answer question without a usable rubric so the grading service falls back to holistic LLM grading
- `objective`: creates an objective-style question with a deterministic correct answer

The bootstrap response returns:
- `attempt_answer_id`
- `assessment_attempt_id`
- ready-to-use grading URLs

To inspect what is available:

`GET /api/v1/dev/grading-targets`

### Grading API Notes

Per-question grading:
- route: `POST /api/v1/grade/attempt-answer/<attempt_answer_id>`
- source of truth: shared `lms.attempt_answers`, `lms.assessment_questions`, `lms.questions`, `lms.marking_schemes`, `lms.marking_scheme_items`
- write-back: `lms.attempt_answers`, `lms.assessment_attempts`, `lms.assessment_results`, `ai.ai_inference_runs`

Whole-assessment grading:
- route: `POST /api/v1/grade/assessment-attempt/<assessment_attempt_id>`
- grades each answer on the attempt, then refreshes attempt/result rollups

Optional request body:

```json
{
  "dry_run": false,
  "force": false
}
```

Behavior:
- If `human_score` already exists, the service preserves it and returns the existing grade.
- If `ai_score` already exists and `force=false`, the service returns the existing AI grade.
- If the student answer is blank, the service returns `0` with direct feedback and no LLM call.
- If the question is objective and a correct answer exists in `rubric_json`, the service grades deterministically.
- If a rubric/marking scheme exists, the LLM scores per rubric item and generates feedback.
- If no rubric exists, the LLM falls back to holistic grading and marks the answer for review.

## Current State
- The app is currently running on CPU, so inference latency is still high.
- `Qwen/Qwen2.5-0.5B-Instruct` is the current active model because it is the best working balance in this environment.
- Deprecated standalone ASAG CRUD routes and local persistence models have been removed.
- Grading now uses shared LMS/AI contracts instead of local SQLite-era tables.
- Broader AI workflow endpoints beyond grading are still to be built.
