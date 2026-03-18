# ZivAI AI Services Engine (MindSpore / MindNLP)

MindSpore-powered AI service for ZivAI. This service is moving beyond standalone ASAG and is being aligned to the shared `core-backend` database contracts for grading, inference tracing, mastery-related workflows, and other AI capabilities.

## Features
- Local LLM smoke test endpoint
- ASAG short-answer scoring endpoint
- Teacher assessment generation endpoint
- DKT mastery update and mastery lookup endpoints
- General OCR endpoint for images, scanned PDFs, and digital documents
- Payload-based per-question grading endpoint
- Payload-based whole-assessment grading endpoint
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
- Huawei OCR Python SDK
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

Optional tracked env helper:

```bash
source scripts/export_ai_service_env.sh
```

It exposes the current MindSpore runtime defaults plus the Huawei OCR and DKT env names used by the AI service routes.

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

export ASAG_ENABLED=true
export ASAG_MODEL_DIR="$PWD/models/asag"
export ASAG_CKPT_PATH="$ASAG_MODEL_DIR/asag_mohler_best.ckpt"
export ASAG_MINDIR_PATH="$ASAG_MODEL_DIR/asag_mohler.mindir"
export ASAG_VOCAB_PATH="$ASAG_MODEL_DIR/tokenizer/vocab.json"
export ASAG_RESULTS_PATH="$ASAG_MODEL_DIR/results.json"
export ASAG_MAX_LENGTH=256
export ASAG_SHORT_ANSWER_MAX_MARKS=5

export DKT_MODEL_DIR="$PWD/models/dkt"
export DKT_CLOUD_CKPT_PATH="$DKT_MODEL_DIR/dkt_lstm_cloud.ckpt"
export DKT_SKILL_MAP_PATH="$DKT_MODEL_DIR/skill_map_v1.json"
export DKT_MODEL_META_PATH="$DKT_MODEL_DIR/model_meta.json"
export DKT_DEFAULT_SUBJECT_CODE="computer_science"

export HWC_AK="<huaweicloud_ak>"
export HWC_SK="<huaweicloud_sk>"
export HWC_PROJECT_ID="<huaweicloud_project_id>"
export HWC_OCR_ENDPOINT="https://ocr.ap-southeast-1.myhuaweicloud.com"

ASAG_VENV="$PWD/.engine-venv" bash run.sh
```

Notes:
- Do not commit real database passwords into the repo or `.env` examples.
- `AUTO_CREATE_TABLES=false` is the correct setting when pointing at the shared LMS database.
- `/api/v1/llm/test` is a `POST` route, not a `GET` route.
- The grading endpoints are also `POST` routes and support optional `dry_run` / `force` flags in the JSON body.
- `run.sh` already applies the local `mindnlp` import patch needed for this project, so you do not need to edit package files manually.
- ASAG is optional. If its checkpoint or MindIR is missing or unreadable, grading falls back to the existing LLM path and `/api/v1/asag/score` returns a configuration error until the artifacts are staged in `models/asag/`.

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
- `POST /api/v1/dkt/update`
- `GET /api/v1/dkt/mastery/<student_id>`
- `POST /api/v1/asag/score`
- `POST /api/v1/agents/teacher/assessment-generation`
- `POST /api/v1/agents/teacher/plan-generation`
- `POST /api/v1/agents/student/assessment`
- `POST /api/v1/agents/ocr/general`
- `POST /api/v1/grade/question`
- `POST /api/v1/grade/assessment`
- `POST /api/v1/grade/attempt-answer/<attempt_answer_id>`
- `POST /api/v1/grade/assessment-attempt/<assessment_attempt_id>`
- `POST /api/v1/llm/test`

### ASAG Short-Answer Scoring

Route:
- `POST /api/v1/asag/score`

Use this route to smoke test the MindSpore ASAG regression model directly. It is intended for short-answer questions that have a reference answer.

Request body:

```json
{
  "question": "Explain what binary search is.",
  "reference_answer": "Binary search is a search algorithm for sorted data that checks the middle element and repeatedly halves the search space.",
  "student_answer": "Binary search checks the middle item in sorted data.",
  "max_score": 5,
  "expected_points": [
    "checks the middle element",
    "requires sorted data",
    "halves the search space"
  ]
}
```

Current behavior:
- raw ASAG output is calibrated from `0-5`
- the service normalizes it to `0-1`
- then scales it to `max_score`
- response includes a feedback band plus lightweight feedback fields

If the checkpoint or MindIR is missing or unreadable, this route returns `503` with a configuration error instead of silently failing.

### Teacher Assessment Generation

Route:
- `POST /api/v1/agents/teacher/assessment-generation`

This route is currently implemented to match the request body already sent by `zivai-web`.
It returns a top-level JSON array so the current frontend can consume it without a patch.

Current request body:

```json
{
  "context": "Generate questions for Binary Search quiz. Focus on practical understanding.",
  "difficulty": "medium",
  "questionTypes": "mixed",
  "numberOfQuestions": 5,
  "attributes": {
    "Algorithms": "Core search and sort techniques",
    "Problem Solving": "Applying algorithmic reasoning"
  },
  "referenceDocuments": [
    {
      "documentName": "binary_search_notes.pdf",
      "markdown": "# Binary Search\\nBinary search works on sorted data..."
    }
  ],
  "tags": ["Algorithms", "Binary Search"]
}
```

Current response body:

```json
[
  {
    "text": "What is binary search primarily used for?",
    "type": "multiple_choice",
    "options": [
      "Searching sorted data",
      "Sorting data",
      "Compressing files",
      "Encrypting data"
    ],
    "correctAnswer": "Searching sorted data",
    "correctAnswers": ["Searching sorted data"],
    "explanation": "Award the mark for identifying binary search as a search algorithm for sorted data.",
    "difficulty": "medium",
    "tags": ["Algorithms", "Binary Search"],
    "points": 1,
    "maxMarks": 1,
    "markingGuide": {
      "mode": "objective",
      "expectedAnswer": "Searching sorted data",
      "rubricItems": []
    },
    "rubricJson": {
      "correctAnswer": "Searching sorted data",
      "correctAnswers": ["Searching sorted data"],
      "expectedAnswer": "Searching sorted data",
      "markingGuide": "Award the mark for identifying binary search as a search algorithm for sorted data.",
      "rubricItems": []
    },
    "referenceFallbackUsed": false,
    "sourceDocumentsUsed": ["binary_search_notes.pdf"]
  },
  {
    "text": "Explain one condition required before binary search can be used.",
    "type": "short_answer",
    "options": [],
    "correctAnswer": null,
    "correctAnswers": [],
    "explanation": "Award marks for stating that the data must be sorted.",
    "difficulty": "medium",
    "tags": ["Algorithms", "Binary Search"],
    "points": 4,
    "maxMarks": 4,
    "markingGuide": {
      "mode": "rubric",
      "expectedAnswer": "The data must be sorted before binary search can be used.",
      "rubricItems": [
        {
          "index": 1,
          "description": "States that the data must be sorted",
          "marks": 4,
          "keywords": ["sorted"]
        }
      ]
    },
    "rubricJson": {
      "correctAnswer": null,
      "correctAnswers": [],
      "expectedAnswer": "The data must be sorted before binary search can be used.",
      "markingGuide": "Award marks for stating that the data must be sorted.",
      "rubricItems": [
        {
          "index": 1,
          "description": "States that the data must be sorted",
          "marks": 4,
          "keywords": ["sorted"]
        }
      ]
    },
    "referenceFallbackUsed": false,
    "sourceDocumentsUsed": ["binary_search_notes.pdf"]
  }
]
```

Behavior:
- `questionTypes=multiple_choice` produces objective questions.
- `questionTypes=structured` produces short-answer questions with rubrics.
- `questionTypes=mixed` produces a mix of objective and structured questions.
- A marking guide is generated for every question.
- If `referenceDocuments` are missing, empty, or unusable, the service falls back to the provided `context`, `attributes`, `tags`, and general Computer Science knowledge instead of failing.
- The response already includes `rubricJson`-style data that can later be mapped into the shared LMS persistence layer.

### General OCR

Route:
- `POST /api/v1/agents/ocr/general`

This is a separate OCR route. It does not change the existing grading or teacher-agent contracts.

Purpose:
- extract text from images and scanned PDFs using Huawei OCR
- extract text from digital PDFs, `.docx`, and plain-text files without paying OCR cost when direct text is already available
- return `referenceDocuments` payloads that can be passed into the teacher generation flows later

Huawei OCR environment variables:

```bash
export HWC_AK="<huaweicloud_ak>"
export HWC_SK="<huaweicloud_sk>"
export HWC_PROJECT_ID="<huaweicloud_project_id>"
export HWC_OCR_ENDPOINT="https://ocr.ap-southeast-1.myhuaweicloud.com"
export HWC_HOST="ocr.ap-southeast-1.myhuaweicloud.com"
export HWC_FORCE_TRAILING_SLASH=false
export HWC_HTTP_TIMEOUT_SECONDS=180

export HWC_GENERAL_TEXT_DETECT_DIRECTION=true
export HWC_GENERAL_TEXT_QUICK_MODE=false
export HWC_GENERAL_TEXT_MAX_ORIGINAL_FILE_SIZE_BYTES=7340032
export HWC_GENERAL_TEXT_MAX_ENCODED_IMAGE_BYTES=2500000
export HWC_GENERAL_TEXT_MAX_IMAGE_WIDTH=2200
export HWC_GENERAL_TEXT_MAX_IMAGE_HEIGHT=2200
export HWC_GENERAL_TEXT_JPEG_QUALITY=0.72
export HWC_GENERAL_TEXT_MIN_JPEG_QUALITY=0.50
export HWC_GENERAL_TEXT_ADAPTIVE_RESIZE_PERCENT=85
export HWC_GENERAL_TEXT_MAX_ADAPTIVE_PASSES=6
export HWC_GENERAL_TEXT_PDF_RENDER_DPI=200
export HWC_GENERAL_TEXT_MAX_PDF_PAGES=50
```

Request:
- `multipart/form-data`
- supported fields:
  - `file` or repeated `files`
  - optional `request` JSON blob
  - or direct form fields such as `module`, `source`, `language`, `preferDigitalExtraction`, `forceOcr`

Frontend compatibility:
- If the request is a bare file upload with no extra form fields, the route returns the legacy top-level array currently expected by `zivai-web/src/services/aiService.ts`.
- If the request includes a `request` JSON blob, extra form fields, or `response_mode=full`, the route returns the richer object wrapper documented below.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agents/ocr/general" \
  -F "file=@/path/to/binary-search-notes.pdf" \
  -F 'request={"source":"teacher_assessment_generation","module":"Binary Search","preferDigitalExtraction":true,"forceOcr":false,"detectDirection":true,"returnMarkdownResult":true}'
```

Example response:

```json
{
  "status": "ok",
  "request": {
    "source": "teacher_assessment_generation",
    "module": "Binary Search",
    "url": null,
    "language": null,
    "preferDigitalExtraction": true,
    "forceOcr": false,
    "detectDirection": true,
    "quickMode": false,
    "characterMode": false,
    "singleOrientationMode": false,
    "returnMarkdownResult": true,
    "maxPages": null
  },
  "documents": [
    {
      "documentName": "binary-search-notes.pdf",
      "fileFormat": "pdf",
      "pageNumber": 1,
      "totalPages": 2,
      "engine": "builtin-digital-text",
      "mode": "digital_text",
      "fullText": "Binary search works on sorted data...",
      "markdown": "Binary search works on sorted data...",
      "averageConfidence": null,
      "wordsBlockCount": 14,
      "module": "Binary Search",
      "source": "teacher_assessment_generation",
      "referenceDocument": {
        "documentName": "binary-search-notes.pdf#page-1",
        "markdown": "Binary search works on sorted data..."
      },
      "metadata": {
        "source": "pdfplumber"
      }
    }
  ],
  "combinedText": "Binary search works on sorted data...",
  "referenceDocuments": [
    {
      "documentName": "binary-search-notes.pdf#page-1",
      "markdown": "Binary search works on sorted data..."
    }
  ],
  "warnings": [
    "Used built-in digital PDF extraction for 'binary-search-notes.pdf' where selectable text was available."
  ]
}
```

Behavior:
- Images and scanned PDFs use Huawei OCR through the Python SDK.
- Text PDFs use direct extraction first when `preferDigitalExtraction=true`.
- `.docx`, `.txt`, `.md`, `.csv`, and `.json` use built-in text extraction.
- If Huawei OCR credentials are missing, digital documents still work, but image and scanned-PDF OCR will fail with a clear configuration error.

Legacy response example for current `zivai-web` OCR calls:

```json
[
  {
    "documentName": "binary-search-notes.pdf#page-1",
    "markdown": "Binary search works on sorted data..."
  }
]
```

### Teacher Development Plan Generation

Route:
- `POST /api/v1/agents/teacher/plan-generation`

This route currently supports the legacy request body already sent by `zivai-web` and returns a `Plan`-shaped object that the teacher workspace can save and render immediately.

Current request body:

```json
{
  "firstName": "Tariro",
  "lastName": "Moyo",
  "subjectName": "Computer Science",
  "subjectID": "subject-uuid",
  "currentOverallScore": "62.5%",
  "potentialOverallScore": "73%",
  "targetScore": "85%",
  "overallPerformance": "Average",
  "overallEngagement": "Medium",
  "attributeDetails": [
    {
      "name": "Algorithms",
      "currentScore": "55%",
      "potentialScore": "70%",
      "targetScore": "75%",
      "gap": "20%",
      "weight": "1"
    },
    {
      "name": "Problem Solving",
      "currentScore": "60%",
      "potentialScore": "78%",
      "targetScore": "80%",
      "gap": "20%",
      "weight": "1"
    }
  ],
  "context": "Focus on actionable steps, varied resources, and clear goals.",
  "referenceDocuments": []
}
```

Current response body:

```json
{
  "name": "Computer Science Development Plan",
  "description": "Personalized plan for Tariro Moyo targeting Algorithms and Problem Solving.",
  "progress": 0,
  "potentialOverall": 78,
  "eta": 28,
  "performance": "Average",
  "skills": [
    {
      "name": "Algorithms",
      "score": 75,
      "subskills": [
        {
          "name": "Algorithms mastery target",
          "score": 75,
          "color": "yellow"
        }
      ]
    },
    {
      "name": "Problem Solving",
      "score": 80,
      "subskills": [
        {
          "name": "Problem Solving mastery target",
          "score": 80,
          "color": "yellow"
        }
      ]
    }
  ],
  "steps": [
    {
      "title": "Review core concepts for Algorithms",
      "type": "document",
      "content": "<p><strong>Critical Skill Focus:</strong> Algorithms</p><p><strong>Subject:</strong> Computer Science</p><p><strong>Teacher Objective:</strong> Close the learner's gaps in Algorithms and Problem Solving.</p><p><strong>Guidance:</strong> Use scaffolded instruction, concrete examples, focused practice, and short mastery checks.</p><p><strong>Why this matters:</strong> Current 55% vs target 75% (gap 20%).</p>",
      "link": "",
      "additionalResources": [],
      "order": 1
    }
  ],
  "subjectId": "subject-uuid",
  "referenceFallbackUsed": true,
  "criticalSkillsUsed": ["Algorithms", "Problem Solving"]
}
```

Behavior:
- The route returns a `Plan`-compatible object for the current teacher UI.
- Only critical skills with positive gaps are used when building the plan.
- Generated `steps` are normalized so they address the critical skills only.
- If the model drifts or returns unusable JSON, the service falls back to a deterministic plan instead of returning an empty workflow.
- If `referenceDocuments` are missing, empty, or unusable, the service falls back to the learner profile, critical-skill data, and general subject knowledge.

### Student Assessment

Route:
- `POST /api/v1/agents/student/assessment`

This route matches the current `zivai-web/src/services/externalAssessmentService.ts` contract.

Request:
- `multipart/form-data`
- send either:
  - `text` + `module`
  - or `file` + `module`

Examples:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agents/student/assessment" \
  -F "text=Binary search checks the middle element and works on sorted data." \
  -F "module=Binary Search"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agents/student/assessment" \
  -F "file=@/path/to/answer.jpg" \
  -F "module=Binary Search"
```

Current response body:

```json
{
  "module": "Binary Search",
  "filename": "answer.jpg",
  "content_type": "image/jpeg",
  "ocr_type": "ocr",
  "markdown": "Binary search checks the middle element and works on sorted data.",
  "pages": 1,
  "assessment": {
    "is_correct_module": true,
    "confidence_assessment_score": 0.82,
    "total_possible_marks": 10,
    "marks_achieved": 7,
    "marks_percentage": 70,
    "overall_feedback": "Your response is relevant to the module and shows partial understanding, but it needs more detail on the key conditions and process.",
    "strengths": [
      "Your response stays on the declared module.",
      "You identified one core idea correctly."
    ],
    "improvements": [
      "Explain the main process more fully.",
      "Add one or two more module-specific ideas."
    ],
    "criteria": [
      {
        "criterion": "Module relevance",
        "score": 8,
        "feedback": "The response stays on topic."
      },
      {
        "criterion": "Concept understanding",
        "score": 7,
        "feedback": "The answer shows partial conceptual understanding."
      },
      {
        "criterion": "Clarity and completeness",
        "score": 6,
        "feedback": "The response is understandable but not yet complete."
      }
    ],
    "assessment_details": {
      "response": {
        "max_marks": 10,
        "awarded_marks": 7,
        "feedback": "Your response is relevant to the module and shows partial understanding, but it needs more detail on the key conditions and process.",
        "improvement": "Explain the main process more fully."
      }
    },
    "detected_module": "Binary Search",
    "mark_consistency_check": "consistent",
    "marking_scheme_used": false
  }
}
```

Behavior:
- If `file` is provided, the route first extracts text using the OCR module.
- If the file contains digital text, it prefers direct extraction before OCR.
- If OCR/model JSON fails, the route falls back to a deterministic heuristic assessment instead of returning no feedback.
- This route is currently stateless and does not persist to the shared LMS database yet.

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

Primary payload-based question grading:
- route: `POST /api/v1/grade/question`
- intended caller: `core-backend`, using the canonical question + answer snapshot
- works with or without a rubric/guide

Example request body:

```json
{
  "request_context": {
    "assessment_attempt_id": "b9d0a4d8-1c1d-4e0b-9f08-6d5e8b7a1111",
    "attempt_answer_id": "f7c85d3f-2a5f-47f8-8c4e-3f2d30b5a123",
    "assessment_question_id": "11111111-2222-3333-4444-555555555555",
    "question_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "student_id": "99999999-8888-7777-6666-555555555555",
    "school_id": "12345678-1234-1234-1234-123456789012"
  },
  "question": {
    "text": "Explain what binary search is and state two conditions required before it can be used.",
    "subject": "Computer Science",
    "topic": "Search Algorithms",
    "question_type": "short_answer",
    "max_marks": 5
  },
  "student_answer": {
    "text": "Binary search checks the middle item and keeps searching in the half where the answer could be. It works when the data is sorted."
  },
  "marking_guide": {
    "rubric_items": [
      {
        "index": 1,
        "description": "Defines binary search as checking the middle element and halving the search space",
        "marks": 3,
        "keywords": ["middle", "halving", "search space"]
      },
      {
        "index": 2,
        "description": "States that the data must be sorted",
        "marks": 1,
        "keywords": ["sorted", "ordered"]
      }
    ],
    "expected_answer": null,
    "expected_points": []
  },
  "options": {
    "dry_run": false,
    "force": false,
    "allow_holistic_fallback": true
  }
}
```

Primary payload-based whole-assessment grading:
- route: `POST /api/v1/grade/assessment`
- grades every question payload in the request body and returns question-level feedback

Example request body:

```json
{
  "request_context": {
    "assessment_attempt_id": "b9d0a4d8-1c1d-4e0b-9f08-6d5e8b7a1111",
    "student_id": "99999999-8888-7777-6666-555555555555",
    "school_id": "12345678-1234-1234-1234-123456789012"
  },
  "questions": [
    {
      "request_context": {
        "attempt_answer_id": "f7c85d3f-2a5f-47f8-8c4e-3f2d30b5a123",
        "assessment_question_id": "11111111-2222-3333-4444-555555555555",
        "question_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
      },
      "question": {
        "text": "Explain what binary search is and state two conditions required before it can be used.",
        "subject": "Computer Science",
        "topic": "Search Algorithms",
        "question_type": "short_answer",
        "max_marks": 5
      },
      "student_answer": {
        "text": "Binary search checks the middle item and keeps searching in the half where the answer could be. It works when the data is sorted."
      },
      "marking_guide": {
        "rubric_items": [
          {
            "index": 1,
            "description": "Defines binary search as checking the middle element and halving the search space",
            "marks": 3
          },
          {
            "index": 2,
            "description": "States that the data must be sorted",
            "marks": 1
          }
        ]
      }
    }
  ],
  "options": {
    "dry_run": false,
    "force": false,
    "allow_holistic_fallback": true
  }
}
```

Payload grading behavior:
- If a marking guide exists, the LLM grades per rubric item.
- If no marking guide exists, the service falls back to holistic grading.
- The response always includes `feedback_summary`, `strengths`, `missing_points`, and `next_steps`.
- Payload routes are stateless and currently do not write back to the DB.

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
- The active DKT artifact is `models/dkt/dkt_lstm_cloud.ckpt` with `models/dkt/skill_map_v1.json`.
- Deprecated standalone ASAG CRUD routes and local persistence models have been removed.
- Grading now uses shared LMS/AI contracts instead of local SQLite-era tables.
- Broader AI workflow endpoints beyond grading are still to be built.

## DKT Endpoints

The DKT integration uses the checkpoint and frozen skill map described in `msmodels/workspace/dkt/update/README_DKT_MODEL.md`, but the live service now reads them from `models/dkt/`.

Active artifacts:
- `models/dkt/dkt_lstm_cloud.ckpt`
- `models/dkt/skill_map_v1.json`
- `models/dkt/model_meta.json`
- `models/dkt/dkt_lstm_edge.mindir` (kept for later edge work, not the primary backend runtime)

### Update student mastery

Route:
- `POST /api/v1/dkt/update`

Request body:

```json
{
  "student_id": "8a93f4a1-4a96-4ba0-a331-56bf8f1a5e18",
  "subject_code": "computer_science",
  "events": [
    {
      "skill_code": "CS.F3.ALG.DEBUG_ALGORITHMS",
      "is_correct": 1,
      "score": 2.0,
      "max_score": 2.0,
      "event_time": "2026-03-18T18:00:00Z",
      "assessment_attempt_id": null,
      "attempt_answer_id": null
    }
  ],
  "persist": true,
  "include_mastery_vector": false
}
```

Behavior:
- loads existing `lms.interaction_events` history for the student and subject
- appends the incoming events
- runs DKT inference in MindSpore
- writes:
  - `lms.interaction_events`
  - `lms.mastery_snapshots`
  - `lms.mastery_snapshot_skills`
  - `lms.student_attributes`
  - `ai.ai_inference_runs`

Response body:

```json
{
  "status": "ok",
  "result": {
    "student_id": "8a93f4a1-4a96-4ba0-a331-56bf8f1a5e18",
    "subject_id": "6f2f68e0-3f55-4a5d-a2e5-0f7ef983f5c5",
    "subject_code": "computer_science",
    "average_mastery": 0.612341,
    "risk_level": "medium",
    "weak_skills": [
      {
        "skill_code": "CS.F3.ALG.DEBUG_ALGORITHMS",
        "mastery_prob": 0.233114,
        "skill_name": "Debug algorithms",
        "skill_id": "f1d8c87d-1dc5-48ad-8bd5-d72f7d2f7af0"
      }
    ],
    "mastery_vector": null,
    "snapshot_id": "0d2c4b48-2c4d-4d5a-b201-c48be9c30878",
    "trace_id": "dkt-3f1f3d52bcb24d23",
    "persisted": true,
    "events_applied": 1,
    "ignored_skill_codes": [],
    "snapshot_time": "2026-03-18T18:01:12.302124Z",
    "model_name": "DKT Computer Science Form 3-4",
    "model_version": "2026-03-05T20:46:38.288227+00:00"
  }
}
```

### Get latest student mastery

Route:
- `GET /api/v1/dkt/mastery/<student_id>?subject_code=computer_science`

Optional query params:
- `subject_id`
- `subject_code`
- `full=true` to include the full mastery vector
- `refresh=true` to compute from interaction history if no snapshot is present

Response body:

```json
{
  "status": "ok",
  "result": {
    "student_id": "8a93f4a1-4a96-4ba0-a331-56bf8f1a5e18",
    "subject_id": "6f2f68e0-3f55-4a5d-a2e5-0f7ef983f5c5",
    "subject_code": "computer_science",
    "average_mastery": 0.612341,
    "risk_level": "medium",
    "weak_skills": [],
    "mastery_vector": null,
    "snapshot_id": "0d2c4b48-2c4d-4d5a-b201-c48be9c30878",
    "snapshot_time": "2026-03-18T18:01:12.302124Z",
    "source": "dkt_update",
    "trace_id": null
  }
}
```
