# 🎓 ASAG Engine – Option A (MindNLP / Pangu)
Automated Short Answer Grading engine for Computer Science (and other subjects) using **MindSpore + MindNLP** (or **Pangu NLP**) and a **rubric/markscheme-first** workflow.

✅ Teachers upload **Question Papers** + **Marking Schemes** (PDF/DOCX)  
✅ System extracts questions + marking points (no OCR / no MindOCR)  
✅ Students submit answers  
✅ Model grades strictly using rubric/markscheme and returns **validated JSON**  
✅ Stores audit trail: raw model output + validated result

---

## ✨ Features
- **Paper upload** (PDF/DOCX)
- **Markscheme upload** (PDF/DOCX)
- Auto **question extraction**
- Auto **rubric extraction** / mapping from markscheme → question ids
- **Single grading** and optional **batch grading**
- Strict **JSON output** with server-side validation + score clamping
- Runs locally with:
  - `LLM_PROVIDER=mindnlp` (Qwen2.5 etc.)
  - `LLM_PROVIDER=pangu` (Pangu checkpoints if compatible)

---

## 🧱 Tech Stack
- Flask (API)
- SQLAlchemy (ORM)
- PostgreSQL (recommended) or SQLite (dev)
- MindSpore + MindNLP for inference
- pdfplumber + python-docx for text extraction (no OCR)

---

---

## ✅ Requirements
- Python 3.10+
- (Optional) PostgreSQL 14+

Install system deps (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y python3-venv build-essential libpq-dev postgresql postgresql-contrib

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip


pip install -r requirements.txt

**##  .env file **

FLASK_ENV=production
HOST=127.0.0.1
PORT=8000
DEBUG=false

DATABASE_URL=postgresql+psycopg2://postgres:password@172.20.48.1:5432/asag_engine_cs
AUTO_CREATE_TABLES=true

# mindnlp | pangu
LLM_PROVIDER=mindnlp

# Smaller model for stability
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

# Tokens needed for grading JSON
MAX_NEW_TOKENS=256

# MindSpore execution mode
# GRAPH_MODE = faster but heavy memory
# PYNATIVE_MODE = safer for development

MS_MODE=PYNATIVE_MODE

# Uploads

UPLOAD_DIR=data/uploads

# Paper / Markscheme parsing
Chunk sizes (characters)

PAPER_CHUNK_SIZE=12000
PAPER_CHUNK_OVERLAP=800

MS_CHUNK_SIZE=14000
MS_CHUNK_OVERLAP=800

# Curriculum Alignment

ALIGN_BATCH_SIZE=5
