set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

select_venv_activate() {
  # 1) explicit override (e.g. ASAG_VENV=/path/to/venv)
  if [ -n "${ASAG_VENV:-}" ] && [ -f "${ASAG_VENV}/bin/activate" ]; then
    echo "${ASAG_VENV}/bin/activate"
    return
  fi

  # 2) active venv if it belongs to this project
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
    case "$VIRTUAL_ENV" in
      "$PROJECT_ROOT"/*)
        echo "${VIRTUAL_ENV}/bin/activate"
        return
        ;;
    esac
  fi

  # 3) preferred local env names
  if [ -f "$PROJECT_ROOT/.engine-venv/bin/activate" ]; then
    echo "$PROJECT_ROOT/.engine-venv/bin/activate"
    return
  fi
  if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "$PROJECT_ROOT/.venv/bin/activate"
    return
  fi

  # 4) any active venv as last resort
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
    echo "${VIRTUAL_ENV}/bin/activate"
    return
  fi

  echo ""
}

ACTIVATE_PATH="$(select_venv_activate)"
if [ -z "$ACTIVATE_PATH" ]; then
  echo "No virtual environment found. Create one (.engine-venv or .venv) and install dependencies."
  exit 1
fi
source "$ACTIVATE_PATH"
echo "[venv] using $VIRTUAL_ENV"
export PYTHONPATH="$PWD/src"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export MS_DEVICE_TARGET="${MS_DEVICE_TARGET:-CPU}"
export MS_STRICT_DEVICE="${MS_STRICT_DEVICE:-true}"
export ASAG_ENABLED="${ASAG_ENABLED:-true}"
export ASAG_MODEL_DIR="${ASAG_MODEL_DIR:-$PROJECT_ROOT/models/asag}"
export ASAG_CKPT_PATH="${ASAG_CKPT_PATH:-$ASAG_MODEL_DIR/asag_mohler_best.ckpt}"
export ASAG_MINDIR_PATH="${ASAG_MINDIR_PATH:-$ASAG_MODEL_DIR/asag_mohler.mindir}"
export ASAG_VOCAB_PATH="${ASAG_VOCAB_PATH:-$ASAG_MODEL_DIR/tokenizer/vocab.json}"
export ASAG_RESULTS_PATH="${ASAG_RESULTS_PATH:-$ASAG_MODEL_DIR/results.json}"
export ASAG_MAX_LENGTH="${ASAG_MAX_LENGTH:-256}"
export ASAG_HIDDEN_SIZE="${ASAG_HIDDEN_SIZE:-256}"
export ASAG_DROPOUT_PROB="${ASAG_DROPOUT_PROB:-0.2}"
export ASAG_RAW_SCORE_MAX="${ASAG_RAW_SCORE_MAX:-5.0}"
export ASAG_SHORT_ANSWER_MAX_MARKS="${ASAG_SHORT_ANSWER_MAX_MARKS:-5.0}"
export DKT_MODEL_DIR="${DKT_MODEL_DIR:-$PROJECT_ROOT/models/dkt}"
export DKT_CLOUD_CKPT_PATH="${DKT_CLOUD_CKPT_PATH:-$DKT_MODEL_DIR/dkt_lstm_cloud.ckpt}"
export DKT_SKILL_MAP_PATH="${DKT_SKILL_MAP_PATH:-$DKT_MODEL_DIR/skill_map_v1.json}"
export DKT_MODEL_META_PATH="${DKT_MODEL_META_PATH:-$DKT_MODEL_DIR/model_meta.json}"
export DKT_EDGE_MINDIR_PATH="${DKT_EDGE_MINDIR_PATH:-$DKT_MODEL_DIR/dkt_lstm_edge.mindir}"
export DKT_DEFAULT_SUBJECT_CODE="${DKT_DEFAULT_SUBJECT_CODE:-computer_science}"
export DKT_WEAK_SKILL_LIMIT="${DKT_WEAK_SKILL_LIMIT:-5}"
export HWC_OCR_ENDPOINT="${HWC_OCR_ENDPOINT:-https://ocr.ap-southeast-1.myhuaweicloud.com}"
export HWC_HOST="${HWC_HOST:-ocr.ap-southeast-1.myhuaweicloud.com}"
export HWC_FORCE_TRAILING_SLASH="${HWC_FORCE_TRAILING_SLASH:-false}"
export HWC_HTTP_TIMEOUT_SECONDS="${HWC_HTTP_TIMEOUT_SECONDS:-180}"
export HWC_GENERAL_TEXT_DETECT_DIRECTION="${HWC_GENERAL_TEXT_DETECT_DIRECTION:-true}"
export HWC_GENERAL_TEXT_QUICK_MODE="${HWC_GENERAL_TEXT_QUICK_MODE:-false}"
export HWC_GENERAL_TEXT_MAX_ORIGINAL_FILE_SIZE_BYTES="${HWC_GENERAL_TEXT_MAX_ORIGINAL_FILE_SIZE_BYTES:-7340032}"
export HWC_GENERAL_TEXT_MAX_ENCODED_IMAGE_BYTES="${HWC_GENERAL_TEXT_MAX_ENCODED_IMAGE_BYTES:-2500000}"
export HWC_GENERAL_TEXT_MAX_IMAGE_WIDTH="${HWC_GENERAL_TEXT_MAX_IMAGE_WIDTH:-2200}"
export HWC_GENERAL_TEXT_MAX_IMAGE_HEIGHT="${HWC_GENERAL_TEXT_MAX_IMAGE_HEIGHT:-2200}"
export HWC_GENERAL_TEXT_JPEG_QUALITY="${HWC_GENERAL_TEXT_JPEG_QUALITY:-0.72}"
export HWC_GENERAL_TEXT_MIN_JPEG_QUALITY="${HWC_GENERAL_TEXT_MIN_JPEG_QUALITY:-0.50}"
export HWC_GENERAL_TEXT_ADAPTIVE_RESIZE_PERCENT="${HWC_GENERAL_TEXT_ADAPTIVE_RESIZE_PERCENT:-85}"
export HWC_GENERAL_TEXT_MAX_ADAPTIVE_PASSES="${HWC_GENERAL_TEXT_MAX_ADAPTIVE_PASSES:-6}"
export HWC_GENERAL_TEXT_PDF_RENDER_DPI="${HWC_GENERAL_TEXT_PDF_RENDER_DPI:-200}"
export HWC_GENERAL_TEXT_MAX_PDF_PAGES="${HWC_GENERAL_TEXT_MAX_PDF_PAGES:-50}"

# Expose CUDA/cuDNN runtime libs installed via pip (torch nvidia-* wheels).
SITE_PKG="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

# MindNLP 0.5.1 imports diffusers unconditionally and can crash at import-time
# in MindSpore-only setups. Apply an idempotent local patch in the venv.
MINDSNLP_INIT="$SITE_PKG/mindnlp/__init__.py"
if [ -f "$MINDSNLP_INIT" ]; then
  python - "$MINDSNLP_INIT" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
txt = path.read_text(encoding="utf-8")
needle = "\nfrom . import diffusers\n"
patched = "\n# from . import diffusers  # patched by run.sh\n"
if needle in txt:
    path.write_text(txt.replace(needle, patched), encoding="utf-8")
    print(f"[patch] disabled mindnlp diffusers auto-import in {path}")
PY
fi

if [ "${MS_DEVICE_TARGET^^}" = "GPU" ]; then
  CUDA_LIB_PATHS=(
    "$SITE_PKG/nvidia/cudnn/lib"
    "$SITE_PKG/nvidia/cuda_runtime/lib"
    "$SITE_PKG/nvidia/cuda_nvrtc/lib"
    "$SITE_PKG/nvidia/cublas/lib"
    "$SITE_PKG/nvidia/cufft/lib"
    "$SITE_PKG/nvidia/cusolver/lib"
    "$SITE_PKG/nvidia/curand/lib"
    "$SITE_PKG/nvidia/cusparse/lib"
    "$SITE_PKG/nvidia/nvtx/lib"
    "$SITE_PKG/nvidia/nvjitlink/lib"
    "/lib/x86_64-linux-gnu"
    "/usr/lib/x86_64-linux-gnu"
  )
  for p in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "$p" ]; then
      export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
    fi
  done

  REQUIRED_CUDA_SOS=(
    "$SITE_PKG/nvidia/cublas/lib/libcublas.so.11"
    "$SITE_PKG/nvidia/curand/lib/libcurand.so.10"
    "$SITE_PKG/nvidia/cudnn/lib/libcudnn.so.8"
    "$SITE_PKG/nvidia/cuda_runtime/lib/libcudart.so.11.0"
    "$SITE_PKG/nvidia/cusolver/lib/libcusolver.so.11"
    "$SITE_PKG/nvidia/cufft/lib/libcufft.so.10"
    "$SITE_PKG/nvidia/cusparse/lib/libcusparse.so.11"
    "$SITE_PKG/nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2"
  )

  missing_count=0
  for so_path in "${REQUIRED_CUDA_SOS[@]}"; do
    if [ ! -f "$so_path" ]; then
      echo "[missing] $so_path"
      missing_count=$((missing_count + 1))
    fi
  done

  if [ "$missing_count" -gt 0 ]; then
    echo
    echo "Missing CUDA/cuDNN cu11 runtime libraries required by MindSpore GPU."
    echo "Run: ./scripts/install_cuda_deps.sh"
    echo "Then rerun: bash run.sh"
    exit 1
  fi
else
  echo "[mindspore] using ${MS_DEVICE_TARGET} mode; skipping GPU runtime checks"
fi

python -m asag_engine.api.app
