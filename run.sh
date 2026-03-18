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
