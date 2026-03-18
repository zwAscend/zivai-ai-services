#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

select_python_bin() {
  # 1) explicit override
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  # 2) active venv, but only if it belongs to this project
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    case "$VIRTUAL_ENV" in
      "$PROJECT_ROOT"/*)
        echo "${VIRTUAL_ENV}/bin/python"
        return
        ;;
    esac
  fi

  # 3) preferred local env names
  if [[ -x "$PROJECT_ROOT/.engine-venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/.engine-venv/bin/python"
    return
  fi
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/.venv/bin/python"
    return
  fi

  # 4) any active venv (last resort)
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return
  fi

  # 5) system fallback
  echo "python"
}

PYTHON_BIN="$(select_python_bin)"
CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip-zivai}"
WHEEL_DIR="${WHEEL_DIR:-$CACHE_DIR/wheels}"
PIP_TIMEOUT="${PIP_TIMEOUT:-180}"
PIP_RETRIES="${PIP_RETRIES:-30}"
MAX_DOWNLOAD_ATTEMPTS="${MAX_DOWNLOAD_ATTEMPTS:-3}"

CUDNN_SPEC="nvidia-cudnn-cu11==8.9.6.50"
CUDNN_URL="https://files.pythonhosted.org/packages/85/2d/3f083fcff1c302119f48e7b30a5f7b23db793f262f900943a9eb456b9e4d/nvidia_cudnn_cu11-8.9.6.50-py3-none-manylinux1_x86_64.whl"
CUDNN_SHA256="319a8f7ca3d65139f1b69998595c7076ae0e4271a325e5dfde50a3ca31f55584"
CUDNN_WHEEL="nvidia_cudnn_cu11-8.9.6.50-py3-none-manylinux1_x86_64.whl"

PACKAGES=(
  "nvidia-cublas-cu11==11.11.3.6"
  "nvidia-cuda-runtime-cu11==11.8.89"
  "nvidia-cuda-nvrtc-cu11==11.8.89"
  "nvidia-cufft-cu11==10.9.0.58"
  "nvidia-curand-cu11==10.3.0.86"
  "nvidia-cusolver-cu11==11.4.1.48"
  "nvidia-cusparse-cu11==11.7.5.86"
)

CONFLICTING_CU12_PACKAGES=(
  nvidia-cublas-cu12
  nvidia-cuda-cupti-cu12
  nvidia-cuda-nvrtc-cu12
  nvidia-cuda-runtime-cu12
  nvidia-cudnn-cu12
  nvidia-cufft-cu12
  nvidia-cufile-cu12
  nvidia-curand-cu12
  nvidia-cusolver-cu12
  nvidia-cusparse-cu12
  nvidia-cusparselt-cu12
  nvidia-nccl-cu12
  nvidia-nvjitlink-cu12
  nvidia-nvshmem-cu12
  nvidia-nvtx-cu12
)

mkdir -p "$CACHE_DIR"
mkdir -p "$WHEEL_DIR"

echo "[python] using $PYTHON_BIN"
"$PYTHON_BIN" -m pip --version

installed_version() {
  local pkg_name="$1"
  "$PYTHON_BIN" -m pip show "$pkg_name" 2>/dev/null | awk '/^Version:/{print $2}'
}

required_relpath_for_pkg() {
  local pkg_name="$1"
  case "$pkg_name" in
    nvidia-cublas-cu11) echo "nvidia/cublas/lib/libcublas.so.11" ;;
    nvidia-cuda-runtime-cu11) echo "nvidia/cuda_runtime/lib/libcudart.so.11.0" ;;
    nvidia-cuda-nvrtc-cu11) echo "nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2" ;;
    nvidia-cudnn-cu11) echo "nvidia/cudnn/lib/libcudnn.so.8" ;;
    nvidia-cufft-cu11) echo "nvidia/cufft/lib/libcufft.so.10" ;;
    nvidia-curand-cu11) echo "nvidia/curand/lib/libcurand.so.10" ;;
    nvidia-cusolver-cu11) echo "nvidia/cusolver/lib/libcusolver.so.11" ;;
    nvidia-cusparse-cu11) echo "nvidia/cusparse/lib/libcusparse.so.11" ;;
    *) echo "" ;;
  esac
}

sha256_of() {
  local file_path="$1"
  sha256sum "$file_path" | awk '{print $1}'
}

download_resume() {
  local url="$1"
  local target="$2"

  if command -v wget >/dev/null 2>&1; then
    wget \
      --continue \
      --tries=0 \
      --timeout=30 \
      --read-timeout=30 \
      --retry-connrefused \
      --waitretry=2 \
      --output-document="$target" \
      "$url"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl \
      --location \
      --continue-at - \
      --retry 100 \
      --retry-all-errors \
      --retry-delay 2 \
      --connect-timeout 15 \
      --output "$target" \
      "$url"
    return 0
  fi

  echo "Neither wget nor curl was found. Install one of them to enable resumable download."
  return 1
}

install_cudnn_resumable() {
  local pkg_name="${CUDNN_SPEC%%==*}"
  local wanted_version="${CUDNN_SPEC#*==}"
  local target="$WHEEL_DIR/$CUDNN_WHEEL"

  current_version="$(installed_version "$pkg_name" || true)"
  if [[ -n "$current_version" && "$current_version" == "$wanted_version" ]]; then
    echo "[skip] $pkg_name==$current_version already installed"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_DOWNLOAD_ATTEMPTS )); do
    echo "[download] $CUDNN_SPEC (attempt $attempt/$MAX_DOWNLOAD_ATTEMPTS)"
    download_resume "$CUDNN_URL" "$target"

    if [[ ! -f "$target" ]]; then
      echo "[error] Missing downloaded wheel: $target"
      return 1
    fi

    got_hash="$(sha256_of "$target")"
    if [[ "$got_hash" == "$CUDNN_SHA256" ]]; then
      echo "[ok] SHA256 verified for $CUDNN_WHEEL"
      "$PYTHON_BIN" -m pip install \
        --cache-dir "$CACHE_DIR" \
        --timeout "$PIP_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        "$target"
      return 0
    fi

    echo "[warn] SHA256 mismatch for $CUDNN_WHEEL"
    echo "       expected: $CUDNN_SHA256"
    echo "       got:      $got_hash"
    echo "       removing corrupt file and retrying."
    rm -f "$target"
    attempt=$((attempt + 1))
  done

  echo "[error] Failed to fetch a valid $CUDNN_SPEC after $MAX_DOWNLOAD_ATTEMPTS attempts."
  return 1
}

install_cudnn_resumable

echo "[cleanup] removing conflicting cu12 runtime wheels (if installed)"
for pkg in "${CONFLICTING_CU12_PACKAGES[@]}"; do
  if "$PYTHON_BIN" -m pip show "$pkg" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip uninstall -y "$pkg"
  fi
done

SITE_PKG="$("$PYTHON_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

for spec in "${PACKAGES[@]}"; do
  pkg_name="${spec%%==*}"
  wanted_version=""
  if [[ "$spec" == *"=="* ]]; then
    wanted_version="${spec#*==}"
  fi

  current_version="$(installed_version "$pkg_name" || true)"
  required_rel="$(required_relpath_for_pkg "$pkg_name")"
  required_abs=""
  if [[ -n "$required_rel" ]]; then
    required_abs="$SITE_PKG/$required_rel"
  fi

  if [[ -n "$current_version" ]]; then
    if [[ -z "$wanted_version" || "$current_version" == "$wanted_version" ]]; then
      if [[ -n "$required_abs" && ! -f "$required_abs" ]]; then
        echo "[repair] $pkg_name==$current_version is registered but missing $required_abs"
      else
        echo "[skip] $pkg_name==$current_version already installed"
        continue
      fi
    fi
  fi

  echo "[install] $spec"
  "$PYTHON_BIN" -m pip install \
    --force-reinstall \
    --no-deps \
    --cache-dir "$CACHE_DIR" \
    --timeout "$PIP_TIMEOUT" \
    --retries "$PIP_RETRIES" \
    "$spec"
done

echo "[verify] checking expected sonames for MindSpore GPU"
declare -a REQUIRED_SOS=(
  "$SITE_PKG/nvidia/cublas/lib/libcublas.so.11"
  "$SITE_PKG/nvidia/curand/lib/libcurand.so.10"
  "$SITE_PKG/nvidia/cudnn/lib/libcudnn.so.8"
  "$SITE_PKG/nvidia/cuda_runtime/lib/libcudart.so.11.0"
  "$SITE_PKG/nvidia/cusolver/lib/libcusolver.so.11"
  "$SITE_PKG/nvidia/cufft/lib/libcufft.so.10"
  "$SITE_PKG/nvidia/cusparse/lib/libcusparse.so.11"
  "$SITE_PKG/nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2"
)
for so_path in "${REQUIRED_SOS[@]}"; do
  if [[ ! -f "$so_path" ]]; then
    echo "[warn] missing $so_path"
  fi
done

echo "Done. CUDA Python dependencies installed (or already present)."
