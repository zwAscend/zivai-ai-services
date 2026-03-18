#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

select_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    case "$VIRTUAL_ENV" in
      "$PROJECT_ROOT"/*)
        echo "${VIRTUAL_ENV}/bin/python"
        return
        ;;
    esac
  fi

  if [[ -x "$PROJECT_ROOT/.engine-venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/.engine-venv/bin/python"
    return
  fi
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/.venv/bin/python"
    return
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return
  fi

  echo "python"
}

PYTHON_BIN="$(select_python_bin)"
REQ_FILE="${REQ_FILE:-$PROJECT_ROOT/requirements.txt}"
CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip-zivai}"
WHEEL_DIR="${WHEEL_DIR:-$CACHE_DIR/wheels}"
PIP_TIMEOUT="${PIP_TIMEOUT:-240}"
PIP_RETRIES="${PIP_RETRIES:-40}"
PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-80}"
MAX_DOWNLOAD_ATTEMPTS="${MAX_DOWNLOAD_ATTEMPTS:-5}"

MINDSPORE_SPEC="mindspore==2.7.0"
MINDSPORE_WHEEL="mindspore-2.7.0-cp310-cp310-manylinux1_x86_64.whl"
MINDSPORE_URL="https://files.pythonhosted.org/packages/79/67/d040e1e1a6596b9fd26c88dcf07a34c7e169a0f5a32f20a33d01ab0fc7a7/mindspore-2.7.0-cp310-cp310-manylinux1_x86_64.whl"
MINDSPORE_SHA256_ALLOWLIST="${MINDSPORE_SHA256_ALLOWLIST:-7b110af7a8321ebb331480d287b974490678be832c01d9f1036240d2099249c9,57ee0c1f32855937dca0c68da8451f4c4ca5f90c307884f6288834dc9f4fa17d}"

mkdir -p "$CACHE_DIR" "$WHEEL_DIR"

echo "[python] using $PYTHON_BIN"
"$PYTHON_BIN" -m pip --version
"$PYTHON_BIN" -m pip install -U pip

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

  echo "Neither wget nor curl is installed."
  return 1
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

hash_allowed() {
  local got="$1"
  local old_ifs="$IFS"
  IFS=','
  for h in $MINDSPORE_SHA256_ALLOWLIST; do
    if [[ "$got" == "${h// /}" ]]; then
      IFS="$old_ifs"
      return 0
    fi
  done
  IFS="$old_ifs"
  return 1
}

install_mindspore_resumable() {
  local target="$WHEEL_DIR/$MINDSPORE_WHEEL"
  local attempt=1

  if "$PYTHON_BIN" -m pip show mindspore >/dev/null 2>&1; then
    local current_version
    current_version="$("$PYTHON_BIN" -m pip show mindspore | awk '/^Version:/{print $2}')"
    if [[ "$current_version" == "2.7.0" ]]; then
      echo "[skip] $MINDSPORE_SPEC already installed"
      return 0
    fi
  fi

  while (( attempt <= MAX_DOWNLOAD_ATTEMPTS )); do
    echo "[download] $MINDSPORE_SPEC (attempt $attempt/$MAX_DOWNLOAD_ATTEMPTS)"
    download_resume "$MINDSPORE_URL" "$target"

    if [[ ! -f "$target" ]]; then
      echo "[error] missing file: $target"
      return 1
    fi

    local got_hash
    got_hash="$(sha256_of "$target")"
    if hash_allowed "$got_hash"; then
      echo "[ok] SHA256 verified for $MINDSPORE_WHEEL ($got_hash)"
      "$PYTHON_BIN" -m pip install \
        --cache-dir "$CACHE_DIR" \
        --timeout "$PIP_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        --resume-retries "$PIP_RESUME_RETRIES" \
        "$target"
      return 0
    fi

    echo "[warn] SHA mismatch for $MINDSPORE_WHEEL"
    echo "       allowed:  $MINDSPORE_SHA256_ALLOWLIST"
    echo "       got:      $got_hash"
    echo "       deleting corrupt file and retrying."
    rm -f "$target"
    attempt=$((attempt + 1))
  done

  echo "[error] failed to fetch a valid $MINDSPORE_SPEC after $MAX_DOWNLOAD_ATTEMPTS attempts."
  return 1
}

install_mindspore_resumable

TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

# mindspore is installed from local wheel above to avoid repeated huge downloads.
grep -Ev '^[[:space:]]*mindspore==' "$REQ_FILE" > "$TMP_REQ"

echo "[install] remaining requirements from $REQ_FILE"
"$PYTHON_BIN" -m pip install \
  --cache-dir "$CACHE_DIR" \
  --timeout "$PIP_TIMEOUT" \
  --retries "$PIP_RETRIES" \
  --resume-retries "$PIP_RESUME_RETRIES" \
  -r "$TMP_REQ"

echo "Done. Requirements installed with persistent cache and resume settings."
