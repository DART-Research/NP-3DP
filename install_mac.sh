#!/usr/bin/env bash
# Bootstrap script for NP Slicing on macOS (and other Unix-like systems).
# Creates or recreates the local .venv, upgrades pip, and installs packages
# declared in requirements.txt. Requires Python 3.11+.
#
# Usage:
#   ./install_mac.sh
#   ./install_mac.sh --recreate
#   ./install_mac.sh --python /usr/local/bin/python3.11

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="${SCRIPT_DIR}"
PYTHON_BINARY=""
RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BINARY="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

check_python() {
  local candidate="$1"
  if ! command -v "$candidate" >/dev/null 2>&1; then
    return 1
  fi
  if "$candidate" -c 'import sys; exit(0 if (sys.version_info >= (3, 11)) else 1)'; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

resolve_python() {
  if [[ -n "$PYTHON_BINARY" ]]; then
    if check_python "$PYTHON_BINARY" >/dev/null; then
      printf '%s\n' "$PYTHON_BINARY"
      return 0
    fi
    echo "Provided Python executable '$PYTHON_BINARY' is not a usable Python 3.11+ interpreter." >&2
    exit 1
  fi

  local candidates=(
    python3.12
    python3.11
    python3
    python
  )

  for candidate in "${candidates[@]}"; do
    if resolved=$(check_python "$candidate"); then
      printf '%s\n' "$resolved"
      return 0
    fi
  done

  echo "Unable to locate a Python 3.11+ interpreter. Install Python or pass --python /path/to/python." >&2
  exit 1
}

PYTHON_CMD="$(resolve_python)"

pushd "$REPO_ROOT" >/dev/null

VENV_PATH="${REPO_ROOT}/.venv"
if [[ -d "$VENV_PATH" ]]; then
  if [[ $RECREATE -eq 1 ]]; then
    echo "Removing existing virtual environment at ${VENV_PATH}"
    rm -rf "$VENV_PATH"
  else
    echo ".venv already exists; it will be reused. Pass --recreate to rebuild."
  fi
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment (.venv)..."
  "$PYTHON_CMD" -m venv ".venv"
fi

VENV_PYTHON="${VENV_PATH}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment appears invalid (missing bin/python)." >&2
  exit 1
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "requirements.txt not found at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

echo "Installing dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

popd >/dev/null

cat <<'EOF'

Virtual environment ready.
Activate it via: source ./.venv/bin/activate
EOF
