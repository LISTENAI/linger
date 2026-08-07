#!/bin/sh

set -e

TARGET_CONDA_ENV=${LINGER_CONDA_ENV:-linger_thinker_3.x}

if [ -n "$TARGET_CONDA_ENV" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$TARGET_CONDA_ENV" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null || true)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate "$TARGET_CONDA_ENV"
        else
            echo "WARN: cannot locate conda.sh; continue with current Python."
        fi
    else
        echo "WARN: conda is not available; continue with current Python."
    fi
fi

PYTHON_BIN=${PYTHON:-python3}
MAX_JOBS=${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' was not found." >&2
    exit 1
fi

TORCH_CUDA_VERSION=$("$PYTHON_BIN" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)

if [ -z "${LINGER_CPU_ONLY+x}" ]; then
    if [ -z "$TORCH_CUDA_VERSION" ]; then
        LINGER_CPU_ONLY=1
    elif ! command -v nvcc >/dev/null 2>&1; then
        LINGER_CPU_ONLY=1
    else
        LINGER_CPU_ONLY=0
    fi
fi
export LINGER_CPU_ONLY

echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "GCC-version: $(command -v gcc || echo not found)"
echo "CUDA-version: $(command -v nvcc || echo not found)"
echo "CMAKE-version: $(command -v cmake || echo not found)"
echo "PyTorch CUDA: ${TORCH_CUDA_VERSION:-not available}"
echo "LINGER_CPU_ONLY: $LINGER_CPU_ONLY"
echo "MAX_JOBS: $MAX_JOBS"

"$PYTHON_BIN" -m pip uninstall linger -y || true
rm -rf dist/*.gz
if [ -e "build" ]; then
    rm -rf build
fi

# pip install -r requirements.txt

MAX_JOBS=$MAX_JOBS "$PYTHON_BIN" setup.py sdist
echo "MAX_JOBS-------------"
MAX_JOBS=$MAX_JOBS "$PYTHON_BIN" -m pip install dist/*.gz --no-build-isolation
