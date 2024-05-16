#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"

# load .env
set -a # Automatically export subsequent variables
[ -f .env ] && . .env
set +a # Cancel automatically export

# Run the tests
pytest -s --cov=mosstool --cov-report=html --cov-report=term-missing
