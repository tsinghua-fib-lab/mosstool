#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd ${PROJECT_DIR}
pip3 install .
pdoc -o docs/ -d markdown ./mosstool
# 查找目录下的所有文件，并对每个文件执行sed命令
find "docs/" -name "*.html" -type f -exec sed -i 's/mosstool.html/mosstool/g' {} +
