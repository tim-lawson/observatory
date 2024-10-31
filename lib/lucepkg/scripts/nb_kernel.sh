#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd $SCRIPT_DIR

# Source shellenv to get the clr command
. shellenv.sh

# Check if at least one package name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <package1> [package2] [package3] ..."
    exit 1
fi

# Register each package's kernel
for pkg in "$@"; do
    clr activate $pkg
    .venv/bin/python -m ipykernel install --user --name=$pkg --display-name "TL Remote: $pkg"
done
