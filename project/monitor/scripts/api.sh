#!/bin/bash
set -e
# Navigate to the parent directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd $SCRIPT_DIR

# Default values
PORT=8888
DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]
do
    case $1 in
        --port|-p)
        PORT="$2"
        shift 2
        ;;
        --dev|-d)
        DEV_MODE=true
        shift
        ;;
        *)
        shift
        ;;
    esac
done

# Set up the command
if [ "$DEV_MODE" = true ]; then
    COMMAND=".venv/bin/python3 -m uvicorn monitor.server:app --host 0.0.0.0 --port $PORT --reload"
else
    COMMAND=".venv/bin/python3 -m uvicorn monitor.server:app --host 0.0.0.0 --port $PORT --workers 1"
fi

# Execute the command
$COMMAND
