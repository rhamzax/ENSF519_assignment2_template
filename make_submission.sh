#!/bin/sh
set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

if command -v python3 >/dev/null 2>&1; then
    exec python3 "$script_dir/make_submission.py"
elif command -v python >/dev/null 2>&1; then
    exec python "$script_dir/make_submission.py"
else
    echo "Python 3 is required to create the submission archive." >&2
    exit 1
fi
