#!/bin/bash

INPUT_PATH="${1:-.}"

if [ -d "$INPUT_PATH" ]; then
  PROJECT_DIR="$INPUT_PATH"
  REPORT_DIR="$PROJECT_DIR/static_reports"
  PY_FILES=$(find "$PROJECT_DIR" -type f -name "*.py")
else
  PY_FILES="$INPUT_PATH"
  PROJECT_DIR=$(pwd)
  REPORT_DIR="$PROJECT_DIR/static_reports"
fi

PYLINT_RC="$PROJECT_DIR/.pylintrc"
mkdir -p "$REPORT_DIR"

echo
echo "=== Running Per-File Static Analysis ==="
echo

# Loop over each file
for file in $PY_FILES; do
    relname=$(realpath --relative-to="$PROJECT_DIR" "$file")
    escaped=$(echo "$relname" | tr '/' '_' | tr '.' '_')

    echo
    echo "-> Analyzing $relname"

    pylint --rcfile="$PYLINT_RC" --output-format=json "$file" > "$REPORT_DIR/pylint_$escaped.json" 2>/dev/null
    mypy "$file" > "$REPORT_DIR/mypy_$escaped.txt" 2>&1
    bandit -f json -o "$REPORT_DIR/bandit_$escaped.json" "$file" > /dev/null
    vulture "$file" > "$REPORT_DIR/vulture_$escaped.txt" 2>&1
    dodgy "$file" > "$REPORT_DIR/dodgy_$escaped.txt" 2>&1
done

# Safety runs once for whole env (dependencies)
echo
echo "-> Running safety (once for full environment)"
safety check --full-report > "$REPORT_DIR/safety.txt" 2>&1

echo
echo "✅ Per-file static analysis complete."
echo "Reports written to: $REPORT_DIR"
