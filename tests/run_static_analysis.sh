#!/bin/bash

PROJECT_DIR="${1:-.}"
REPORT_DIR="$PROJECT_DIR/static_reports"
PYLINT_RC="$PROJECT_DIR/.pylintrc"

mkdir -p "$REPORT_DIR"

echo
echo "=== Running Static Analysis in WSL2 ==="
echo

# Run Pylint (semantic only)
echo "-> Running pylint"
pylint --rcfile="$PYLINT_RC" --output-format=json "$PROJECT_DIR" > "$REPORT_DIR/pylint.json" 2>/dev/null

# Run MyPy
echo "-> Running mypy"
mypy "$PROJECT_DIR" > "$REPORT_DIR/mypy.txt" 2>&1

# Run Bandit
echo "-> Running bandit"
bandit -r "$PROJECT_DIR" -f json -o "$REPORT_DIR/bandit.json"

# Run Vulture
echo "-> Running vulture"
vulture "$PROJECT_DIR" > "$REPORT_DIR/vulture.txt" 2>&1

# Run Safety (Vulnerabilities)
echo "-> Running safety"
safety check --full-report > "$REPORT_DIR/safety.txt" 2>&1

# Run Dodgy (Secrets)
echo "-> Running dodgy"
dodgy "$PROJECT_DIR" > "$REPORT_DIR/dodgy.txt" 2>&1

# Generate Markdown Summary
SUMMARY="$REPORT_DIR/summary_report.md"
echo "## Static Analysis Summary" > "$SUMMARY"
echo "" >> "$SUMMARY"

# Pylint Summary
echo "### Pylint (Semantic Errors Only)" >> "$SUMMARY"
if [[ -s "$REPORT_DIR/pylint.json" ]]; then
    jq -r '.[] | "- [" + (.type | ascii_upcase) + "] `" + .path + ":" + (.line|tostring) + "` - " + .message + " (`" + .symbol + "`)"' "$REPORT_DIR/pylint.json" >> "$SUMMARY"
else
    echo "- No output from pylint." >> "$SUMMARY"
fi

# MyPy Summary
echo -e "\n### MyPy" >> "$SUMMARY"
sed 's/^/- /' "$REPORT_DIR/mypy.txt" >> "$SUMMARY"

# Bandit Summary
echo -e "\n### Bandit (Security)" >> "$SUMMARY"
if [[ -s "$REPORT_DIR/bandit.json" ]]; then
    jq -r '.results[] | "- [SEVERITY: " + .issue_severity + "] `" + .filename + ":" + (.line_number|tostring) + "` - " + .issue_text' "$REPORT_DIR/bandit.json" >> "$SUMMARY"
else
    echo "- No Bandit output." >> "$SUMMARY"
fi

# Vulture
echo -e "\n### Vulture (Dead Code)" >> "$SUMMARY"
sed 's/^/- /' "$REPORT_DIR/vulture.txt" >> "$SUMMARY"

# Safety
echo -e "\n### Safety (Vulnerable Dependencies)" >> "$SUMMARY"
sed 's/^/- /' "$REPORT_DIR/safety.txt" >> "$SUMMARY"

# Dodgy
echo -e "\n### Dodgy (Secrets / Insecure Patterns)" >> "$SUMMARY"
sed 's/^/- /' "$REPORT_DIR/dodgy.txt" >> "$SUMMARY"

echo
echo "✅ Static analysis complete."
echo "Summary saved to: $SUMMARY"
