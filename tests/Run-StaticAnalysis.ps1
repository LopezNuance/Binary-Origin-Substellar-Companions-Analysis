param (
    [string]$ProjectPath = ".",
    [string]$ReportPath = "./static_reports"
)

# Ensure report directory exists
if (-Not (Test-Path $ReportPath)) {
    New-Item -ItemType Directory -Path $ReportPath | Out-Null
}

Write-Output "`n=== Starting Static Analysis ===`n"

# Tools to run
$pylintRc = ".pylintrc"

# Run Pylint
Write-Output "-> Running pylint (semantic checks only)..."
$pylintOut = Join-Path $ReportPath "pylint.json"
$pylintCmd = "pylint --rcfile=$pylintRc --output-format=json $ProjectPath > `"$pylintOut`""
Invoke-Expression $pylintCmd

# Run MyPy
Write-Output "-> Running mypy (type checking)..."
$mypyOut = Join-Path $ReportPath "mypy.txt"
mypy $ProjectPath > $mypyOut 2>&1

# Run Bandit
Write-Output "-> Running bandit (security)..."
$banditOut = Join-Path $ReportPath "bandit.json"
bandit -r $ProjectPath -f json -o $banditOut

# Run Vulture (dead code)
Write-Output "-> Running vulture (dead code)..."
$vultureOut = Join-Path $ReportPath "vulture.txt"
vulture $ProjectPath > $vultureOut 2>&1

# Run Safety (dependency vulnerabilities)
Write-Output "-> Running safety (CVE check)..."
$safetyOut = Join-Path $ReportPath "safety.txt"
safety check --full-report > $safetyOut 2>&1

# Run Dodgy (secrets detection)
Write-Output "-> Running dodgy (secrets)..."
$dodgyOut = Join-Path $ReportPath "dodgy.txt"
dodgy $ProjectPath > $dodgyOut 2>&1

# Generate Markdown Summary
$summaryPath = Join-Path $ReportPath "summary_report.md"
"## Static Analysis Summary`n" | Out-File $summaryPath

# Pylint Summary
"### Pylint (Errors only)" | Out-File $summaryPath -Append
if (Test-Path $pylintOut) {
    $pylintJson = Get-Content $pylintOut | ConvertFrom-Json
    foreach ($issue in $pylintJson) {
        $line = "- [$($issue.type.ToUpper())] `$($issue.path):$($issue.line)` - $($issue.message) (`$($issue.symbol)`)"
        $line | Out-File $summaryPath -Append
    }
} else {
    "- No pylint output" | Out-File $summaryPath -Append
}

# MyPy Summary
"### MyPy" | Out-File $summaryPath -Append
Get-Content $mypyOut | ForEach-Object { "- $_" } | Out-File $summaryPath -Append

# Bandit Summary
"### Bandit" | Out-File $summaryPath -Append
if (Test-Path $banditOut) {
    $banditJson = Get-Content $banditOut | ConvertFrom-Json
    foreach ($r in $banditJson.results) {
        "- [SEVERITY: $($r.issue_severity)] `$($r.filename):$($r.line_number)` - $($r.issue_text)" | Out-File $summaryPath -Append
    }
} else {
    "- No Bandit output" | Out-File $summaryPath -Append
}

# Vulture
"### Vulture (Dead Code)" | Out-File $summaryPath -Append
Get-Content $vultureOut | ForEach-Object { "- $_" } | Out-File $summaryPath -Append

# Safety
"### Safety (Dependency CVEs)" | Out-File $summaryPath -Append
Get-Content $safetyOut | ForEach-Object { "- $_" } | Out-File $summaryPath -Append

# Dodgy
"### Dodgy (Secrets / Bad Practices)" | Out-File $summaryPath -Append
Get-Content $dodgyOut | ForEach-Object { "- $_" } | Out-File $summaryPath -Append

Write-Output "`nAll reports saved in: $ReportPath"
Write-Output "Summary report written to: $summaryPath"
