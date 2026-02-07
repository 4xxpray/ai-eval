#!/bin/bash
# Posts evaluation results as PR comment
# Usage: ./scripts/pr-comment.sh <results.json>

set -euo pipefail

results_path="${1:-}"
if [[ -z "$results_path" ]]; then
  echo "usage: $0 <results.json>" >&2
  exit 1
fi

if [[ ! -f "$results_path" ]]; then
  echo "results file not found: $results_path" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found, skipping PR comment" >&2
  exit 0
fi

case "${GITHUB_EVENT_NAME:-}" in
  pull_request|pull_request_target) ;;
  *)
    echo "not a pull_request event, skipping PR comment" >&2
    exit 0
    ;;
esac

if [[ -z "${GITHUB_EVENT_PATH:-}" || ! -f "$GITHUB_EVENT_PATH" ]]; then
  echo "GITHUB_EVENT_PATH missing, skipping PR comment" >&2
  exit 0
fi

pr_number=""
if command -v jq >/dev/null 2>&1; then
  pr_number="$(jq -r '.pull_request.number // empty' "$GITHUB_EVENT_PATH")"
fi

if [[ -z "$pr_number" ]]; then
  echo "PR number not found, skipping PR comment" >&2
  exit 0
fi

body_file="$(mktemp)"
trap 'rm -f "$body_file"' EXIT

if command -v jq >/dev/null 2>&1; then
  threshold="$(jq -r '.threshold // empty' "$results_path")"
  summary_line="$(jq -r '.summary | "Suites: \(.total_suites) | Cases: \(.total_cases) | Passed: \(.passed_cases) | Failed: \(.failed_cases)"' "$results_path")"
  rows="$(jq -r '.suites[] |
    if (.error // "") != "" then
      "| \((.prompt | tostring | if length > 0 then . else "-" end) | gsub("[\r\n]"; " ") | gsub("\\|"; "\\\\|")) | \((.suite | tostring | if length > 0 then . else "-" end) | gsub("[\r\n]"; " ") | gsub("\\|"; "\\\\|")) | - | - | - | - | - | \((.error | tostring) | gsub("[\r\n]"; " ") | gsub("\\|"; "\\\\|")) |"
    else
      "| \((.prompt | tostring | if length > 0 then . else "-" end) | gsub("[\r\n]"; " ") | gsub("\\|"; "\\\\|")) | \((.suite | tostring | if length > 0 then . else "-" end) | gsub("[\r\n]"; " ") | gsub("\\|"; "\\\\|")) | \(.total_cases) | \(.passed_cases) | \(.failed_cases) | \(.pass_rate | @sprintf "%.3f") | \(.avg_score | @sprintf "%.3f") | - |"
    end' "$results_path")"
  {
    echo "## Prompt Evaluation Results"
    echo
    if [[ -n "$threshold" && "$threshold" != "null" ]]; then
      echo "Threshold: $threshold"
      echo
    fi
    echo "$summary_line"
    echo
    echo "| Prompt | Suite | Cases | Passed | Failed | Pass Rate | Avg Score | Error |"
    echo "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    echo "$rows"
  } > "$body_file"
else
  {
    echo "Prompt evaluation results (raw JSON):"
    cat "$results_path"
  } > "$body_file"
fi

gh pr comment "$pr_number" --body-file "$body_file"
