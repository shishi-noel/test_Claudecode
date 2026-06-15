#!/bin/bash
# Auto-sync CLAUDE.md to Git so Obsidian Git plugin can pull it.
# Triggered by Claude Code PostToolUse hook after Write/Edit tool calls.
# Receives tool use JSON on stdin.

INPUT=$(cat)

# Extract file_path from tool input JSON
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null)

# Only process CLAUDE.md files
if [[ "$FILE_PATH" != *"CLAUDE.md" ]]; then
    exit 0
fi

# Find git repo root from the file's directory
REPO_ROOT=$(git -C "$(dirname "$FILE_PATH")" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    exit 0
fi

cd "$REPO_ROOT" || exit 0

BRANCH=$(git branch --show-current)
if [ -z "$BRANCH" ]; then
    exit 0
fi

# Stage CLAUDE.md
git add CLAUDE.md

# Skip if nothing changed
if git diff --cached --quiet; then
    exit 0
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "docs: sync CLAUDE.md [${TIMESTAMP}]"

# Push with retry (exponential backoff)
for WAIT in 0 2 4 8 16; do
    if [ "$WAIT" -gt 0 ]; then
        sleep "$WAIT"
    fi
    if git push -u origin "$BRANCH" 2>&1; then
        echo "[obsidian-sync] CLAUDE.md pushed to ${BRANCH}"
        exit 0
    fi
done

echo "[obsidian-sync] ERROR: push failed after retries" >&2
exit 1
