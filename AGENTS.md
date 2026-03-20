# AGENTS.md

This file provides guidance to Codex and other coding agents when working in
this repository.

## Project Goal

Document the current project goal here in one or two concrete sentences.

## Project Status

- Keep this section current as the project evolves
- Call out the active milestone or priority
- Prefer the current code and config files over stale documentation

## Environment

- Use the project's intended local environment for Python and tooling commands
- Prefer commands that match the host shell and operating system
- Record any required external data, services, or credentials in project docs

## Working Style

- Investigate the user's stated issue before exploring alternatives
- Do not guess about behavior without reading the code that produces it
- Prefer minimal, reversible changes over broad rewrites
- Do not rename functions, classes, files, or move logic across files unless necessary
- Respect existing file boundaries unless the task requires a structural change

## Validation

- After code changes, run the smallest useful validation for the affected area
- Prefer targeted checks before full end-to-end runs
- If a command is environment-sensitive, verify it is running in the intended environment

## Dependencies

- Prefer backward-compatible fixes unless the task explicitly includes upgrades
- If a change depends on a newer package API, add a compatibility fallback or pin intentionally
- Avoid replacing core dependencies without a clear reason

## Documentation And Comments

- Keep documentation aligned with the current codebase
- Add comments only where they clarify non-obvious logic
- Do not force comment-heavy rewrites of otherwise clear code
- If guidance becomes stale, update it instead of silently ignoring it

## Review Priorities

- Prioritize regressions, stale guidance, environment mismatches, and unsafe workflows
- When reviewing changes, prefer concrete findings with file and line references over broad summaries
