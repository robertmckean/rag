---
name: push-project
description: Prepare, version, commit, and push a repository change safely. Use when the user asks Codex to push a project, cut a release, tag a version, update the changelog, refresh dependency snapshots, or perform a disciplined pre-push workflow for git changes.
---

# Push Project

This repository currently uses Windows PowerShell helper scripts and a Conda environment selected from `.conda-env.txt`. Prefer those conventions over ad hoc shell commands when validation or environment exports depend on Python packages.

Follow this workflow in order. Complete each step before moving to the next unless the repository clearly does not use that artifact.

1. Inspect the working tree with targeted git commands. Confirm the intended scope and call out untracked files that may need to be committed.
2. Review the diff so the pushed scope is explicit. Summarize the effective change set before modifying release artifacts.
3. Run the smallest useful validation for the changed area. In this repository, prefer PowerShell commands from the workspace root and use `tools\run_in_env.ps1` when a Python check depends on the configured Conda environment. If preparing a tagged release, prefer the broadest practical validation for the repository.
4. Refresh the dependency snapshot used by the repository, such as `requirements.txt`, `environment.yml`, or a lockfile, when the committed state should capture exact package versions. Do not add a second snapshot format unless the repo already uses it. If no snapshot file exists yet, inspect the repo docs and current files before introducing one.
5. Determine the next version number from the repository's existing scheme. Inspect tags, changelog entries, package metadata, and recent commits instead of guessing. Unless the user says otherwise, treat this repository's default release increment as the lowest component in `v0.1.n`, meaning bump `n` only. If the repo is still a scaffold and no scheme exists, say so clearly before inventing one.
6. Add or update the changelog entry for that version. Keep it scoped to the actual diff. If the repository does not yet contain a changelog location, confirm whether one should be created rather than assuming a format.
7. Update stale documentation or agent guidance affected by the change, including `AGENTS.md`, `README.md`, `docs\project_summary.md`, or environment notes when they are inaccurate. This repository currently contains placeholder docs, so release prep should check them explicitly.
8. Stage only the intended files. Avoid broad `git add .` unless the repository state is intentionally all-inclusive and has been reviewed.
9. Create a descriptive versioned commit message that matches the repo's existing style. If no stable style is visible, prefer `v0.1.n - <short release summary>` with the chosen version substituted in.
10. Push the current branch. If the repository uses release tags, create the matching tag and push it after the commit succeeds. When no contrary convention is visible, use a lightweight tag name equal to the version string, such as `v0.1.4`.

## Execution Rules

- Read repository conventions before choosing version numbers, changelog format, or dependency snapshot commands.
- Unless instructed otherwise, bump only the lowest version component used by this repository's current scheme. For the stated default `v0.1.n` pattern, increment `n`.
- Prefer minimal reversible edits. Do not rewrite unrelated docs while doing release prep.
- Treat a dirty worktree as normal. Never revert unrelated changes you did not make.
- If push, tag, or environment export requires sandbox escalation, request approval with a short justification rather than skipping the step.
- If validation fails, stop the release flow, report the failure clearly, and fix only if the user asked for end-to-end completion.
- If `.git` metadata is unavailable in the current workspace snapshot, explain that version, tag, and push steps cannot be completed from local inspection alone.

## Repository Notes

- Use PowerShell-native commands by default.
- Use the Conda environment from `.conda-env.txt`; the current configured value is `drum310`.
- Prefer `tools\run_in_env.ps1 <script_path> [args...]` for Python entry points that need the project environment.
- `README.md` and `docs\project_summary.md` are scaffold placeholders today, so they are weak sources of release truth until updated.
- `AGENTS.md` is the strongest local guidance source for validation style and minimal-change expectations.

## Useful Checks

- Use `git status --short` or `git status --short --branch` to inspect scope.
- Use `git diff --stat`, `git diff -- <path>`, and `git diff --cached` to separate unstaged and staged scope.
- Use `git tag --sort=-version:refname` and recent changelog entries to infer the next version.
- Prefer targeted test commands first; only run full suites when release scope or repo policy warrants it.
- Use `powershell -ExecutionPolicy Bypass -File .\tools\run_in_env.ps1 <script>` when validating Python code against the intended local environment.

## Default Release Wording

- Preferred commit message fallback: `v0.1.n - <short release summary>`
- Preferred tag fallback: `v0.1.n`
- Preferred changelog heading fallback: `## v0.1.n - YYYY-MM-DD`
- Keep the summary concrete and scoped to the shipped change, for example `v0.1.4 - Add PowerShell launch helpers`.

## Output Expectations

- Tell the user what changed scope is being pushed.
- Call out validation run and result.
- State the chosen version and tag explicitly.
- Mention any skipped step and why it did not apply.
