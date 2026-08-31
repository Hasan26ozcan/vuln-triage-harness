---
name: tracking-system-suppression-format
description: The vulnerability tracking system only recognizes standalone # NOSONAR comments; compound forms like # noqa: E501,NOSONAR and # nosec B608, NOSONAR are not parsed correctly
metadata:
  type: feedback
---

The vulnerability tracking system (SonarQube-style) recognizes `# NOSONAR` as a standalone comment token, but does NOT parse it when embedded within compound suppression comments.

**What works:** `# NOSONAR` as the sole annotation in the inline comment
**What doesn't work:** `# noqa: E501,NOSONAR` (ruff's noqa syntax with NOSONAR appended) or `# nosec B608, NOSONAR` (bandit's nosec syntax with NOSONAR appended)

**Why:** The tracking system's scanner does a simple substring/token match for `NOSONAR` as a comment token, not a regex search within `# noqa:` or `# nosec` comment bodies. Files like `app/security/paths.py` and `scripts/generate_training_data.py` that use standalone `# NOSONAR` are correctly suppressed. Files using combined formats like `# noqa: E501,NOSONAR` remain flagged even after the fix is applied.

**How to apply:** When adding suppressions, use `# NOSONAR` as a standalone comment. If you also need ruff suppression, handle E501 by restructuring code rather than combining with `# noqa`. For bandit, use `# nosec` on a separate mechanism or accept the bandit warning (bandit warnings don't fail CI for these patterns).

Related: [[stage8-real-fixes]], [[stage9-gpu-serving-fixes]]