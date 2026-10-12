---
id: CTR-INDEX-001
type: index
title: Contracts index
status: implemented
authority: normative
provenance: authored
---

# Contracts

A contract here is normative and executable: it states a rule, names the modules
that keep it, and names the test that fails when they stop. Prose elsewhere in
the repository is descriptive. **Where a contract and a description disagree,
the contract wins.**

| Contract | Subject |
|---|---|
| [CAPABILITIES.md](CAPABILITIES.md) | Every capability claim, its implementation, its enforcing test, and an executable probe that it still fires |

The shape — YAML frontmatter with `authority`, `status`, and a `scope` naming
the contract's own enforcing test — is taken from the contract package in
apexin-ai/agent-ml-research. The addition here is that a capability's `live_path`
is executable, so a claim demonstrates itself against a constructed state rather
than pointing at a file.
