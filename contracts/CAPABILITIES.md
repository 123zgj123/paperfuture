---
id: CTR-CAPABILITY-REGISTER-001
type: contract
title: Capability register — every claim names its implementation, its test, and its live path
status: implemented
authority: normative
provenance: authored
subject: capability-claims
version: "1.0"
scope:
  - src/farfield/extras/openworld/capabilities.py
  - tests/regressions/test_capability_register.py
supersedes: []
---

# Capability register

This contract is normative. Where a README, an architecture note or a module
docstring describes a capability and this register disagrees, **the register
wins**, because the register is executable and prose is not.

## Why it exists

The recurring failure in this codebase was never broken code. It was claims that
outlived their implementation:

| Claim | What was actually true |
|---|---|
| contradiction tracking | its only writers were an event nobody emitted and a helper nobody called; twelve consumers read a list that could not fill |
| resource-aware scheduling | the budget filter compared two metric keys nothing wrote, so it refused nothing |
| experimental adequacy | the quality bar demanded a sample size no worker reported, so it could never be met |
| high-impact verification debt | it keyed on an evidence field nothing set |
| `archive_debt` | it read a metric key nothing wrote, so it was always zero |
| "the single long-term memory of the production path" | said by a module that had left the production path |

Each of these was found by a person reading code, or by a live run — never by a
test, because nothing asked a claim to demonstrate itself. A gate guarding a
condition that cannot arise reads as permanently satisfied, which is strictly
worse than no gate.

## Obligations

Every entry in `REGISTER` carries:

- **`claim`** — what the runtime asserts it can do, in a sentence.
- **`implemented_by`** — the module(s) that perform it. Checked to exist.
- **`enforced_by`** — the test(s) that fail when it stops working. Checked to exist.
- **`live_path`** — an executable probe that builds a real state and asserts the
  mechanism fires. **A capability claiming `implemented` without one is a
  failing entry**, because an unchecked claim is exactly the defect above.
- **`status`** — `implemented` / `partial` / `absent`. Anything other than
  `implemented` must state `limits`.
- **`surface`** — `production` / `experimental` / `partial` / `legacy`.
  Production requires reachable + executable + tested + a live-path probe.
  `production_manifest()` is the only list that may be advertised as a
  production capability.
- **`validated`** — `offline` or `live`. Live means a bounded run against a real
  model and a real attested world exercised it, not that a test passed.

## What this contract does not do

It does not measure research quality. A register full of firing probes says the
runtime can perform the operations it claims; it says nothing about whether the
science those operations produce is any good. Three capabilities are recorded
`partial` precisely because their completion path is unproven, and one
(`harness_patch_proposal`) is explicitly not evidence that recursive
self-improvement works.

## Running it

```bash
PYTHONPATH=src python3.11 -c "
from farfield.extras.openworld.capabilities import audit
import json; print(json.dumps(audit(), indent=2))"

PYTHONPATH=src python3.11 -m unittest tests.regressions.test_capability_register -q
```
