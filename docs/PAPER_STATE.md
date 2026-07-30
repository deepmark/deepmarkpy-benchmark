# Paper-reproducing repository state (P0.2 record)

Status: **resolved 2026-07-30 — owner ruling: the `v1.0.0`/`v1.1.0` releases
were made in error by another developer; their tags have been purged from the
repository and versioning restarts fresh with the packaging release.** They
are not part of this effort's lineage (neither is an ancestor of `main`'s
current line), and no tag reproduces the paper's evaluated state. The
best-evidence candidate for the evaluated state remains `c66a786` (below);
tagging it is optional and stays with the owner.

REORG_PLAN.md §2.5/§6 (P0.2) requires confirming which release tag reproduces
the state evaluated in the paper (IEEE Access, vol. 14, 2026, pp. 62031-62044,
DOI 10.1109/ACCESS.2026.3685903), with the expectation that it is `v1.0.0`.

## Finding: neither existing tag reproduces the evaluated state

Paper timeline (from the published front matter): **received 28 March 2026,
accepted 15 April 2026, published 20 April 2026.**

- `v1.0.0` ("Release deepmarkpy v1.0.0", `3dc9b5b`, 2026-07-09) postdates
  publication by almost three months. Its tree is the v1.x packaging attempt
  (`src/deepmarkpy` layout, `deepmark-benchmark` console script, canonicalized
  attack-parameter names with namespaced CLI overrides) and contains the
  Codec2 vocoder attack (merged 2026-07-03) — none of which existed in the
  evaluated state. Neither `v1.0.0` nor `v1.1.0` is an ancestor of the current
  `main` lineage (merge-base with it: `b1496e4`).
- `v1.1.0` (`bf1d9e2`, 2026-07-17) additionally replaced the HTTP-proxy
  execution of six AI attacks with in-process implementations — further from
  the evaluated client-server architecture.

## Candidate commits for the evaluated state

Attack/model directory counts on the `main` lineage put the paper window
(43 attack dirs / 6 models; the paper's "40 attacks" counts differently than
raw directories) between:

| Candidate | Date | Rationale |
|-----------|------|-----------|
| `c66a786` (merge of PR #10, "new_attacks2") | 2026-03-25 | Tree at submission (received 28 March). **Recommended**: evaluations for the submitted manuscript necessarily ran on or before this state. |
| `7332a35` (merge of PR #14) | 2026-04-07 | Tip during the review window (accepted 15 April). Substantive changes vs `c66a786` (benchmark.py, base classes, many attacks) — applies only if evaluations were re-run during review. |

## Disposition

Owner ruling (2026-07-30): ignore the v1.x releases. Recommended evaluated
commit: `c66a786` (unless evaluation logs/records show re-runs after
25 March 2026). No tag has been created by this effort; creating one remains
optional and with the owner.
