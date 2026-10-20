# Reproduction

Use Python 3.11 or later for the three repository tools. Scientific packages have separate historical environments; see [ENVIRONMENTS.md](ENVIRONMENTS.md).

1. Run `python tools/verify_archive.py` from the repository root.
2. Obtain the exact dataset and the five privately held companion archives, subject to their redistribution terms. This ZIP contains no trained checkpoints or residual-array payloads.
3. Restore with `python tools/restore_artifacts.py /path/to/NASA_SMAP_MSL_Artifacts_*.zip`, then run `python tools/verify_archive.py --artifacts`.
4. Work in a separate empty workspace. Never write into the retained results.

| Study | Entry point and prerequisites |
|---|---|
| EXP001–EXP005 | Stage with `python tools/stage_workspace.py EXP001 --destination /absolute/path/to/empty-workspace`. Inside the staged `Code/`, use its numbered `src/v2_run_exp###*.py` script. TRAIN data are required; exact original CPU pins are incomplete. |
| EXP006–EXP008, EXP010–EXP011 | The helper stages the `package/` for these identifiers. Read the package protocol and numbered notebook before execution. EXP011 additionally needs the original EXP010 results ZIP. |
| EXP009 | Use the E01 resume notebook and recorded committed units. The preserved fresh-start serializer is defective; E01 is a recorded-state recovery, not a fresh-start fix. |
| EXP012 | Use separate preparation and evaluation notebooks. Preserve the 40 GB compatibility gates where applicable. The staging helper does not assemble this study. |
| EXP013–EXP015 | Use each numbered notebook and package protocol. Missing EXP013 trained states block exact final replay and the dependent comparisons. |
| Released-LSTM follow-ups | `python tools/stage_workspace.py followups --destination /absolute/path/to/empty-workspace`; then follow the staged README files. |

Historical notebooks expect original input ZIPs with exact hashes. They are retained execution recipes, not self-contained uploads of this public repository ZIP. Some inputs can be restored from the companion archives; others are absent.

Publication edits predate this review. Their effects include mismatched internal seals and hard-coded historical paths. See [PUBLIC_RELEASE_REVIEW.md](PUBLIC_RELEASE_REVIEW.md) and `provenance/HISTORICAL_HASH_CONFLICTS.csv`. The public manifest verifies publication bytes only. Do not bypass a scientific gate, overwrite an original seal, or label a new run as historical output.

Use only trusted, hash-checked historical inputs. Legacy upstream viewers use `eval` on CSV fields, some early checkpoints use pickle-backed loading, and some old extraction code assumes trusted ZIPs. These are preserved sources, not services for untrusted uploads.
