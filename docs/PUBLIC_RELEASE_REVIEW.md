# Public-release review

**Verdict: reviewed archival candidate; keep the existing GitHub repository private until its history is checked.** This package preserves the available study; it is not a claim of complete end-to-end reproducibility.

## Remaining problems

| Issue | Consequence / next action |
|---|---|
| Private Git history inaccessible | The upload has no `.git` history or project remote. The GitHub connection is not installed. Old commits, refs, releases, Actions logs, and attachments remain unaudited. Prefer a separate repository from this clean tree; see [GITHUB_UPDATE.md](GITHUB_UPDATE.md). |
| Original seals differ from public copies | This review found 23 existing local file-hash conflicts before editing. There are now 30 directly resolvable conflicts listed in `provenance/HISTORICAL_HASH_CONFLICTS.csv`. Other original seals refer to excluded or external members. Exact scientific replay cannot be certified from this tree. Original seals are unchanged; no gates were disabled. |
| Missing/external scientific inputs | Checkpoints and residual arrays are not in this ZIP. Five private companion archives are required for many studies. EXP011 individual fit files and the full EXP013 trained-state/calibration backup remain missing; EXP014–EXP015 exact replay depends on the latter. See [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md). |
| Historical environment limits | Early CPU pins are incomplete. Later studies require their recorded, separate CPU/GPU stacks. EXP009 E01 is a resume procedure; its original fresh-start serializer remains defective. No scientific job or GPU validation was run during this release review. |
| Distribution and reuse terms | Caltech and GDN notices are retained. Full NASA data/weight redistribution and pinned CATCH archive permissions are unverified. They are external, not included here. Small original label CSVs remain with Telemanom attribution and its notice; no separate dataset-wide grant is asserted. Original project code has no blanket reuse licence. Choose one only with the rights holder's authority; public visibility alone does not license reuse. |

## Necessary corrections

- Removed remaining private references, the unused manuscript file listing, and one byte-identical historical notebook copy. Detailed scientific protocols and hash-bound duplicate sources are retained because their capsule provenance needs them.
- Corrected the unsupported EXP012 staging example to EXP006. Replaced obsolete follow-up delivery/report instructions with short working-directory, dependency, and input guidance.
- Added full Caltech notice coverage for detector adaptations. Documented the upstream README/licence discrepancy rather than assuming Apache terms.
- Made the public verifier reject unexpected files, unsafe manifest paths, and symlinks. Historical scientific gates remain intact.
- Updated only the public packaging manifest. Prior publication edits remain recorded separately in `provenance/PUBLICATION_EDITS.csv`; this review's changes are in `provenance/PUBLIC_REVIEW_CHANGES.csv`.

## Evidence and limits

All **2,616 uploaded files** were read and hashed. The file-by-file ledger records text scans, structured-format parsing, source inventories, or binary inspection. Checks covered 427 Python files (224 distinct byte versions), 14 notebooks, 211 CSV files, JSON/JSONL/YAML files, logs, scripts, patches, licences, five PDFs, 16 PNGs, and the SVG. Python source was parsed statically; this was not execution testing of every scientific path. PDF text/metadata and PNG decoding/metadata were checked; the PNG set was visually inspected. No copied paper PDF was present. Existing PDFs are project result figures.

No prohibited personal references or credential-pattern matches remain in the final file contents or names. Upstream public attribution/contact metadata was retained. Pattern scans cannot prove the absence of every possible secret. Local Markdown file links resolve. Historical absolute paths, original ZIP inputs, and missing files are documented rather than silently replaced.

All original numerical CSV, JSON/JSONL (apart from the public packaging manifest), YAML configuration, TEX result rows, and figure bytes are unchanged. All scientific Python computations are unchanged: four source files have only documented prose/docstring or test-name edits, verified by normalised AST comparison. The archive-tool verifier is the only changed operational Python logic. No experiments, model inference, metric calculation, or result regeneration were performed.

Fourteen isolated tests of the repository tools passed using temporary non-scientific fixtures: exact verification, changed/extra/symlink rejection, missing external input, restore correctness/idempotency, conflict/corruption/traversal rejection, staging refusal, and unsafe manifest rejection. These checks do not validate missing checkpoints or current Colab compatibility.

The final ZIP has a new public SHA-256 manifest covering its delivered files except the manifest itself. It is a publication-byte check, separate from frozen scientific source seals. `PUBLIC_REVIEW_FILE_AUDIT.csv` inventories the uploaded members; newly added review files are covered by the final public manifest. No GitHub changes were made.

## Sources

- [Telemanom actual licence](https://github.com/khundman/telemanom/blob/master/LICENSE.txt): Caltech copyright, redistribution conditions, and disclaimer. Its [README](https://github.com/khundman/telemanom) separately says Apache 2.0 and describes dataset acquisition; that discrepancy is retained and explained.
- [GDN licence](https://github.com/d-ailin/GDN/blob/main/LICENSE): MIT; agrees with the pinned copy already supplied.
- [CATCH official repository](https://github.com/decisionintelligence/CATCH): no licence was established from the inspected listing; the pinned archive was not supplied in this review ZIP.
- [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/): do not establish rights for every dataset, contractor contribution, or model in this external benchmark archive.
- [GitHub licensing](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) and [visibility](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility): public visibility differs from an open-source licence; Actions history and logs become public when repository visibility changes.
