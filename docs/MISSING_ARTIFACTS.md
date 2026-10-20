# Known gaps

- **EXP011:** some individual classifier-fit files are missing; aggregates are saved.
- **EXP013:** final trained-state backup (including 22 fits and associated scoring/calibration state) was not recovered. Its saved final tables and audits **are present**.
- **EXP014–015:** original reports are retained, but full checkpoint replay needs the missing EXP013 states.
- **EXP001–005:** some intermediate outputs and exact original CPU dependency pins are unavailable.
- **Privacy edits:** paths in some original logs, configs and notebooks were neutralised. Their frozen hashes still describe the private original runs; do not claim byte-identical reproduction from modified public copies.

Do not fill gaps by inventing checkpoint files or rerunning an experiment and calling it historical output.
