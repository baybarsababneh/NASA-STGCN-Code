# Environments

There is no single `requirements.txt`: experiments ran in different environments.

- **EXP001–005:** early CPU PyTorch runs; original dependency pins incomplete.
- **EXP006–010:** archived GPU/CPU notebooks, original requirements and freeze files.
- **EXP011:** original CPU requirements are inside its package.
- **EXP012–015:** separate preparation, scoring, execution and evaluation environments; use each package's recorded pins and gates.

Do **not** merge all packages into one environment. See each experiment's `README.md`, `RUN_FREEZE.json` and package requirements.
