# Notebook timing evidence

The historical runtime audit is preserved unchanged. In this research archive, runnable notebooks have cleared outputs and session metadata. Their original existing stdout/error/text displays and whitelisted execution timestamps are retained in `execution_records/` beside those notebooks (or inside EXP012's preparation/execution directory). `provenance/SANITISATION.json` connects original notebook hashes, cleaned hashes, and extracted records.

These timing fields are transcribed metadata, not new elapsed measurements. A cell can include setup, replay, resumed work, or several stages. The runtime audit's fit/stage/whole-run distinctions still apply. Rich binary/HTML displays and session identifiers are omitted; source code and frozen scientific numerical exports remain available. No notebook cells were executed to prepare the transcripts.
