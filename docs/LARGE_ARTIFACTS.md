# Large artifact archives

The companion files are named `NASA_SMAP_MSL_Artifacts_01_of_05.zip` through `05_of_05.zip`. They hold restored datasets, weights and selected original run records. They are **not in this Git repository**.

Restore locally, if you already have access and permission:

```bash
python tools/restore_artifacts.py /path/to/NASA_SMAP_MSL_Artifacts_*.zip
python tools/verify_archive.py --artifacts
```

The restore script validates file sizes and SHA-256 checksums before writing. It does not execute experiments. Required object hashes are listed in `ARTIFACT_PARTS.json` and `provenance/ARTIFACT_PATHS.json`.

**Do not upload the companion archives publicly until their upstream rights are confirmed.**
