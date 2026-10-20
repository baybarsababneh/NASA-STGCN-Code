# Third-party material

| Component | Terms and scope |
|---|---|
| Telemanom, California Institute of Technology, 2018 | The bundled `LICENSE.txt` grants redistribution under three conditions: retain the notice, conditions, and disclaimer; reproduce them with binary distributions; and do not imply endorsement. Its text has BSD-style terms, although the upstream README says Apache 2.0. Preserve the actual bundled notice; do not relicense these copies as Apache. |
| GDN, d-ailin, 2021 | MIT. The original notice remains in `experiments/EXP014/package/vendor/gdn/LICENSE` and `references/third_party_licenses/pinned_gdn_LICENSE`. The two declared PyG compatibility changes are recorded in EXP013 source and EXP014 vendor provenance. |
| CATCH | Referenced by pinned repository commit; its implementation archive is not in this Git ZIP. No redistribution licence was established from the available pinned-source record or current upstream listing. Keep that archive private unless permission is established. |

The Telemanom notice also accompanies adapted detector code, including `followups/released_lstm/sources/native_errors_integer_compat.py`, historical vendored code, project Hundman-detector adaptations, and the small label CSV copies. Its complete text is at [`references/third_party_licenses/telemanom_LICENSE.txt`](references/third_party_licenses/telemanom_LICENSE.txt). The integer-index compatibility change is recorded in the follow-up source and execution records. This attribution does not grant separate rights to raw telemetry or released model weights.

Raw NASA telemetry, pretrained HDF5 models, and project checkpoint/array payloads are external. No dataset-specific redistribution grant for the full supplied archive was verified. NASA media-use guidance is not a substitute for dataset and contractor rights. Do not publish the companion archives on the assumption that the NASA name makes every file public domain. See [`docs/DATA.md`](docs/DATA.md).

No new blanket licence is assigned to original project code. The rights holder must choose reuse terms before describing this as an open-source release. Public visibility alone does not provide an open-source licence. Existing third-party terms remain in force. Package dependencies are installed separately, not redistributed as part of this ZIP.

Sources checked: [Telemanom licence](https://github.com/khundman/telemanom/blob/master/LICENSE.txt), [GDN licence](https://github.com/d-ailin/GDN/blob/main/LICENSE), [CATCH](https://github.com/decisionintelligence/CATCH), and [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
