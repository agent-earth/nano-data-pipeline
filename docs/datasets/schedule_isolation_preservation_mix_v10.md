# Schedule Isolation Preservation Mix v10

V10 isolates recurring-schedule supervision as the last numerical failure
family. It changes exactly 8 train rows from v6 and keeps the other 184 rows
byte-identical. The frozen 32-step schedule exposes 7 replacements.

It consumes only irreversible abstract feedback and verified synthetic rows.
No benchmark, canary, model-output, teacher-output, or independent-holdout row
enters training.

- samples: 192;
- train / development: 160 / 32;
- replacement / unchanged: 8 / 184;
- overlap with v1-v6 identities and signatures: 0;
- SHA256:
  `2bb712de519149d776b1c346466ee49d20017f1065aa3d1b44ae59eb6f5b973a`.

Reproduce with `build-schedule-isolation-preservation-mix`, then
`scripts/validate_release.py`.
