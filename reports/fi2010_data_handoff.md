# FI-2010 data handoff

## Verified source

Place the official archive at `%USERPROFILE%\Downloads\FI-2010-official.zip`.

- Required size: `1,830,875,986` bytes
- Required SHA-256: `bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6`
- Required inner member: `published/BenchmarkDatasets/BenchmarkDatasets.zip`
- Dataset ID: `73eb48d7-4dbc-4a10-a52a-da745b47a649`
- PID: `urn:nbn:fi:csc-kata20170601153214969115`
- Licence: CC BY 4.0

The import command hashes every outer byte before extraction, validates paths/types/counts/sizes and
compression ratios, extracts only the required inner archive to a `.partial` path, verifies the byte
count, flushes it and atomically promotes it. Existing unmanifested, partial or mismatched files fail
closed.

## Commands

```powershell
Set-Location 'C:\Users\hp\Desktop\lob-alpha-research'
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-import --verify-only
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-import
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-audit
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-develop
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-freeze
.\.venv\Scripts\python.exe -m lob_alpha.cli fi2010-report
```

The normal audit opens and validates CF_1-CF_8 training/test payloads plus Train_CF_9. It records only
central-directory metadata for Test_CF_9 and writes that identity to a separate holdout manifest.
Development and freeze load only the development manifest. Do not invoke `fi2010-holdout` during
development.

## Local outputs and recovery

- `data/raw/fi2010/source_manifest.json`: exact imported-source binding.
- `data/raw/fi2010/development_manifest.json`: CF_1-CF_8 pairs plus Train_CF_9.
- `data/raw/fi2010/holdout_manifest.json`: Test_CF_9 central metadata only.
- `artifacts/fi2010/development/`: fold metrics and frozen selection inputs.
- `artifacts/fi2010/freeze/`: candidate JSON, final fitted model and its manifest.

All are gitignored. If import crashes before promotion, remove only the reviewed
`BenchmarkDatasets.zip.partial` and rerun. If a completed inner archive exists without its source
manifest, do not adopt it automatically: remove the reviewed local import and repeat exact-source
verification. Development artifacts are immutable; after a non-holdout failure, use a new empty
artifact directory. Once either holdout claim/seal exists, never rerun or delete it—the one-shot study
is irreversibly consumed and an incomplete result must be reported as such.

Resource preflight records free disk and available physical memory. Per-fold evidence reports exact
NumPy input-array bytes, explicitly not operating-system peak RAM. No raw member, parsed matrix, model
binary or large generated table belongs in version control.
