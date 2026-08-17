# Manual Optiver data handoff

The Optiver - Trading at the Close competition data is licensed input. Do not add the ZIP,
`train.csv`, prepared Parquet, fitted models, or row-level outputs to Git.

1. Create a free account at Kaggle in a browser.
2. Open the **Optiver - Trading at the Close** competition page and accept its rules and data
   licence.
3. Download the competition ZIP manually. Kaggle API credentials are not required.
4. Leave the ZIP in `Downloads`, place it in the repository root, or keep it elsewhere and pass
   its exact path.
5. From the repository root, run:

   ```powershell
   .\scripts\import_optiver_zip.ps1 -ZipPath "C:\path\to\optiver-trading-at-the-close.zip"
   ```

   If exactly one matching ZIP exists in the repository root or `Downloads`, `-ZipPath` may be
   omitted. The extractor rejects absolute/drive/UNC paths, forward- and backslash traversal,
   links, Windows-reserved or ADS names, case collisions, ambiguous `train.csv` members, excessive
   member counts, uncompressed sizes and compression ratios, interrupted outputs and an existing
   destination. It extracts only
   `data/raw/optiver/train.csv`.
6. Before modelling, run the target-blind metadata audit and full preparation workflow:

   ```powershell
   .\scripts\audit_prepare_equity.ps1
   ```

The first audit reads schema and identifiers but excludes `target`; it confirms that the observed
date range matches the frozen 0-480 registration. The second audit checks that every present target
is numeric and finite, records rare missing-target coverage without dropping those rows, and
computes no performance. Preparation then writes ignored, hash-addressed per-date Parquet files.
