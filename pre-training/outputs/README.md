# outputs folder

This folder holds everything produced by the pipeline: one subfolder per
conversion run, named `[timestamp]_[dataset]/`, containing that run's PNG
pages plus the `-OCR.csv` and `-SUMMARIES.csv` files generated alongside them.

## Source control policy

- Nothing generated in this folder is committed to git — PNGs, CSVs, and
  per-run subfolders are all git-ignored.
- This README is the one exception, kept as documentation.
