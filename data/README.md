# Versioned author-created data

This directory holds compact parameter tables and summary statistics that are
small enough to version alongside the code. They allow verification of the
reported results without re-running the full pipeline or obtaining third-party
geospatial inputs.

## model_params/

- `ols_coefficients.csv` — OLS coefficient estimates, standard errors, t-values,
  p-values, and VIF.
- `mgwr_bandwidths.csv` — MGWR bandwidths (nearest neighbours) and bandwidth type.
- `lcz_lst_summary.csv` — LCZ-level mean LST, standard deviation, and cell count.

## What is NOT here

Third-party geospatial data, processed rasters, model objects (`.pkl`,
`.joblib`), manuscript files, and publisher PDFs. These are git-ignored and
must be obtained separately following `docs/data_sources.md`.
