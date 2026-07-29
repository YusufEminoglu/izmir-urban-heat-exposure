# Reproducibility guide

## 1. Create the environment

Conda/Mamba is recommended because Rasterio, GDAL, GEOS, and PROJ must remain compatible.

~~~bash
mamba env create -f environment.yml
mamba activate izmir-urban-heat-exposure
python scripts/validate_metadata.py
~~~

A pip-only setup is also pinned in `requirements.txt`, but system geospatial libraries may still be required.

## 2. Configure Earth Engine without committing secrets

Authenticate with your own Google Earth Engine account. Define these settings in your shell or an untracked local environment file:

~~~text
EE_PROJECT=your-ee-cloud-project
GEE_DRIVE_FOLDER=your-export-folder
EE_ROI_LOCAL=data/01_raw/paper1/izmir_study_boundary.geojson
GEE_DOWNLOAD_DIR=data/01_raw/paper1
P1_GRID_RESOLUTION=100
~~~

If `EE_ROI_LOCAL` is not set, `src/_gee_config.py` requires `EE_ROI_ASSET`. The publication audit rejects private asset IDs, hard-coded Drive folders, user-specific absolute paths, and common token formats.

## 3. Obtain source data

Follow [data_sources.md](data_sources.md), keep a provenance ledger with download date/version, and confirm each provider's terms. Do not add provider rasters to Git.

Expected ignored stages:

~~~text
data/01_raw/paper1/
data/02_interim/paper1/
data/03_processed/paper1/
data/04_models/paper1/
outputs/
logs/
~~~

## 4. Run the pipeline

The numeric prefixes encode dependency order.

~~~bash
python src/p1_00_inventory_and_boundary.py
python src/p1_01_population_controls.py
python src/p1_02_gee_lcz_wudapt.py
python src/p1_03_dsm_morphology.py
python src/p1_04_gee_lst_summer.py
python src/p1_04b_clip_align_covariates.py
python src/p1_04c_srtm_terrain.py
python src/p1_05_spatial_alignment.py
python src/p1_06_descriptive_lcz_tests.py
python src/p1_07_baseline_ols.py
python src/p1_10_build_datacubes.py
python src/p1_09_gwr_preliminary.py
~~~

Some scripts require completed upstream outputs. Run sequentially.

## 5. Regenerate figures

Figure and table scripts are in `src/p1_12_figure_cartography.py`,
`src/p1_13_figure_panels.py`, `src/p1_14_manuscript_tables.py`,
`src/p1_15_manuscript_figures.py`, `src/p1_fig08_heat_justice.py`, and
`src/p1_fig09_mahalle_priority.py`. They expect completed processed outputs.
Final authored PNGs are included in `figures/png` for inspection without
third-party inputs.

~~~bash
python src/p1_12_figure_cartography.py
python src/p1_13_figure_panels.py
python src/p1_14_manuscript_tables.py
python src/p1_15_manuscript_figures.py
python src/p1_fig08_heat_justice.py
python src/p1_fig09_mahalle_priority.py
~~~

## 6. Audit before publishing

~~~bash
python -m compileall -q src scripts
python scripts/validate_metadata.py
python scripts/publication_audit.py
~~~

The audit also asserts that exactly nine author figures are present and that no
PDF, manuscript source, model object, raw raster, private Earth Engine
identifier, or credential-like string is publishable.
