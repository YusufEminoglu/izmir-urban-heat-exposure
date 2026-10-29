# Methods at a glance

This document maps the manuscript methods to the public implementation. It is
a technical companion, not a replacement for the peer-reviewed article.

## Analysis frame

The study covers Izmir Province in western Turkiye. Continuous layers are
harmonised to a 100 m reference grid in EPSG:32635 using bilinear resampling;
categorical layers use nearest-neighbour resampling. The final analysis grid
holds 1,197,430 cells (1,197,379 with complete covariates).

## Stage I — cloud-local data cube

A Landsat 8/9 summer (JJA) composite provides daytime land surface temperature
at 100 m resolution. Spectral indices (NDVI, NDBI, SAVI), albedo, and the
UTFVI thermal comfort index are derived from the same Landsat stack. The data
cube is assembled as an xarray Dataset aligned to the 100 m analysis grid.

Relevant implementation:

- `src/p1_04_gee_lst_summer.py` — GEE export of summer LST and spectral indices.
- `src/p1_04b_clip_align_covariates.py` — clip and align externally downloaded covariates.
- `src/p1_04c_srtm_terrain.py` — SRTM elevation, slope, and TPI.
- `src/p1_05_spatial_alignment.py` — raster-to-grid alignment at 100 m.
- `src/p1_10_build_datacubes.py` — xarray NetCDF data cube assembly.
- `src/p1_11_gee_annual_lst_cube.py` — 2014-2024 annual LST cube for temporal context.

## Stage II — LCZ and morphological controls

WUDAPT Local Climate Zones are resampled to 100 m. Fourteen LCZ types are
mapped in the study area. CORINE land cover, GLC-FCS30D land-cover
trajectory, Meta tree canopy height, and DSM-derived morphology provide
complementary urban-form descriptors.

Relevant implementation:

- `src/p1_02_gee_lcz_wudapt.py` — GEE export of WUDAPT LCZ.
- `src/p1_03_dsm_morphology.py` — DSM/SYM morphological parameters.
- `src/p1_06_descriptive_lcz_tests.py` — Kruskal-Wallis and post-hoc LCZ-LST differentiation.

## Stage III — global OLS and spatial diagnostics

A physical-environmental OLS model regresses summer LST on NDVI, NDBI, albedo,
elevation, slope, tree canopy height, and LCZ-built fraction. Variance
inflation factors confirm acceptably low multicollinearity. Residual Moran's I
is computed across multiple spatial weights matrices to diagnose spatial
dependence.

Relevant implementation:

- `src/p1_07_baseline_ols.py` — OLS estimation, VIF, residual diagnostics.
- `src/p1_14_manuscript_tables.py` — manuscript-grade coefficient and diagnostic tables.

## Stage IV — multi-scale geographically weighted regression (MGWR)

MGWR assigns each term its own spatial bandwidth, allowing locally varying
relationships. The revised model has a local intercept and NDVI, NDBI and albedo
terms, an adaptive bisquare kernel and golden-section bandwidth search, fitted on
a stratified 4,000-cell sample; a repeat from global starting values reaches a
higher AICc (3931.6 against 3919.8), so the reported solution is retained.

| Term | Bandwidth (nearest neighbours) |
|---|---|
| Intercept | 20 |
| NDVI | 89 |
| NDBI | 1,071 |
| Albedo | 101 |

Relevant implementation:

- `src/p1_09_gwr_preliminary.py` — initial GWR exploration with sampled grid.
- `src/p1_12_figure_cartography.py` — coefficient surface cartography.
- `src/p1_13_figure_panels.py` — journal-grade figure panels.
- `src/p1_15_manuscript_figures.py` — final manuscript figure generation.

## Stage V — mahalle heat-justice translation

Population vulnerability indicators (elderly ratio, child ratio, population
density from HRSL) are aggregated to mahalle administrative units. Bivariate
Moran testing evaluates the spatial overlap between daytime heat exposure and
vulnerability. An MGWR-informed mahalle priority surface identifies actionable
intervention zones.

Relevant implementation:

- `src/p1_01_population_controls.py` — HRSL population and demographic indicators.
- `src/p1_fig08_heat_justice.py` — heat-justice sociodemographic synthesis (Figure 8).
- `src/p1_fig09_mahalle_priority.py` — mahalle priority panels (Figure 9).

## Validation and robustness

- LCZ thermal differentiation: Kruskal-Wallis H = 446,302.8, p < 0.001, eta-squared = 0.373; the ordering holds in all thirteen summers 2014-2026 (Kendall's W = 0.970).
- OLS: adjusted R-squared = 0.782, maximum VIF = 6.96, residual Moran's I = 0.724.
- MGWR: R-squared = 0.900 (sample), residual Moran's I = -0.079; grid hold-out R-squared = 0.839; 10 km block cross-validation 0.737 (OLS with the same predictors 0.616, full global OLS 0.791).
- Spatial error and spatial lag models were also estimated as supplementary diagnostics.

## Determinism

Random seeds are fixed (seed 42) for all stochastic components including MGWR
initialisation. Geospatial results still depend on the exact provider vintages
and the GDAL/PROJ stack, so the environment files are pinned and provenance
should be recorded for each rerun.
