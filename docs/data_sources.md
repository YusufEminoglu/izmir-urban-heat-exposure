# Data sources and redistribution boundary

No third-party raster or vector is redistributed in this repository. Obtain
each input from the provider, record the version and access date, and apply the
provider terms that are current for your use. The notes below were checked
against provider pages on 29 July 2026; they are provenance guidance, not legal
advice.

| Layer | Provider / period / nominal resolution | Role | Access and licence note |
|---|---|---|---|
| WUDAPT Local Climate Zones | WUDAPT; static v3; 100 m | Primary morphological classification | [WUDAPT LCZ portal](https://lcz-generator.rub.de/); consult WUDAPT data policy |
| Landsat 8/9 Collection 2 Tier 1 | USGS; JJA 2025; thermal native 100 m | Daytime summer LST and spectral indices | [Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2); public domain |
| Landsat 8 annual composites | USGS; 2014-2024 | Temporal LST context cube | Same as above |
| SRTM GL1 v003 | NASA/USGS; static; 30 m | Elevation, slope, and TPI | [USGS SRTM](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1); US government source |
| CORINE Land Cover | Copernicus; 1990-2018; 100 m | Land-cover context and built fraction | [CORINE](https://land.copernicus.eu/en/products/corine-land-cover); Copernicus data policy |
| GLC-FCS30D LULC | Zhang et al.; 1985-2022; 30 m | Long-run land-cover trajectory | [Zenodo 8239305](https://zenodo.org/records/8239305) |
| Meta tree canopy height | Meta / WRI; 2020; 1 m native | Vegetation structure and cooling potential | [WRI dataset record](https://datasets.wri.org/datasets/meta-tree-canopy-height) |
| HRSL population | Meta / CIESIN; circa 2024; ~30 m | Population exposure and demographic indicators | [HDX catalogue](https://data.humdata.org/dataset/?q=High%20Resolution%20Population%20Density%20Maps) |
| VIIRS night-time lights | Earth Observation Group; 2024 annual; ~500 m | Urban activity and development proxy | [EOG VNL products](https://eogdata.mines.edu/products/vnl/) |
| Izmir mahalle boundaries | Izmir Metropolitan Municipality; 2024 | Administrative planning units | Verify at municipality open-data portal |
| Izmir DSM/SYM | Izmir Metropolitan Municipality; 2020; 1 m native | Urban morphology and sky view factor | Same as above |

## Processing services

Google Earth Engine is used for cloud-scale compositing and export. Users
authenticate with their own account and Cloud project.

## Why the data are absent

Several inputs restrict redistribution, have non-commercial or share-alike
conditions, or are too large for a source repository. Keeping them out of Git
also avoids presenting a stale mixed-vintage snapshot as canonical.

## Citation ledger

For a rerun, record at minimum: provider and product title; version, DOI, or
asset ID; access date and original URL; temporal coverage and processing
filters; licence observed on that date; reprojection, clipping, resampling,
and nodata decisions.
