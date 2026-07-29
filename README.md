<p align="center">
  <img src="docs/assets/banner.svg" alt="Izmir Urban Heat Exposure" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-54c5a8?style=flat-square"></a>
  <a href="LICENSE-docs"><img alt="Figures and docs: CC BY 4.0" src="https://img.shields.io/badge/figures%20%26%20docs-CC%20BY%204.0-efb366?style=flat-square"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-78dce8?style=flat-square">
  <img alt="Resolution 100 metres" src="https://img.shields.io/badge/analysis%20grid-100%20m-d4ef9c?style=flat-square">
  <img alt="Publication audit" src="https://img.shields.io/badge/publication%20audit-pass-54c5a8?style=flat-square">
</p>

# Mapping urban heat exposure across local climate zones for heat-vulnerability planning

This is the open research companion to a study of how urban form, vegetation, and social vulnerability structure daytime summer heat exposure across Izmir, Turkiye. It connects a cloud-local Landsat data cube to Local Climate Zone (LCZ) morphology, tests global and multi-scale spatially varying regression models, and translates the evidence into neighbourhood-level heat-justice planning priorities.

> The manuscript is under review at *Cities*. No article DOI has been assigned. This repository contains code, author-created figures, and compact parameter tables — not the manuscript, publisher PDF, or third-party geospatial data.

<table>
  <tr>
    <td align="center"><strong>15</strong><br><sub>LCZ classes represented</sub></td>
    <td align="center"><strong>&eta;&sup2; = 0.349</strong><br><sub>LCZ thermal differentiation</sub></td>
    <td align="center"><strong>R&sup2; = 0.880</strong><br><sub>MGWR sample fit</sub></td>
    <td align="center"><strong>I = &minus;0.091</strong><br><sub>MGWR residual autocorrelation</sub></td>
    <td align="center"><strong>1,187,717</strong><br><sub>valid 100 m grid cells</sub></td>
  </tr>
</table>

<p align="center">
  <img src="figures/png/fig09_mahalle_heat_justice_priority.png" alt="Mahalle heat-justice priority surface" width="94%">
</p>

## Why this repository matters

The scientific contribution is not another single-city heat map. It is the explicit coupling of LCZ-defined urban morphology, a cloud-local data cube that leaves no proprietary dependency, multi-scale spatially varying regression that isolates *where* each heat driver matters most, and an administratively legible mahalle-level priority surface that connects remote-sensing evidence to planning decisions.

- Fifteen WUDAPT Local Climate Zones are populated across the Izmir metropolitan area, and LCZ class alone explains 34.9% of summer daytime LST variance.
- A physical-environmental OLS model captures 69.9% of LST variance, but the residual Moran's *I* of 0.811 confirms that global models mask substantial spatial structure.
- Multi-scale MGWR assigns each predictor its own spatial bandwidth and raises the sample fit to *R*&sup2; = 0.880 while driving residual spatial autocorrelation to near zero (*I* = &minus;0.091).
- Local condition numbers remain below 30, confirming a well-behaved local model.
- NDVI, NDBI, and albedo coefficients vary substantially across space and scale, revealing zone-specific cooling potentials: canopy in low-rise residential, albedo in industrial.
- Mahalle-level bivariate Moran testing shows that daytime heat&ndash;vulnerability overlap is spatially mixed rather than uniformly positive, and an MGWR-informed mahalle priority surface identifies actionable intervention zones.

## The analytical chain

~~~mermaid
flowchart LR
    A["Landsat 8/9 summer LST<br>+ spectral indices + albedo"] --> B["100 m cloud-local<br>data cube"]
    C["WUDAPT LCZ"] --> D["LCZ thermal<br>differentiation"]
    E["SRTM terrain<br>+ Meta TCH canopy"] --> F["OLS baseline<br>+ VIF diagnostics"]
    B --> F
    D --> G["Residual Moran<br>spatial diagnostics"]
    F --> G
    G --> H{"MGWR<br>multi-scale"}
    H --> I["Coefficient atlas<br>bandwidth heterogeneity"]
    J["Population vulnerability<br>mahalle administrative"] --> K["Bivariate Moran<br>heat-vulnerability overlap"]
    I --> L["Mahalle heat-justice<br>priority surface"]
    K --> L
~~~

Method details are documented in [docs/methods.md](docs/methods.md).

## Reproduce the workflow

The fastest local setup uses Conda/Mamba:

~~~bash
git clone https://github.com/YusufEminoglu/izmir-urban-heat-exposure.git
cd izmir-urban-heat-exposure
mamba env create -f environment.yml
mamba activate izmir-urban-heat-exposure
python scripts/validate_metadata.py
python scripts/publication_audit.py
~~~

For Earth Engine exports, copy `.env.example` to your own untracked environment configuration and set `EE_PROJECT` and `GEE_DRIVE_FOLDER`. Then follow the staged recipe in [docs/reproducibility.md](docs/reproducibility.md).

Third-party rasters are deliberately absent. [docs/data_sources.md](docs/data_sources.md) links each provider and records the redistribution boundary.

## Repository map

~~~text
izmir-urban-heat-exposure/
├── src/                    analysis, GEE export, figure, and table code
├── data/model_params/      versioned author-created model parameters
├── figures/png/            nine final author-generated figures
├── scripts/                metadata validation and pre-publication audit
├── docs/                   methods, provenance, reproducibility, project site
├── CITATION.cff            software and preferred article citation metadata
├── .zenodo.json            archive-ready author and ORCID metadata
├── environment.yml         geospatial Conda environment
└── requirements.txt        pinned Python environment
~~~

## Visual atlas

| LCZ thermal differentiation | Bivariate heat&ndash;vegetation atlas |
|---|---|
| [![LCZ thermal](figures/png/fig04_lcz_thermal_differentiation.png)](figures/png/fig04_lcz_thermal_differentiation.png) | [![Heat vegetation](figures/png/fig05_bivariate_hex_heat_vegetation.png)](figures/png/fig05_bivariate_hex_heat_vegetation.png) |

| Baseline model diagnostics | MGWR coefficient atlas |
|---|---|
| [![Baseline diagnostics](figures/png/fig06_baseline_model_diagnostics.png)](figures/png/fig06_baseline_model_diagnostics.png) | [![MGWR atlas](figures/png/fig07_mgwr_coefficient_atlas.png)](figures/png/fig07_mgwr_coefficient_atlas.png) |

| Heat-justice synthesis | Mahalle priority surface |
|---|---|
| [![Heat justice](figures/png/fig08_heat_justice_sociodemographic_synthesis.png)](figures/png/fig08_heat_justice_sociodemographic_synthesis.png) | [![Priority](figures/png/fig09_mahalle_heat_justice_priority.png)](figures/png/fig09_mahalle_heat_justice_priority.png) |

## Reproducibility boundary

This repository makes the analysis logic inspectable, but it cannot legally or responsibly mirror every provider raster. Reproduction therefore has two levels:

1. Code-level reproduction: environment, workflow order, parameters, and authored figures are complete here.
2. Data-level rerun: users obtain the listed datasets under their original terms, authenticate with their own Earth Engine account, and populate ignored staging directories.

The model parameter tables and coefficient summaries released here allow verification of the reported statistics without re-running the full pipeline.

## Authors

| Author | Affiliation | Profiles |
|---|---|---|
| **Yusuf Eminoglu** · corresponding author | Department of City and Regional Planning, Dokuz Eylul University | [![GitHub](https://img.shields.io/badge/GitHub-YusufEminoglu-181717?logo=github)](https://github.com/YusufEminoglu) [![ORCID](https://img.shields.io/badge/ORCID-0009--0005--6000--2934-a6ce39?logo=orcid)](https://orcid.org/0009-0005-6000-2934) [![ResearchGate](https://img.shields.io/badge/ResearchGate-profile-00ccbb?logo=researchgate)](https://www.researchgate.net/profile/Yusuf-Eminoglu?ev=hdr_xprf) [![AVESIS](https://img.shields.io/badge/AVESIS-DEU-004b8d)](https://avesis.deu.edu.tr/yusuf.eminoglu) |
| **Kemal Mert Cubukcu** | Department of City and Regional Planning, Dokuz Eylul University | [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--3604--7014-a6ce39?logo=orcid)](https://orcid.org/0000-0003-3604-7014) [![ResearchGate](https://img.shields.io/badge/ResearchGate-profile-00ccbb?logo=researchgate)](https://www.researchgate.net/profile/K-Mert-Cubukcu) [![AVESIS](https://img.shields.io/badge/AVESIS-DEU-004b8d)](https://avesis.deu.edu.tr/mert.cubukcu) |

## Citation

Use the repository's "Cite this repository" control, powered by [CITATION.cff](CITATION.cff). Until the article is accepted, please treat the preferred article metadata as "under review" and do not invent a DOI.

## Licence and attribution

Code is MIT licensed. Documentation and author-generated figures are CC BY 4.0. Input datasets retain their providers' terms and are not redistributed. See [NOTICE.md](NOTICE.md) and [docs/data_sources.md](docs/data_sources.md).

Created by Yusuf Eminoglu and Kemal Mert Cubukcu at Dokuz Eylul University, Department of City and Regional Planning, Izmir, Turkiye.
