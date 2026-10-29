# Water Accounting Plus (WA+) — West Rapti Basin Training Pipeline

This repository contains the complete, reproducible **Water Accounting Plus (WA+)** computational pipeline, interactive Google Colab notebooks, and core Python engine (`west_rapti.py`) developed for the capacity-building training program on satellite-based river basin water accounting.

The training was conducted for Nepal's **Water and Energy Commission Secretariat (WECS)**, **Department of Water Resources and Irrigation (DoWRI)**, and the **World Bank**, focusing on the **West Rapti River Basin** in western Nepal (outlet at DHM Gauge Station Q375, Kusum).

---

## Executive Summary & Core Philosophy

Conventional water resource assessments rely on sparse point-gauge networks and canal intake logs, which miss unmonitored groundwater pumping, informal diversions, and return flows. **Water Accounting Plus (WA+)** uses wall-to-wall satellite remote sensing (precipitation, evapotranspiration, land cover, and biomass) combined with an offline root-zone soil water balance model to track water availability, consumption, and productivity across entire river basins.

### Key Conceptual Pillars
* **Consumption-Centric Accounting:** *"Account what is consumed, not just what is diverted."* Water diverted for irrigation is not lost to the basin; only consumed **Blue ET** is permanently removed, while unconsumed fractions return to stream networks and aquifers for downstream reuse.
* **Canonical Reference Grid:** All multi-source satellite datasets (ranging from 10 m to 5.5 km native resolutions) are resampled and aligned onto a single, locked **300 m canonical grid** (`gridspec.json`).
* **The Four WA+ Land Use Categories:**
  1. **Protected Land Use (PLU):** Conservation-first areas (glaciers, national parks, protected wetlands).
  2. **Utilized Land Use (ULU):** Natural ecosystems with minimal management (natural forests, shrublands, grasslands).
  3. **Modified Land Use (MLU):** Human-modified landscapes where water is unmanaged (rainfed cropland, plantations, urban settlements).
  4. **Managed Water Use (MWU):** Active water abstraction and management (irrigated cropland, reservoirs, urban water supply).

---

## Repository & Pipeline Architecture

The pipeline consists of a master setup script, six sequential practical laboratory notebooks, a complementary SAR change-detection notebook, and a shared Python core module (`west_rapti.py`).

```
west_rapti/
├── 00_Setup.ipynb                   # Master setup & dependency installer
├── Lab1_delineate_interactive.ipynb # Basin delineation & grid snapping
├── Lab2_acquire_interactive__3_.ipynb # Satellite data stack acquisition & bias correction
├── Lab3_landcover_interactive.ipynb # Land cover classification & WA+ land-use mapping
├── Lab4_hydroloop_interactive.ipynb # Hydroloop soil-moisture balance (Green/Blue ET split)
├── Lab5_sheets_interactive.ipynb   # Standardized WA+ Sheets 1–4 generation
├── Lab6_synthesis_interactive.ipynb # Utilisable flow, e-flows & scenario brief
├── Lab6b_sar_change_interactive.ipynb # Sentinel-1 SAR flood & disaster change detection
├── west_rapti.py                    # Core WA+ engine, raster utilities, and hydroloop model
├── westrapti_prep.py                # Preprocessing helper for DHM gauge & discharge data
├── selftest_west_rapti.py           # Unit self-test suite for the WA+ engine
└── outputs/                         # Analysis-ready data cubes and deliverables
    ├── gridspec.json                # Shared 300 m spatial grid specification
    ├── basin_config.json            # Central pipeline metadata & file registry
    ├── basin_mask.nc                # NetCDF 300 m basin boundary mask
    ├── dem_utm.tif                  # Copernicus DEM (UTM Zone 44N)
    ├── P_corrected.nc               # Gauge bias-corrected monthly rainfall cube
    ├── ET.nc                        # Monthly actual evapotranspiration cube
    ├── LAI.nc                       # Monthly Leaf Area Index cube
    ├── NDM.nc                       # Monthly Net Dry Matter (biomass) cube
    ├── lu.nc                        # WA+ 4-category land-use NetCDF map
    ├── etrain.nc                    # Green ET (rain-fed soil moisture consumption)
    ├── etincr.nc                    # Blue ET (incremental / irrigation consumption)
    ├── sro.nc                       # Surface runoff cube
    ├── perc.nc                      # Deep soil percolation cube
    ├── Q_routed.csv                 # Routed streamflow time series
    ├── wa_sheets_summary.json       # Summary metrics for WA+ Sheets 1–4
    └── hydroloop_params.json        # Calibrated model settings for scenario re-runs
```

---

## Detailed Laboratory Workflows

### `00_Setup.ipynb` — System Setup & Self-Test
* Mounts Google Drive (`My Drive/west_rapti`) and installs required Python packages (`pysheds`, `rioxarray`, `geemap`, `geedim`, `wapordl`, `dem-stitcher`, `earthengine-api`).
* Configures Earth Engine Cloud Project ID (`participant.json`).
* Runs `selftest_west_rapti.py` to verify DEM sink filling, D8 flow routing, and grid pinning without external data dependencies.
* Processes DHM hydromet gauge files using `westrapti_prep.py`.

### `Lab 1` — Basin Delineation & Canonical Grid Setup
* **Objective:** Delineate the catchment upstream of DHM Station Q375 (Kusum gauge, 82.0939 E, 28.0075 N).
* **Workflow:** Interactive Earth Engine outlet selection -> 90 m Copernicus GLO-30 DEM extraction -> Pit filling & D8 flow routing -> Snapping to 300 m canonical grid (EPSG:32644) -> Exporting `gridspec.json` and `basin_config.json`.

### `Lab 2` — Satellite Data Acquisition & Evaluation
* **Objective:** Retrieve monthly hydro-climatic data stacks and enforce water balance closure.
* **Data Stack:** CHIRPS daily precipitation, FAO WaPOR v3 AETI / MODIS MOD16A2GF ET, MODIS LAI (MOD15A2H), and MOD17A2HGF GPP / WaPOR NPP.
* **Bias Correction:** Calibrates CHIRPS rainfall against DHM rain gauges using elevation-dependent Inverse Distance Weighting (`elevation_idw`).

### `Lab 3` — Land Cover, Land Use Mapping & Cube Assembly
* **Objective:** Map land cover, isolate Class 8 (Irrigated Cropland), and freeze the analysis-ready data cube.
* **Methodology:** ESA WorldCover 10 m baseline combined with Random Forest classification trained on Sentinel-1/2 composites and LGRIP30 / WorldCereal irrigation masks. Resamples features to 300 m and exports `lu.nc`.

### `Lab 4` — Hydroloop Soil-Moisture Balance Model
* **Objective:** Run an offline, pixel-based 1D root-zone water balance model in Python to partition satellite ET into Green ET (`etrain`) and Blue ET (`etincr`).
* **Mechanics:** 
  Soil Bucket Capacity SM_max = Root Depth * AWC
  Calculates canopy interception (I), net rainfall (P_net), soil moisture storage (dSM), surface runoff (SRO), deep percolation (Perc), and baseflow routing (Q_routed).
* **Validation:** Model soil moisture dynamics are validated against Sentinel-1 C-band VV radar backscatter.

### `Lab 5` — Standard WA+ Accounts (Sheets 1–4)
* **Objective:** Compute standardized WA+ reporting sheets and execute cross-sheet balance checks:
  * **Sheet 1 (Resource Base):** Gross inflow, net inflow, landscape green ET, incremental blue ET, storage change, and river outflow.
  * **Sheet 2 (Evapotranspiration):** Beneficial vs. non-beneficial consumption and natural vs. managed land use.
  * **Sheet 3 (Productivity):** Agricultural Water Productivity (WP = yield / ET_consumed, in kg/m3).
  * **Sheet 4 (Withdrawals):** Gross abstractions, consumed blue ET, and recoverable return flows.

### `Lab 6` — Synthesis, Environmental Flows, Scenarios & Policy Brief
* **Objective:** Translate accounting outputs into allocable water policy using the availability cascade:
  Utilisable Flow = Available Flow - Environmental Flow - Committed Flow
* **Policy Evaluation:** Compares legal environmental flow requirements under:
  1. *Hydropower Policy 2001:* 10% of minimum monthly average flow.
  2. *Water Resources Bill 2026:* 30% year-round flow share.
* **Stress Testing:** Simulates climate scenarios (+/- 15% precipitation, +2 deg C warming) and irrigation expansion.

### `Lab 6b` — Sentinel-1 SAR Disaster & Water Change Detection
* Standalone exercise using Sentinel-1 C-band Synthetic Aperture Radar (SAR) backscatter (sigma-0 in dB) to map surface water dynamics, flood extent, and terrain changes.

---

## Key Hydrological Metrics (West Rapti Case Study)

| Hydrological Metric | Volume / Value | Source / Context |
| :--- | :--- | :--- |
| **Gross Basin Inflow (P)** | 9,348 MCM/yr | Sheet 1 (CHIRPS gauge-corrected) |
| **Net Inflow (P - closure)** | 9,332 MCM/yr | Sheet 1 |
| **Landscape Green ET (ET_green)** | 3,594 MCM/yr | Sheet 1 (Rain-fed consumption) |
| **Incremental Blue ET (ET_blue)** | 254 MCM/yr | Sheet 1 (Irrigated consumption) |
| **Storage Change (dS)** | 16 MCM/yr | Sheet 1 |
| **Modelled Basin Outflow (Q_model)** | 5,483 MCM/yr | Sheet 1 (Kusum outlet) |
| **Gauged Outlet Discharge (Q_gauged)** | 4,888 MCM/yr | DHM Station Q375 |
| **Committed Downstream Irrigation** | 196 MCM/yr | Baseline (22,342 ha) |

---

## Installation & Setup Guide

### Prerequisites
* **Google Account** with access to Google Colab and Google Drive.
* **Google Earth Engine (GEE) Account:** Registered Cloud Project ID with the Earth Engine API enabled.

### Running on Google Colab (Recommended)
1. Download or copy the `west_rapti` directory into your Google Drive at `My Drive/west_rapti`.
2. Open `00_Setup.ipynb` in Google Colab.
3. In Cell 1, enter your Earth Engine Cloud Project ID:
   ```python
   EE_PROJECT = 'your-ee-project-id'
   ```
4. Click **Runtime -> Run all**. Ensure all setup checks pass.
5. Execute the notebooks in sequential order (Lab1 -> Lab2 -> Lab3 -> Lab4 -> Lab5 -> Lab6).

### Running Locally
1. Clone this repository and navigate to the folder:
   ```bash
   cd west_rapti
   ```
2. Install dependencies:
   ```bash
   pip install earthengine-api geemap geedim pysheds rioxarray xarray rasterio scikit-learn wapordl dem-stitcher gdown pandas numpy matplotlib
   ```
3. Authenticate Earth Engine:
   ```bash
   earthengine authenticate
   ```
4. Run the engine self-test:
   ```bash
   python selftest_west_rapti.py
   ```

---

## Remote Sensing Data Sources Summary

| Product | Measured Parameter | Resolution | Purpose in WA+ |
| :--- | :--- | :--- | :--- |
| **CHIRPS Daily** | Rainfall (P) | 0.05 deg (~5.5 km) | Primary precipitation driver |
| **WaPOR v3 AETI / MOD16** | Actual Evapotranspiration (ET) | 300 m / 500 m | Total water consumption |
| **Sentinel-2 Multispectral** | Reflectance / NDVI / NDWI | 10 m – 20 m | Land cover & crop mapping |
| **Sentinel-1 GRD SAR** | C-band Radar Backscatter | 10 m | Soil moisture & water detection |
| **Copernicus DEM GLO-30** | Surface Elevation | 30 m | Basin delineation & flow routing |
| **ESA WorldCover 10m** | Baseline Land Cover | 10 m | Initial physical cover classes |
| **LGRIP30 / WorldCereal** | Irrigation Mask | 10 m – 30 m | Isolating Class 8 Irrigated Cropland |
| **MOD15A2H / MOD17A2HGF** | LAI & Biomass (GPP/NPP) | 500 m | Interception & Water Productivity |

---

## Key References & Methodology

* **WA+ Framework:** Karimi, P., Bastiaanssen, W.G.M. & Molden, D. (2013). *Water Accounting Plus (WA+) — a water accounting procedure for complex river basins based on satellite measurements.* Hydrology and Earth System Sciences, 17, 2459–2472.
* **FAO WaPOR Portal:** FAO portal to monitor water productivity through open-access remotely sensed data.
* **Nepal WECS:** Water and Energy Commission Secretariat, Government of Nepal.

---

## License & Citation

This project is shared under the **MIT License**.

When referencing or utilizing this pipeline, please cite:
> *Water Accounting Plus (WA+) West Rapti Basin Training Pipeline.* Developed for WECS, DoWRI, and the World Bank Capacity Building Program, 2026.
