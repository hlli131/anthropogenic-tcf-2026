# Human-influenced heterogeneity in global distribution of tropical cyclone frequency trends
![Status](https://img.shields.io/badge/Status-Under_Review-yellow)
![Version](https://img.shields.io/badge/Version-2026.09.10-red)
![Language](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

> For peer review only. The final version will be updated upon acceptance.

<!-- > For any inquiries, please feel free to reach out via email: [hlli@smail.nju.edu.cn](mailto:hlli@smail.nju.edu.cn) 📧 -->


## 📖 Brief introduction
This repository includes the following directories:
- *`observed_interhemispheric_contrast`*
- *`primary_factor_identification`*
- *`detection_and_attribution_analysis`*
- *`physical_mechanism_explanation`*
- *`source_data`*

| Directory name | Description |
| ---------- | ---------- |
| *observed_interhemispheric_contrast* | Analyze and plot the heterogeneity in global TCF trends (**Fig.1**) |
| *primary_factor_identification* | Identify key factors and quantify their contributions and interactions using IML (**Fig.2**) |
| *detection_and_attribution_analysis* | Detect and attribute TCF to human fingerprints using SVD, OF, and CMIP6 simulations (**Figs.3–4**) |
| *physical_mechanism_explanation* | Explain the physical mechanism through coupled thermodynamic and dynamic pathways (**Fig.5**) |
| *source_data* | Source data for the paper (**Figs.1–4**)|


## ⚙️ Configuration (desktop)
- **Platform**: Windows Subsystem for Linux (WSL)  

- **Dependencies**:
  ```
  Python==3.11
  numpy==1.26.4
  scipy==1.14.0
  pandas==2.2.3
  xarray==2025.4.0
  netCDF4==1.7.2
  matplotlib==3.10.0
  cartopy==0.24.1
  cmaps==2.0.1  
  shap==0.47.2
  scikit-learn==1.6.1
  scikit-explain==0.1.4
  xgboost==3.0.1
  lightgbm==4.6.0
  statsmodels==0.14.4
  pymannkendall==1.4.3
  metpy==1.7.0
  tcpyPI=1.4.0
  xesmf==0.8.7 (not recommended on Windows)
  ```

- **Hardware**:
  ```
  RAM: 32 GB
  CPU: Intel(R) Core(TM) i5-14500 (14 cores, 20 threads)
  GPU: NVIDIA GeForce GTX 750 Ti (dedicated) & Intel(R) UHD Graphics 770 (integrated)
  ```


## 🚀 Installation
All required packages can be installed via `conda` (from [**Conda-forge**](https://conda-forge.org)) or `pip` (from [**PyPI**](https://pypi.org)) using the following commands:
```
# Using conda (recommended) ✅
conda install <package_name> -c conda-forge

# Using pip ✅
pip install <package_name>
```
**⏱️ Expected installation time**: Generally completes **within 5 minutes**, depending on the network speed and system configuration.


## 🧪 Reproducing the results
### 1. Observed heterogeneity in global distribution of TCF trends (Fig.1)
Processed data is provided in `observed_interhemispheric_contrast/derived_data`, so **Fig.1 can be generated directly**:

```bash
python observed_interhemispheric_contrast/02_plot_fig1.py \
    --data-dir observed_interhemispheric_contrast/derived_data \
    --output Fig1.pdf
```

To reproduce the analysis from the [**IBTrACS**](https://www.ncei.noaa.gov/products/international-best-track-archive) dataset, please **run the following scripts in sequence**:

```bash
python observed_interhemispheric_contrast/01_analyze_observed_tcf.py \
    --ibtracs ibtracs.ALL.list.v04r01.csv \
    --output-dir observed_interhemispheric_contrast/derived_data

python observed_interhemispheric_contrast/02_plot_fig1.py \
    --data-dir observed_interhemispheric_contrast/derived_data \
    --output Fig1.pdf
```

### 2. Primary environmental factors influencing global TCF (Fig.2)
Processed data is not provided here due to its large size. 
To reproduce the results, please **download the [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive), 
[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means), 
and [ORAS5](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5) datasets and run the following scripts in sequence**:

```bash
python primary_factor_identification/01_prepare_data.py \
    --input-dir input_data \
    --output primary_factor_identification/processed_data/IML_data.csv

python primary_factor_identification/02_train_models.py \
    --data primary_factor_identification/derived_data/IML_data.csv \
    --output-dir primary_factor_identification/derived_data

python primary_factor_identification/03_plot_fig2.py \
    --data-dir primary_factor_identification/derived_data \
    --output Fig2.pdf
```


### 3. Detection and attribution of anthropogenic fingerprints (Figs.3–4)


### 4. Physical mechanism driving TCF changes (Fig.5)




## 📦 Data availability
Original datasets for full analysis are publicly available from the following sources:

### TC observations
- **International Best Track Archive for Climate Stewardship (IBTrACS)**  
  Source: https://www.ncei.noaa.gov/products/international-best-track-archive  

### Atmospheric reanalysis
- **ECMWF Fifth Generation Reanalysis (ERA5)**  
  Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means  

### Oceanic datasets
- **ECMWF Ocean Reanalysis System 5 (ORAS5)**  
  Source: https://cds.climate.copernicus.eu/datasets/reanalysis-oras5  

- **Hadley Centre Sea Ice and Sea Surface Temperature (HadISST)**  
  Source: https://www.metoffice.gov.uk/hadobs/hadisst  

- **Extended Reconstructed Sea Surface Temperature version 6 (ERSSTv6)**  
  Source: https://www.ncei.noaa.gov/products/extended-reconstructed-sst  

### Multimodel simulations
- **Coupled Model Intercomparison Project Phase 6 (CMIP6)**  
  Source: https://pcmdi.llnl.gov/CMIP6  


## 📄 Licence
> This repository is open source under the **MIT License**.
