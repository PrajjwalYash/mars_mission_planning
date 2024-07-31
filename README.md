# Mars Mission Planning

## Overview
This repository contains tools and models for planning Martian missions, focusing on optimizing power management, dust impact, and site selection. The project integrates Radiative Transfer Models (RTM) with machine learning to address key challenges in Martian exploration.

## Objectives

### Power Management Planning
- Plan Martian missions to meet the power requirements of all subsystems.
- Implement a Radiative Transfer Model (RTM) to account for high variability in solar irradiance at different landing sites.
- Ingest data from the Mars Climate Database (MCD) and the best available literature.

### Dust Deposition Impact
- Study the impact of dust deposition on solar array generation.
- Use daily mean dust deposition rate data from MCD.
- Apply a linear degradation model using the average degradation rate of 0.2% per sol for InSight lander as the baseline.

### Optimal Site Selection
- Determine the optimal site for each launch window to maximize payload operation duration.
- Combine RTM outputs (direct, diffuse, and effective solar irradiance) with dust deposition reduction factors.
- Estimate the operational hours for each site and launch window combination to identify the best site from a power management perspective.

### Dust Deposition Prediction
- Use machine learning to predict the rate of dust deposition based on RTM outputs.
- Incorporate features such as top-of-atmosphere irradiance, direct component, and diffuse component.
- Address underprediction issues, especially at higher dust deposition rates.

## Models and Data
- **Radiative Transfer Model (RTM)**: Based on COMIMART for calculating solar radiation fluxes on the Martian surface.
- **Data Sources**: Mars Climate Database (MCD) and various relevant literature.
- **Machine Learning Models**: Ensembled tree-based models to predict dust deposition rates.

## Repository Structure

```plaintext
mars_mission_planning/
├── data/
│   ├── site_tau.csv                # Optical depth data from MCD
│   ├── site_dd_mean.csv            # Daily mean dust deposition rate data from MCD
│
├── notebooks/
│   ├── RTM_implementation.py        # RTM-based analysis for objectives 1, 2, and 3
│   ├── mars_environment.py          # Ingest optical depth data for Martian atmospheric environment
│   ├── complete_year_irr.py         # Total, direct, and diffuse irradiance for the entire Martian year
│   ├── complete_year_analysis.py    # Visualization and datasets for a given site
│   ├── dust_deposition_analysis.py  # Dust deposition reduction factor (DD_fac)
│   ├── load_support.py              # Hours of payload support based on solar array size
│   ├── ML_for_dust_deposition_rate_prediction.py # Predict dust deposition rate
│   ├── ML_data_creation.py          # Generate data for ML model from RTM outputs
│   ├── ensembled_tree_models.py     # Fit and tune tree-based regression models
│   ├── svm.py     # Fit and tune SVM and kNN regression models
│   ├── model_evaluation_and_plotting.py # Compare model performance (R2 score, MAE)
│
├── outputs/                         # Generated outputs
│
└── README.md                        # Project overview and instructions
