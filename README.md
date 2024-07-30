Mars Mission Planning
Overview
This repository contains tools and models for planning Martian missions, focusing on optimizing power management, dust impact, and site selection. The project integrates Radiative Transfer Models (RTM) with machine learning to address key challenges in Martian exploration.

**Objectives**
_Power Management Planning:_
Plan Martian missions to meet the power requirements of all subsystems.
Implement a Radiative Transfer Model (RTM) to account for high variability in solar irradiance at different landing sites.
Ingest data from the Mars Climate Database (MCD) and the best available literature.

_Dust Deposition Impact:_
Study the impact of dust deposition on solar array generation.
Use daily mean dust deposition rate data from MCD.
Apply a linear degradation model using the average degradation rate of 0.2% per sol for InSight lander as the baseline.

_Optimal Site Selection:_
Determine the optimal site for each launch window to maximize payload operation duration.
Combine RTM outputs (direct, diffuse, and effective solar irradiance) with dust deposition reduction factors.
Estimate the operational hours for each site and launch window combination to identify the best site from a power management perspective.

_Dust Deposition Prediction:_
Use machine learning to predict the rate of dust deposition based on RTM outputs.
Incorporate features such as top-of-atmosphere irradiance, direct component, and diffuse component.
Address underprediction issues, especially at higher dust deposition rates.

_Models and Data_
Radiative Transfer Model (RTM): Based on COMIMART for calculating solar radiation fluxes on the Martian surface.
Datasource: MCD and various relevant literature
Machine Learning Models: Ensembled tree-based models to predict dust deposition rates.

_Repo structure:_


mars_mission_planning/
│
├── data/
│   ├── site_tau.csv/                # optical depth data from MCD
│   ├── site_dd_mean.csv/          # daily mean dust deposition rate data from MCD
│
├── notebooks/
│   ├── **RTM_implementation.py**   # Python code that runs the RTM-based analysis to meet objectives 1,2 and 3.
│   ├── mars_environment.py # Code to ingest optical depth data for a given site to characterize Martian atmospheric environment.
│   ├── complete_year_irr.py # Code to ingest Mars atmospheric environment for a given site to provide total, direct and diffused irradiance for the entire Martian year.
│   ├── complete_year_analysis.py # Code to generate visualization and datasets for a given site.
│   ├── dust_deposition_analysis.py # Code to ingest daily mean dust deposition rate data for a given site to obtain dust deposition reduction factor (DD_fac).
│   ├── load_support.py # Code to ingest DD_fac, total, direct and diffuse component for a given site to provide hours of payload support that can be provided with a solar array of specified size.
│   ├── **ML_for_dust_deposition_rate_prediction.py** # Code to predict the dust deposition rate.
│   ├── ML_data_creation.py # Code to ingest outputs from RTM for a given site to generate data that can be used by a ML model.
│   ├── ensembled_tree_models.py # Code that fit different tree-based regression models and returns the optimal hyperparameter tuned model
│   ├── model_evaluation_and_plotting.py # Code to compare the performance of different models with the help of different metrics such as R2 score and MAE
│
│
├── outputs/ #Contains the outputs that the repository generates.
│
└── README.md                # Project overview and instructions
