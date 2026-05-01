South Africa Water Quality Prediction DashboardLive Dashboard: https://ai-dss-final-project-julius-rashba.streamlit.app
Author: Julius Rashba
Course: DATA 580F - AI-Driven Decision Support Systems
Institution: Binghamton Univsersity

Project Overview
This project uses machine learning to predict water quality at 162 monitoring sites across South Africa. The system predicts three key parameters:

- Total Alkalinity (mg/L)
- Electrical Conductance (mS/m)
- Dissolved Reactive Phosphorus (μg/L - pollution risk indicator)

<img width="539" height="420" alt="Screenshot 2026-04-30 at 11 51 05 PM" src="https://github.com/user-attachments/assets/bbc6da2b-7738-40b2-8c26-4594b8c7ac8b" />




Pipeline Steps:

1. Data Splitting - Load raw water quality data, create train/validation/test splits
2. TerraClimate Extraction - Extract 14 climate variables from Microsoft Planetary Computer
3. Landsat Extraction - Extract satellite indices (NDVI, NBR, NDBI) and land cover data
4. Copernicus DEM & ESA WorldCover - Extract elevation and global land cover features
5. Data Merging & Feature Engineering - Combine all features, create temporal features (rolling averages, lags)
6. Model Training & Evaluation - Train ML models, evaluate performance, save final models
