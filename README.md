🌊 South Africa Water Quality Prediction DashboardLive Dashboard: https://ai-dss-final-project-julius-rashba.streamlit.app
Author: Julius Rashba
Course: DATA 580F - AI-Driven Decision Support Systems
Institution: Binghamton Univsersity

Project Overview
This project uses machine learning to predict water quality at 162 monitoring sites across South Africa. The system predicts three key parameters:

- Total Alkalinity (mg/L)
- Electrical Conductance (mS/m)
- Dissolved Reactive Phosphorus (μg/L - pollution risk indicator)

Repository Structure
AI-DSS-Final-Project/
│
├── notebooks/           # Complete data pipeline (6 notebooks)
│   ├── Notebook 1 - Data Splitting/
│   ├── Notebook 2 - Terraclimate Extraction/
│   ├── Notebook 3 - Landsat Extraction/
│   ├── Notebook 4 - Copernicus DEM and ESA WorldCover Extraction/
│   ├── Notebook 5 - Data Merging and Feature Engineering/
│   └── Notebook 6 - Model Training and Evaluation/
│
├── dashboard/           # Streamlit web application
│   ├── app.py             # Main dashboard code
│   ├── utils.py           # Helper functions
│   ├── models/            # Trained ML models (.pkl)
│   ├── ml_ready_test.csv  # Test dataset with features
│   └── requirements.txt   # Python dependencies
│
└── data/               # Generated CSV files from pipeline


Each notebook is organized as follows:

Notebook X/
├── NotebookX.ipynb    # Executable Jupyter notebook
├── input/             # Required input files for this step
└── output/            # Generated output files

Pipeline Steps:

1. Data Splitting - Load raw water quality data, create train/validation/test splits
2. TerraClimate Extraction - Extract 14 climate variables from Microsoft Planetary Computer
3. Landsat Extraction - Extract satellite indices (NDVI, NBR, NDBI) and land cover data
4. Copernicus DEM & ESA WorldCover - Extract elevation and global land cover features
5. Data Merging & Feature Engineering - Combine all features, create temporal features (rolling averages, lags)
6. Model Training & Evaluation - Train ML models, evaluate performance, save final models
