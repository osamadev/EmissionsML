# 🌍 EmissionsML: London Atmospheric Emissions Analysis & Forecasting

This repository provides a **machine learning pipeline** for analyzing and predicting air pollution trends in **London**, based on the **[London Atmospheric Emissions Inventory (LAEI) 2019](https://data.london.gov.uk/dataset/london-atmospheric-emissions-inventory--laei--2019)** dataset.

🚀 **Purpose:**  
The project aims to **preprocess emissions data**, **apply machine learning models**, and **visualize insights** to support policymakers, environmental researchers, and sustainability advocates in **understanding London's air pollution trends**.

---

## 📂 Repository Structure

```
📁 EMISSIONSML/
│
├── checkpoints/                       # Model checkpoints (if applicable)
│
├── helpers/                           # Utility functions for data processing
│   ├── __pycache__/                   
│   ├── utilities.py                   # Helper functions for processing data, training and evaluating ML models 
│
├── LAEI2019_dataset/                   # Raw London emissions dataset
│
├── preprocessed_data/                  # Processed & cleaned data
│
├── emissions_data_preprocessing.ipynb  # Notebook for data cleaning & transformation
├── emissions_modeling.ipynb            # Notebook for ML modeling & evaluation
├── emissions_data_preprocessing.html   # HTML export of preprocessing notebook
├── emissions_modeling.html             # HTML export of modeling notebook
│
├── .gitignore                           # Ignore unnecessary files
├── .gitattributes                        # Git repository attributes
└── README.md                            
```

---

## 📌 Overview

### 1️⃣ **Data Source: London Atmospheric Emissions Inventory (LAEI) 2019**
- The dataset provides emissions estimates for **CO₂, NOx, PM10, and PM2.5** across various London boroughs.
- Data includes **transport, domestic, commercial, and industrial emissions**.
- Available at: **[LAEI 2019 Dataset](https://data.london.gov.uk/dataset/london-atmospheric-emissions-inventory--laei--2019)**.

### 2️⃣ **Data Preprocessing (`emissions_data_preprocessing.ipynb`)**
- Cleans and structures the dataset.
- Filters emissions by borough, year, and pollutant type.
- Standardizes units for CO₂ and other emissions.
- Outputs a **preprocessed dataset** ready for modeling.

### 3️⃣ **Machine Learning Modeling (`emissions_modeling.ipynb`)**
- Applies **ML models** (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) to predict emissions trends.
- Evaluates models using **R² score, RMSE, and feature importance analysis**.
- Provides **visualizations** (actual vs. predicted emissions, feature importance visuals).

---

## 📊 Sample Outputs

| Feature | Description |
|---------|------------|
| `final_dataset.csv` | Preprocessed dataset after data wrangling and cleansing |
| `random_forest_model.pkl` | Trained emissions regession model |

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-org/emissionsml.git
cd emissionsml
```

### 2️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3️⃣ Run the Notebooks
```bash
jupyter notebook
```
- Open **`emissions_data_preprocessing.ipynb`** and execute all cells.
- Open **`emissions_modeling.ipynb`** to build models and generate predictions.

---

## 📈 Future Enhancements

- 🕰 **Time-Series Forecasting** (ARIMA, Prophet) for long-term trends.
- 🌍 **Geospatial Analysis** to identify pollution hotspots.
- 🏙 **Borough-Level Reports** for detailed insights.

---

## 📜 License

MIT License – Free to use & modify for research & policy applications.

---
