
# 📈 Time Series Forecasting with XGBoost, CatBoost & TimeSeries Cross-Validation

This repository contains a full end-to-end workflow for forecasting store-level sales using machine-learning–based time series models.
The project demonstrates how to preprocess temporal data, engineer meaningful features, apply *TimeSeriesSplit* for proper backtesting, and train competitive ensemble models like **XGBoost** and **CatBoost**.

The dataset structure reflects the **Store Sales - Time Series Forecasting** challenge (Kaggle), but the workflow is adaptable to any multivariate retail forecasting task.

---

## 🚀 Project Highlights

* ✔️ End-to-end time series ML pipeline
* ✔️ Feature engineering for dates, lags, rolling statistics
* ✔️ TimeSeriesSplit for robust model evaluation
* ✔️ XGBoost & CatBoost regression models
* ✔️ Handling train–test merge with flags
* ✔️ Submission file generation for competition-style evaluation

---

## 🗂️ Project Structure



```bash
# Main notebook: EDA → features → model → tuning → results
├── research/data/
              └── ...csv 
├── research/
        └── time-series-forecasting.ipynb            # Source dataset (not included in repo) 
├── README.md
└── requirements.txt                 # (optional) Python dependencies
```

## 📊 Workflow Overview

### **1. Importing & Exploring Data**

The notebook loads the core dataset components:

* Sales history
* Store & product metadata
* Holidays/events
* Transactions

Initial EDA includes:

* Missing value inspection
* Merging training & test sets using a "test" flag
* Basic trend visualizations

---

### **2. Data Analysis & Visualization**

Key plots and analysis steps include:

* Daily & weekly sales trends
* Category-level aggregations
* Store-level variability
* Holiday/event effects

These insights help drive feature engineering decisions.

---

### **3. Feature Engineering**

The project extracts rich temporal features such as:

* Date-based:
  `year`, `month`, `week`, `day`, `day_of_week`
* Lag features:
  `lag_7`, `lag_14`, `lag_28`
* Rolling statistics:
  `rolling_mean_7`, `rolling_std_7`, etc.
* Store-specific aggregations
* Promotions and holiday signals

These features significantly boost model predictive power.

---

### **4. Modeling: XGBoost & CatBoost**

Two strong gradient-boosting models are applied:

#### **🔸 XGBoost Regressor**

* Tuned using time-series folds
* Supports large, sparse feature matrices
* Efficient and scalable

#### **🔸 CatBoost Regressor**

* Handles categorical variables gracefully
* Requires minimal preprocessing
* Often outperforms other models on retail datasets

**Evaluation Strategy**:
`TimeSeriesSplit` ensures predictions for future periods are validated strictly using past data (no leakage).

---

### **5. Forecast Generation & Submission**

After training:

* Predictions are generated on the official test dataset
* Output is formatted into the required submission CSV
* Ready for Kaggle upload or internal deployment

---

## 🧪 Technologies Used

| Component     | Tools                           |
| ------------- | ------------------------------- |
| Programming   | Python, NumPy, Pandas           |
| Visualization | Matplotlib, Seaborn             |
| Modeling      | XGBoost, CatBoost, scikit-learn |
| Validation    | TimeSeriesSplit                 |
| Environment   | Jupyter Notebook                |

---

## 📁 Dataset

The workflow references the following dataset artifacts:

* `train.csv`
* `test.csv`
* `transactions.csv`
* `holidays_events.csv`
* `stores.csv` 
* `oil.csv`

> Replace paths with your local/Kaggle directory settings as needed.

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/roshankahaneDSAI/Time_Series_Forecasting_Using_CatBoost-.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook time-series-forecasting.ipynb
```

4. Update dataset paths & run all cells.

---

## 📌 Future Improvements

* Incorporate LightGBM for model comparison
* Hyperparameter tuning with Optuna
* Add Prophet / NeuralProphet baselines
* Implement stacking or ensemble blending
* Deploy endpoint using FastAPI

---

## 📝 Author

**Roshan kahane**
Data scientist | AI/ML Engineer | Time Series Specialist
