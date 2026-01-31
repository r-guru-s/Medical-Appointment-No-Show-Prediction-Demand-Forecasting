# Medical Appointment No-Show Prediction & Demand Forecasting

## 🎯 Project Overview
Healthcare ML system reducing **31.8% no-show rate** through risk prediction + demand forecasting.

**ROI**: $697K/year potential savings (34,832 × 25% × $80/appointment)

## 📊 Model Performance

### Classification (No-Show)
| Model | F1 | ROC-AUC | Status |
|-------|----|---------|--------|
| **LightGBM** | **0.642** | **0.787** ✅ | **Best** |
| XGBoost | 0.636 | 0.785 ✅ | Excellent |
| Random Forest | 0.621 | 0.774 ✅ | Good |
| Logistic Reg | 0.525 | 0.660 | Baseline |

**Target**: F1 ≥ 0.70, AUC ≥ 0.75 → **AUC exceeded**

### Regression (Demand)
| Model | MAPE | R² | RMSE | Status |
|-------|------|----|------|--------|
| **LightGBM** | **15.33%** ✅ | **0.795** ✅ | 130.8 | **Crushed** |
| Random Forest | 14.51% ✅ | 0.783 ✅ | 134.5 | Excellent |
| XGBoost | 16.44% ✅ | 0.775 ✅ | 136.9 | Excellent |

**Target**: MAPE < 20%, R² ≥ 0.65 → **Both exceeded by 20%+**

## 🛠️ Tech Stack
Python - Pandas - Scikit-learn - Logistic regression - **LightGBM** - XGBoost - Random Forest
Streamlit (app.py) - Joblib

## 📁 Files
medical-appointment-prediction/

├── Medical_appointment.ipynb     # Generates all CSVs + models

├── app.py                        # Streamlit app

├── noshow_classifier_final.pkl   # Pre-trained classifier

├── demand_regressor_final.pkl    # Pre-trained forecaster

├── Medical_appointment_data.csv    # RAW Data

├── Medical_appointment_features.csv    # Clean Data

└── README.md                    
