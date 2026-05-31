# HR Employee Attrition Prediction
### Vignan Institute of Technology and Management, Berhampur
**Final Year Project | B.Tech CSE (Data Science) | 2026**

---

## Project Overview
This project predicts whether an employee will leave the company (attrition)
using Machine Learning. It uses the IBM HR Analytics dataset and compares
4 different ML models to find the best predictor.

---

## Project Structure
```
project_files/
├── data/
│   └── HR-Employee-Attrition.csv    ← Dataset (1470 employees, 35 features)
├── Employee_Attrition.ipynb          ← Complete analysis: EDA + Models
├── dashboard.py                      ← Streamlit interactive dashboard
├── requirements.txt                  ← Required libraries
└── README.md                         ← This file
```

---

## How to Run

### Step 1: Install libraries
```bash
pip install -r requirements.txt
```

### Step 2: Run Jupyter Notebook (for analysis)
```bash
jupyter notebook Employee_Attrition.ipynb
```

### Step 3: Run Streamlit Dashboard
```bash
streamlit run dashboard.py
```
Open browser: http://localhost:8501

---

## Dataset
- **Source**: IBM HR Analytics (Kaggle)
- **Size**: 1470 employees, 35 features
- **Target**: Attrition (Yes/No)
- **Class Imbalance**: 84% stayed, 16% left

---

## Data Cleaning
Removed 4 constant columns (no predictive value):
- `EmployeeCount` — same value (1) for all employees
- `StandardHours` — same value (80) for all employees
- `Over18` — same value ('Y') for all employees
- `EmployeeNumber` — just an ID, not a feature

---

## Models Compared
| Model | Why Used |
|---|---|
| Logistic Regression | Baseline model, simple and interpretable |
| Decision Tree | Easy to visualize decision logic |
| **Random Forest** ✅ | Best performer — highest AUC and Accuracy |
| Gradient Boosting | Strong ensemble, competitive accuracy |

---

## Model Performance Results
| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
|---|---|---|---|---|---|
| Logistic Regression | 86.39% | 62.07% | 38.30% | 47.37% | 84.38% |
| Decision Tree | 78.91% | 29.73% | 23.40% | 26.19% | 67.69% |
| **Random Forest** ✅ | **87.07%** | **80.00%** | 25.53% | 38.71% | **86.86%** |
| Gradient Boosting | 86.39% | 62.96% | 36.17% | 45.95% | 85.82% |

---

## Best Model — Random Forest
**Random Forest** achieves the **highest AUC Score (86.86%)** and **highest Accuracy (87.07%)**.

It works by building an ensemble of 500 decision trees and combining their predictions —
this reduces overfitting and captures complex, non-linear relationships in HR data.

---

## Key Findings
1. Young employees (25-35) leave most frequently
2. Low salary is the strongest attrition predictor
3. Overtime employees are 3x more likely to leave
4. Sales department has the highest attrition rate
5. Low job satisfaction strongly predicts attrition
6. New employees (0-3 years) have the highest attrition risk

---

## Team
Vignan Institute of Technology and Management
B.Tech CSE (Data Science) — Batch 2022-2026
