# Breast Cancer 10-Year Mortality Risk Prediction

## 🩺 Project Overview

This project aims to predict 10-year mortality risk for breast cancer patients using machine learning and survival analysis techniques. By leveraging clinical, genomic, and treatment-related features from the METABRIC dataset, the model provides personalized risk stratification and supports improved treatment decisions and patient counseling.

## 🧬 Dataset

* **Source**: METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
* **Size**: 2500+ patient records
* **Features Used**: Age at diagnosis, tumor stage, lymph nodes, mutation count, hormone therapy, histologic grade, surgery type, and genomic subtypes, etc.
* **Target**: `10_Year_Mortality` (binary - deceased within 120 months or not)

## 🔍 Project Workflow

1. **Data Preprocessing**

   * Missing value imputation (median/mode)
   * Outlier capping using IQR
   * Feature encoding (label + one-hot)
   * Feature scaling (StandardScaler)
2. **EDA & Visualization**

   * Distribution plots, count plots, correlation matrix
3. **Survival Analysis**

   * Kaplan-Meier curves by Tumor Stage
   * Cox Proportional Hazards model
4. **Feature Selection**

   * ANOVA F-test (Top 20 features selected)
5. **Model Building**

   * Algorithms: Logistic Regression, SVM, Decision Tree, Random Forest
   * GridSearchCV with StratifiedKFold cross-validation
6. **Model Evaluation**

   * Metrics: Accuracy, AUC, Precision, Recall, F1 Score
   * Confusion matrix, ROC curve
   * Feature Importance
7. **Deployment**

   * Streamlit app for interactive mortality risk prediction
   * Model inference using serialized `.pkl` files

## 📊 Model Performance Summary

| Model               | Accuracy  | AUC       | Precision | Recall    | F1 Score  | CV AUC    |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.707     | 0.803     | 0.512     | 0.730     | 0.602     | 0.788     |
| SVM                 | 0.695     | 0.794     | 0.498     | 0.737     | 0.594     | 0.794     |
| Decision Tree       | 0.727     | 0.778     | 0.538     | 0.691     | 0.605     | 0.748     |
| Random Forest       | **0.743** | **0.822** | **0.555** | **0.763** | **0.642** | **0.795** |

### ✅ Best Performing Model: Random Forest

* Achieved the highest **accuracy (74.3%)**, **AUC (0.822)**, **recall (76.3%)**, and **F1 score (0.642)**.
* Indicates robust generalization and balanced performance in both positive and negative class prediction.

## 📈 Visual Outputs

* ROC Curve
* Confusion Matrix for all models
* Feature Importance for Random Forest & Decision Tree
* Kaplan-Meier Curves by Tumor Stage
* Cox Hazard Ratios plot

## 🌐 Streamlit Web App Features

* Interactive UI with sliders and dropdowns
* Model selector (Random Forest / Logistic Regression)
* Real-time risk level prediction (Low / Medium / High)
* Displays 10-Year Mortality probability
* Uses consistent preprocessing pipeline from training phase

## 💾 Deployment & Files

* Models: `logistic_model.pkl`, `rf_model.pkl`
* Preprocessing: `scaler.pkl`, `label_encoders.pkl`, `selected_columns.pkl`
* App: `app.py` (Streamlit UI)
* Assets: All PNGs stored in `/PPT/` for presentation-ready use

## 🚀 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

## 📌 Future Enhancements

* Include additional genomic features (e.g., mRNA expression signatures)
* Integrate SHAP or LIME for interpretability
* Expand to multiclass risk levels (Low/Med/High/Very High)
* Containerize with Docker for scalable deployment

## 👨‍⚕️ Author

**Rejith R.**

* Domain: Healthcare + Data Science
* Connect on [LinkedIn](https://linkedin.com/in/your-profile)

---

> "Predicting mortality isn't just about algorithms—it's about empowering clinicians with actionable insights."
