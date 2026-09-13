# MDI3003 — Advanced Predictive Analytics

## Experiment 08: Agricultural Predictive Analytics

**Student Name:** YUVARAJ K N
**Registration Number:** 23MID0421
**Institution:** Vellore Institute of Technology, Vellore
**Course:** MDI3003 — Advanced Predictive Analytics
**Semester:** Fall 2026–2027

---

## 1. Project Title

**Agricultural Predictive Analytics: Rice Yield Prediction with Crop-Label Classification Extension**

---

## 2. Project Overview

This project implements an agricultural predictive-analytics workflow using Indian agricultural datasets. The main objective is to build and evaluate machine-learning models for **rice-yield prediction** using district-level agricultural data.

The project follows a chronological evaluation strategy:

* Earlier years are used for training.
* The two years immediately before the test period are used for validation.
* The latest two observed years are reserved for final testing.
* The final test set is used only after selecting the best model using validation MAE.

The project also includes a separate crop-label classification experiment using soil and environmental attributes.

The analysis focuses on reproducibility, data auditing, leakage prevention, model comparison, error analysis, and responsible interpretation of agricultural predictions.

---

## 3. Objectives

The main objectives of this assignment are:

1. Formulate a measurable agricultural prediction problem.
2. Inspect and audit the datasets.
3. Verify data types, missing values, units, and dataset structure.
4. Prepare suitable features and target variables.
5. Prevent target leakage by excluding unsuitable predictors.
6. Perform exploratory data analysis.
7. Implement a median baseline and machine-learning regression models.
8. Compare Ridge Regression, Decision Tree, and Random Forest models.
9. Evaluate models using MAE, RMSE, and R².
10. Use chronological train-validation-test splitting.
11. Perform error analysis and year-wise evaluation.
12. Perform rolling-origin validation.
13. Save and reload the trained model.
14. Generate reproducible results and acceptance-audit files.
15. Discuss limitations and responsible agricultural use.

---

## 4. Datasets Used

### Dataset 1 — District-wise, Season-wise Crop Production Data

Dataset 1 contains district-wise and season-wise agricultural information from India.

Important variables include:

* State
* District
* Season
* Crop
* Crop year
* Area
* Production
* Yield

The analysis focuses on the **Rice** crop.

The following variables were excluded from the core prediction model:

* Area
* Production
* Crop

Area and production were excluded because they may create target leakage or may not be available at the intended prediction time.

---

### Dataset 2 — ICRISAT District-Level Agriculture Data

Dataset 2 contains district-level yearly agricultural data, including:

* District code
* Year
* State code
* State name
* District name
* Rice area
* Rice production
* Rice yield
* Area, production, and yield variables for other crops

Dataset 2 contains:

* **16,146 rows**
* **80 columns before cleaning**

The rice-yield target is:

```text
RICE YIELD (Kg per ha)
```

The target was renamed to:

```text
rice_yield_kg_ha
```

The core Dataset 2 model uses the following predictors:

```text
state
district
year
```

The following variables were excluded from the core model:

* Rice area
* Rice production
* Other crop area variables
* Other crop production variables
* Other crop yield variables
* District code
* State code

The exclusion of area and production variables helps avoid leakage and keeps the prediction setting focused on information intended to be available before harvest.

---

### Dataset 3 — Crop Recommendation Dataset

Dataset 3 is used for the classification extension.

The input features are:

* N
* P
* K
* Temperature
* Humidity
* pH
* Rainfall

The target variable is:

```text
label
```

The classification experiment compares:

* Most-frequent baseline
* Logistic Regression
* Decision Tree
* Random Forest

The classification results are treated separately from the rice-yield regression task.

---

### Dataset 4 — FAOSTAT Agricultural Data

Dataset 4 was obtained from FAOSTAT.

The dataset contains country-level agricultural information, including:

* Area
* Item
* Element
* Year
* Unit
* Value

The downloaded data contained only a limited number of years and was aggregated at the country level. Therefore, it was not suitable for the main district-level rice-yield prediction task.

Dataset 4 was used for:

* Dataset inspection
* Basic exploratory analysis
* Understanding country-level agricultural indicators
* Documenting limitations of transferring country-level results to district-level prediction

Dataset 4 was not treated as a valid replacement for the district-level core regression task.

---

## 5. Main Prediction Problem

The main regression problem is:

> Predict rice yield for an observed Indian district using the state, district, season or year-related information available before the target harvest.

For the district-level regression workflow, the target is rice yield.

The prediction unit is a district-level agricultural observation associated with a particular year.

The intended use is educational analysis for regional agricultural planning. The model is not presented as an operational agricultural forecasting system.

---

## 6. Machine-Learning Models

The following regression models were implemented:

### 6.1 Median Baseline

The baseline predicts the median rice yield from the training data for every observation.

The median baseline is useful because it provides a simple reference point and minimizes absolute error among constant predictions.

### 6.2 Ridge Regression

Ridge Regression was implemented with:

```text
alpha = 1.0
```

Categorical variables were one-hot encoded, and the year variable was imputed and standardized inside the preprocessing pipeline.

### 6.3 Decision Tree Regression

The Decision Tree model used the following configuration:

```text
max_depth = 6
min_samples_leaf = 10
random_state = 42
```

### 6.4 Random Forest Regression

The Random Forest model used the following configuration:

```text
n_estimators = 60
max_depth = 12
min_samples_leaf = 5
n_jobs = 2
random_state = 42
```

These settings were used as fixed teaching configurations rather than as optimized agronomic defaults.

---

## 7. Preprocessing

The preprocessing workflow includes:

* Missing-value handling
* One-hot encoding of categorical variables
* Standardization of the year feature
* Training-only fitting of preprocessing transformations
* Unknown-category handling using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing steps were placed inside a scikit-learn `Pipeline` and `ColumnTransformer`.

This prevents information from the validation and test datasets from influencing the learned preprocessing transformations.

---

## 8. Evaluation Strategy

The chronological split strategy is:

* **Training set:** all years before the final four observed years
* **Validation set:** the two years immediately before the final two years
* **Test set:** the latest two observed years

The test set is kept separate until model selection is complete.

The primary evaluation metric is:

```text
Mean Absolute Error — MAE
```

Secondary metrics are:

* Root Mean Squared Error — RMSE
* R² score

MAE is prioritized because it is easy to interpret in the target unit and is less dominated by large individual errors than RMSE.

---

## 9. Exploratory Data Analysis

The EDA stage includes:

* Rice-yield distribution
* Average rice yield over time
* State-wise average rice yield
* District observation counts
* Rice area versus rice yield
* Correlation heatmap of rice-related numeric variables
* Dataset missing-value analysis
* Year-coverage analysis

The figures are used to inspect:

* Target distribution
* Temporal trends
* Geographic differences
* Observation imbalance
* Possible outliers
* Relationships between agricultural variables

All figures should include suitable titles, axis labels, and units.

---

## 10. Error Analysis

The error-analysis stage includes:

* Actual versus predicted rice yield
* Residual distribution
* Largest absolute prediction errors
* Year-wise test performance
* Test-set MAE, RMSE, and R²
* Inspection of the largest five nonzero errors where available

The causes of errors are treated as hypotheses rather than confirmed facts. Possible explanations may include:

* Differences between districts
* Sparse historical observations
* Changes in agricultural practices
* Irrigation differences
* Weather variation
* Crop-variety differences
* Reporting inconsistencies
* District-boundary changes
* Unobserved environmental variables

The available datasets do not independently confirm these explanations.

---

## 11. Rolling-Origin Validation

Rolling-origin validation was implemented to examine model stability across different historical development periods.

For each rolling origin:

1. Only earlier years were used for training.
2. A later development year was used for validation.
3. A fresh preprocessing and model pipeline was fitted.
4. MAE was calculated for the rolling validation year.

Rolling-origin validation is used as a descriptive robustness analysis. It does not establish confidence intervals or prove that the model will perform reliably in future operational settings.

---

## 12. Reproducibility

The project includes:

* Fixed random seed where applicable
* Saved preprocessing and model pipeline
* Saved validation results
* Saved final test results
* Saved error-analysis results
* Saved year-wise evaluation results
* Saved rolling-origin results
* Saved model-reload results
* Saved acceptance-audit files
* Saved final summary files

The saved model was reloaded and used to generate test predictions again. The original and reloaded predictions were compared to verify reproducibility.

Only trusted local model files should be loaded because serialized model files may execute code during deserialization.

---

## 13. Generated Artifacts

The project generates files such as:

```text
dataset2_source_unit_audit.csv
dataset2_basic_audit.csv
dataset2_validation_results.csv
dataset2_final_test_results.csv
dataset2_error_analysis.csv
dataset2_yearwise_results.csv
dataset2_rolling_origin_results.csv
dataset2_reload_results.csv
dataset2_complete_model_comparison.csv
dataset2_final_summary.json
dataset2_acceptance_audit.csv
dataset2_acceptance_audit.json
dataset2_final_report.txt
dataset2_best_model_pipeline.joblib
```

Similar result files were generated for the other completed datasets where applicable.

---

## 14. Repository Structure

A recommended GitHub repository structure is:

```text
MDI3003-Experiment-08/
│
├── README.md
│
├── notebooks/
│   ├── Dataset1_Rice_Yield_Regression.ipynb
│   ├── Dataset2_ICRISAT_Regression.ipynb
│   ├── Dataset3_Crop_Classification.ipynb
│   └── Dataset4_FAOSTAT_Analysis.ipynb
│
├── reports/
│   └── MDI3003_Experiment08_Report.pdf
│
├── results/
│   ├── dataset1_validation_results.csv
│   ├── dataset1_test_results.csv
│   ├── dataset2_validation_results.csv
│   ├── dataset2_final_test_results.csv
│   ├── dataset2_error_analysis.csv
│   ├── dataset2_yearwise_results.csv
│   └── dataset2_rolling_origin_results.csv
│
├── models/
│   └── dataset2_best_model_pipeline.joblib
│
├── artifacts/
│   ├── dataset2_final_summary.json
│   ├── dataset2_acceptance_audit.csv
│   ├── dataset2_acceptance_audit.json
│   └── dataset2_reload_results.csv
│
└── figures/
    ├── rice_yield_distribution.png
    ├── rice_yield_over_time.png
    ├── state_wise_rice_yield.png
    ├── actual_vs_predicted.png
    └── residual_distribution.png
```

The exact filenames may be adjusted according to the files generated in Google Colab.

---

## 15. Limitations

The following limitations apply to this project:

1. The datasets are aggregated at district or country level rather than farm level.
2. The model does not include detailed soil, irrigation, weather, crop-variety, or management information.
3. District boundaries and reporting practices may change over time.
4. Historical data may contain revisions that are not reconstructed by this experiment.
5. Chronological evaluation does not prove spatial generalization to unseen districts.
6. The model should not be interpreted as an agronomically optimal crop-selection system.
7. Predictions are estimates and not guaranteed agricultural outcomes.
8. The fixed model settings are instructional and are not claimed to be optimal.
9. Dataset 4 has a different aggregation level and cannot directly validate district-level predictions.
10. The analysis does not establish causal relationships between features and yield.
11. The model should not be used for real resource-allocation decisions without expert review and additional validation.

---

## 16. Responsible Agricultural Analytics

Agricultural predictions should be interpreted with caution.

Before using such a system for practical decisions, additional information would be required, including:

* Soil properties
* Irrigation availability
* Weather forecasts
* Crop variety
* Pest and disease information
* Local agricultural practices
* Market prices
* Input costs
* Farmer and agronomist knowledge
* Regional validation data

The model should support analysis and discussion rather than replace agricultural experts.

Regional errors should be examined carefully because small or underrepresented districts may behave differently from regions with more observations.

---

## 17. Conclusion

This assignment implements a reproducible agricultural predictive-analytics workflow involving data auditing, exploratory analysis, chronological splitting, preprocessing, baseline comparison, regression modeling, evaluation, error analysis, robustness analysis, and model reloading.

The main regression task predicts rice yield using district, state, and year-related information while excluding area and production variables from the core predictor set to reduce leakage risk.

The project demonstrates how machine-learning models can be compared systematically, but the results should be interpreted as an educational retrospective analysis rather than a validated operational agricultural forecasting system.

---

## 18. References

1. Government of India Open Government Data Platform — District-wise, season-wise crop production statistics.
2. ICRISAT — District-level agricultural data and crop datasets.
3. TCI/Cornell — District-Level Database for Indian Agriculture and Allied Sectors.
4. FAOSTAT — Crops and livestock products.
5. Atharva Ingle — Crop Recommendation Dataset.
6. scikit-learn documentation — Pipelines, preprocessing, regression models, and evaluation metrics.
7. MDI3003 Advanced Predictive Analytics Experiment 08 Student Manual, Version 3.0.

---

## 19. Academic Declaration

This repository contains the implementation and analysis completed for MDI3003 Advanced Predictive Analytics, Experiment 08.

Any external datasets, reused code, libraries, and documentation used in the project should be acknowledged appropriately. Raw datasets should only be redistributed when their licenses and access conditions permit redistribution.
# APAASSIGNMENT8
