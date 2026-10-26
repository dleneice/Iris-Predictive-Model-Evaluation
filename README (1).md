# Iris Predictive Model Evaluation

**Feature Engineering · Regression Error Analysis · Threshold-Based Classification**

An applied predictive analytics project using the Iris dataset to examine data preparation, engineered features, constant regression estimators, and percentile-based species classification. The emphasis is on **comparing prediction strategies, interpreting evaluation metrics, identifying limitations, and recommending defensible next steps**.

This project originated as Assignment 1 in DDS 8555 (Predictive Analysis).

## Project overview

The Iris dataset contains **150 observations** representing three species and four continuous measurements: sepal length, sepal width, petal length, and petal width. The analysis examines two related tasks:

1. **Regression:** Compare two constant estimators of sepal width using several error metrics.
2. **Classification:** Evaluate two rules that assign Iris species using training-set sepal-length percentiles.

These are intentionally simple estimators and decision rules, **not fitted multivariable machine-learning models**. They provide a starting point for evaluating prediction performance and its limitations.

## Tools and methods

- **Python**, **pandas**, **NumPy**, **scikit-learn**, Jupyter Notebook
- Dataset inspection and missing-value assessment
- An engineered ratio feature
- **80/20 stratified train/test split** (`random_state=42`)
- Regression error comparison: ME, MPE, MAE, MSE, MAPE
- Classification comparison: accuracy, precision, recall, F1-score
- Interpretation of bias, class-specific performance, and model limitations

## Data preparation and feature engineering

After checking the dataset structure, variable types, and missingness, the analysis constructed the following feature:

\[
\text{Engineered Feature} = \frac{\text{Sepal Length} \times \text{Sepal Width}}{\text{Petal Length} \times \text{Petal Width}}
\]

The engineered feature was **calculated but not used** in the subsequent regression-estimator or threshold-classifier comparisons. Its predictive value remains an open question for future work.

The data were split into **120 training observations and 30 test observations**, stratified by species.

## Regression: constant estimators of sepal width

Two values were derived from the training set and then used as constant predictions for every held-out observation:

- **Estimator 1:** Mean training-set **petal length**.
- **Estimator 2:** Mean training-set **(sepal length − petal width)**.

### Test-set results

| Evaluation metric | Estimator 1 | Estimator 2 | Interpretation |
|---|---:|---:|---|
| Mean error (ME) | -0.677 | -1.543 | Both overpredicted sepal width on average under the report's error convention |
| Mean percentage error (MPE) | -0.237 | -0.522 | Estimator 1 had less directional percentage bias |
| Mean absolute error (MAE) | 0.694 | 1.543 | Lower is better |
| Mean squared error (MSE) | 0.602 | 2.526 | Lower is better |
| Mean absolute percentage error (MAPE) | 0.242 | 0.522 | Lower is better |

**Finding:** Estimator 1 produced smaller errors on all five measures. However, both estimators returned a single prediction for all observations; neither accounts for individual flower characteristics. Better relative performance should not be interpreted as evidence of strong predictive accuracy.

## Classification: percentile-based species rules

The analysis used **sepal length only** and constructed two threshold rules from the training data:

- **Classifier 1:** 25th and 50th percentiles of training sepal length.
- **Classifier 2:** 50th and 75th percentiles of training sepal length.

The rules assigned one of three Iris species according to whether test sepal length crossed those thresholds.

### Test-set results

| Metric | Classifier 1 | Classifier 2 |
|---|---:|---:|
| Accuracy | 56.7% | 60.0% |
| Macro-averaged precision | 0.55 | 0.58 |
| Macro-averaged recall | 0.57 | 0.60 |
| Macro-averaged F1-score | 0.53 | 0.58 |

**Finding:** Classifier 2 achieved slightly higher overall accuracy and macro F1. On the 30-observation test set, the accuracy difference amounted to **one additional correct classification**.

Class-specific results add context: Classifier 1 achieved **100% recall for Virginica** but performed poorly on Versicolor. Classifier 2 was more balanced overall, yet Versicolor remained the most difficult species to distinguish. This demonstrates why overall accuracy should not be evaluated without examining performance by class.

## Interpretation and limitations

- **Constant predictions:** The regression estimators do not adjust to individual observations.
- **Restricted classifier inputs:** Both classification rules use only sepal length; the other original measurements and the engineered feature are not incorporated.
- **Small evaluation set:** With 30 test cases, small differences in correct predictions noticeably change accuracy.
- **Single split:** Results represent one stratified holdout split rather than performance across multiple resamples.
- **Metric interpretation:** Lower aggregate error or higher accuracy can obscure bias and uneven class-level performance.

## Recommended next steps

1. Establish a meaningful regression baseline using **mean training sepal width**, the actual regression target.
2. Fit regression models that use observation-level predictors, and compare them with the constant estimators.
3. Evaluate whether the engineered feature adds predictive information when combined with other measurements.
4. Compare more informative classification models and predictor sets.
5. Apply **cross-validation** to assess the stability of model comparisons.
6. Examine **confusion matrices** to understand where species misclassifications occur, especially for Versicolor.

These are proposed extensions, **not results claimed by the submitted analysis**.

## Reproducing the analysis

The project uses the Iris dataset distributed through `scikit-learn`; no private data is required.

Install the core libraries:

```bash
python -m pip install numpy pandas scikit-learn notebook
```

Open the accompanying Jupyter notebook (`.ipynb`) in JupyterLab or Jupyter Notebook, then run its cells in order. The exported HTML report provides a readable record of the submitted analysis and outputs.

> **Repository setup:** Add the actual completed notebook and exported HTML alongside this README. If the notebook's filename differs, use that filename when linking to it. Do not substitute an incomplete notebook for the completed analysis without noting the distinction.

## Project scope

This is an educational predictive-analysis exercise focused on analytical reasoning, reproducible comparisons, and evaluation literacy. It does **not** claim production deployment or the development of trained multivariable models.

## Author

**Donita Monroe**  
[GitHub](https://github.com/dleneice) · [LinkedIn](https://www.linkedin.com/in/donita-monroe-07517589/)
