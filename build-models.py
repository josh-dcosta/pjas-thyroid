import sys
from pathlib import Path

# Standard library imports
import warnings

# Third-party imports
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import graphviz

from imblearn.over_sampling import SMOTE
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler
)
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# External libraries for specialized models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Local imports
sys.path.append(str("module"))
from pytorch_ann import PyTorchANN

# Matplotlib settings
plt.style.use('fivethirtyeight')
# %matplotlib inline

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration constants
# =============================================================================
TEST_SIZE = 0.30
USE_ALL_FEATURES = False
USE_SAVE_MODEL = True
USE_TEST_MODEL_PERFORMANCE = True
SHOW_FEATURE_IMPORTANCE = True

USE_SHOW_PLOTS = False
USE_SHOW_MODEL_METRICS = False
SHOW_CLASSIFICATION_REPORT = False
SHOW_ROC_CURVE = False
SHOW_CONFUSION_MATRIX = False

# Path to the CSV file containing the thyroid cancer recurrence data
csv_path = "data/thyroid_diff.csv"

# =============================================================================
# Data reading and initial exploration
# =============================================================================
df = pd.read_csv(csv_path)

# Optional: Quick overview
# df.head()

# =============================================================================
# Exploratory data analysis (optional visualization)
# =============================================================================
if USE_SHOW_PLOTS:
    # Box plot of Age by Recurred
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Recurred', y='Age', data=df)
    plt.title('Box Plot of Age by Recurred')
    plt.xlabel('Recurred')
    plt.ylabel('Age')
    plt.show()

    # Box plot of Age by Stage
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Stage', y='Age', data=df)
    plt.title('Box Plot of Age by Stage')
    plt.xlabel('Stage')
    plt.ylabel('Age')
    plt.show()

    # Gender by Risk
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Gender', hue='Risk', palette='viridis')
    plt.title('Grouped Bar Chart of Gender vs Risk')
    plt.show()

    # Scatter plot of Age vs. Tumor size (T)
    fig = px.scatter(
        df,
        x='Age',
        y='T',
        title='Scatter Plot of Age vs. Tumor size (T)',
        labels={'T': 'Tumor size (T)', 'Age': 'Age'}
    )
    fig.show()

# =============================================================================
# Boolean encoding ('Yes'/'No') columns
# =============================================================================
mapping_yes_no = {'No': 0, 'Yes': 1}
boolean_columns = ['Smoking', 'Hx Smoking', 'Hx Radiotherapy', 'Recurred']
df[boolean_columns] = df[boolean_columns].map(mapping_yes_no.get)

# =============================================================================
# Label encoding for columns with only 2 categories
# =============================================================================
le = LabelEncoder()
columns_to_encode = ['Gender', 'Focality', 'M']

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

# =============================================================================
# Mapping categorical columns with multiple categories
# =============================================================================
# 1. Physical Examination
mapping_phys_exam = {
    'Normal': 0,
    'Diffuse goiter': 1,
    'Single nodular goiter-left': 2,
    'Single nodular goiter-right': 2,
    'Multinodular goiter': 3
}
df['Physical Examination'] = df['Physical Examination'].map(mapping_phys_exam)
if np.isnan(df['Physical Examination'].unique()).any():
    raise ValueError("Check 'Physical Examination' mappings for unseen values.")

# 2. Adenopathy
mapping_adenopathy = {
    'No': 0,
    'Right': 1,
    'Left': 1,
    'Posterior': 2,
    'Bilateral': 3,
    'Extensive': 4
}
df['Adenopathy'] = df['Adenopathy'].map(mapping_adenopathy)
if np.isnan(df['Adenopathy'].unique()).any():
    raise ValueError("Check 'Adenopathy' mappings for unseen values.")

# 3. Pathology
mapping_pathology = {
    'Micropapillary': 0,
    'Papillary': 1,
    'Follicular': 2,
    'Hurthle cell': 3
}
df['Pathology'] = df['Pathology'].map(mapping_pathology)
if np.isnan(df['Pathology'].unique()).any():
    raise ValueError("Check 'Pathology' mappings for unseen values.")

# 4. Risk
mapping_risk = {
    'Low': 0,
    'Intermediate': 1,
    'High': 2
}
df['Risk'] = df['Risk'].map(mapping_risk)
if np.isnan(df['Risk'].unique()).any():
    raise ValueError("Check 'Risk' mappings for unseen values.")

# 5. T (Tumor)
mapping_t = {
    'T1a': 0,
    'T1b': 1,
    'T2': 2,
    'T3a': 3,
    'T3b': 4,
    'T4a': 5,
    'T4b': 6
}
df['T'] = df['T'].map(mapping_t)
if np.isnan(df['T'].unique()).any():
    raise ValueError("Check 'T' mappings for unseen values.")

# 6. N (Nodes)
mapping_n = {
    'N0': 0,
    'N1a': 1,
    'N1b': 2
}
df['N'] = df['N'].map(mapping_n)
if np.isnan(df['N'].unique()).any():
    raise ValueError("Check 'N' mappings for unseen values.")

# 7. Stage
mapping_stage = {
    'I': 0,
    'II': 1,
    'III': 2,
    'IVA': 3,
    'IVB': 4
}
df['Stage'] = df['Stage'].map(mapping_stage)
if np.isnan(df['Stage'].unique()).any():
    raise ValueError("Check 'Stage' mappings for unseen values.")

# 8. Response
mapping_response = {
    'Excellent': 0,
    'Indeterminate': 1,
    'Biochemical Incomplete': 2,
    'Structural Incomplete': 3
}
df['Response'] = df['Response'].map(mapping_response)
if np.isnan(df['Response'].unique()).any():
    raise ValueError("Check 'Response' mappings for unseen values.")

# =============================================================================
# One-Hot Encoding (Thyroid Function)
# =============================================================================
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(df[['Thyroid Function']]).toarray()
feature_labels = np.array(ohe.categories_).ravel()

# Create a DataFrame from the one-hot encoded array
features = pd.DataFrame(feature_array, columns=feature_labels)

# Merge one-hot features back into the main DataFrame
df = pd.concat([df, features], axis=1).drop(['Thyroid Function'], axis='columns')

# =============================================================================
# Scaling the 'Age' column using MinMaxScaler
# =============================================================================
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])[:, 0]

# Optionally save the scaler
if USE_SAVE_MODEL:
    joblib.dump(scaler, "model/pjas-thyroid-Scaler.pkl")

# =============================================================================
# Select features for training
# =============================================================================
if USE_ALL_FEATURES:
    df = df[
        [
            'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy',
            'Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Euthyroid',
            'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism',
            'Physical Examination', 'Adenopathy', 'Pathology', 'Focality',
            'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred'
        ]
    ]
else:
    df = df[['Age', 'Gender', 'T', 'N', 'Focality', 'Response', 'Recurred']]

# Split into input features (X) and target (y)
X = df.drop('Recurred', axis='columns')
y = df['Recurred']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

# =============================================================================
# Handle class imbalance using SMOTE
# =============================================================================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# =============================================================================
# Instantiate classification models
# =============================================================================
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'ANN': PyTorchANN(
        input_dim=X_train.shape[1],
        hidden_dim=32,
        num_hidden_layers=1,
        epochs=20,
        batch_size=32,
        learning_rate=0.001
    )
}

# =============================================================================
# Training and scoring models
# =============================================================================
model_scores = []
for name, model in models.items():
    if USE_ALL_FEATURES:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 2)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Some models (e.g., SVC with default settings) do not have predict_proba
        y_pred_proba = [0] * len(y_test)

    model_scores.append((name, y_test, y_pred, y_pred_proba, model, accuracy))

# Sort models by accuracy
sorted_models = sorted(model_scores, key=lambda x: x[5], reverse=True)

# =============================================================================
# Evaluation and optional reporting
# =============================================================================
model_reports = []

for model_info in sorted_models:
    name, true_vals, preds, preds_proba, clf, accuracy = model_info
    fpr, tpr, _ = roc_curve(true_vals, preds_proba)
    roc_auc = auc(fpr, tpr) * 100
    model_accuracy = accuracy_score(true_vals, preds) * 100
    cls_report = classification_report(true_vals, preds, output_dict=True)

    # Confusion matrix metrics
    cm = confusion_matrix(true_vals, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) else 0

    # Optional logging
    if USE_SHOW_MODEL_METRICS:
        print(f"Model = {name}")
        print(f"Accuracy = {model_accuracy:.2f}")
        print(f"AUC = {roc_auc:.2f}")
        print(f"Sensitivity (Recall) = {sensitivity:.2f}")
        print(f"Specificity = {specificity:.2f}")

    if SHOW_CLASSIFICATION_REPORT:
        print(f"{name} Classification Report:\n{cls_report}\n")

    if SHOW_ROC_CURVE:
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Classification Models")
        plt.legend(loc="lower right")
        plt.show()

    if SHOW_CONFUSION_MATRIX:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    if SHOW_FEATURE_IMPORTANCE and hasattr(clf, "feature_importances_"):
        feature_names = list(df.columns)
        importances = clf.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 7))
        plt.title(f'Feature Importances - {name}')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    # Optional full-dataset "test" performance
    if USE_TEST_MODEL_PERFORMANCE:
        total_rows = len(df)
        matched_count = 0
        failed_predictions = []

        for idx, row in df.iterrows():
            features = row.drop('Recurred').values.reshape(1, -1)
            actual = row['Recurred']
            predicted = clf.predict(features)[0]
            if predicted == actual:
                matched_count += 1
            else:
                failed_predictions.append({
                    'Row Number': idx + 1,
                    'Actual': actual,
                    'Predicted': predicted
                })

        test_accuracy = (matched_count / total_rows) * 100

        model_reports.append({
            'Model Name': name,
            'Total Rows Processed': total_rows,
            'Successfully Matched': matched_count,
            'Failed Count': total_rows - matched_count,
            'Test Accuracy': f"{test_accuracy:.2f}",
            'Model Accuracy': f"{model_accuracy:.2f}",
            'Model Sensitivity': f"{sensitivity:.2f}",
            'Model Specificity': f"{specificity:.2f}",
            'Model AUC': f"{roc_auc:.2f}",
            'Failed Predictions': failed_predictions
        })

        # Optionally save the model
        if USE_SAVE_MODEL:
            joblib.dump(clf, f'model/pjas-thyroid-{name}.pkl')

# =============================================================================
# Model summary report
# =============================================================================
if USE_TEST_MODEL_PERFORMANCE:
    sorted_reports = sorted(
        model_reports,
        key=lambda x: float(x['Test Accuracy']),
        reverse=True
    )
    model_summary_report = pd.DataFrame(sorted_reports)
    model_summary_report.to_csv('report/model_summary_report_6_smote.csv', index=False)