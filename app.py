import os
import joblib
import gradio as gr
import pandas as pd

# =============================================================================
# Configuration & Setup
# =============================================================================
MODEL_SCORES_FILE = "report/model_summary_report_6_smote.csv"
SCALER_FILE = "model/pjas-thyroid-Scaler.pkl"

# Dictionary of model names and their respective pickle files
MODEL_PATHS = {
    'AdaBoost': 'pjas-thyroid-AdaBoost.pkl',
    'CatBoost': 'pjas-thyroid-CatBoost.pkl',
    'Decision Tree': 'pjas-thyroid-Decision Tree.pkl',
    'Gaussian Naive Bayes': 'pjas-thyroid-Gaussian Naive Bayes.pkl',
    'Gradient Boosting': 'pjas-thyroid-Gradient Boosting.pkl',
    'K-Nearest Neighbors': 'pjas-thyroid-K-Nearest Neighbors.pkl',
    'Logistic Regression': 'pjas-thyroid-Logistic Regression.pkl',
    'Random Forest': 'pjas-thyroid-Random Forest.pkl',
    'Support Vector Machine': 'pjas-thyroid-Support Vector Machine.pkl',
    'XGBoost': 'pjas-thyroid-XGBoost.pkl'
}

# =============================================================================
# Load model performance data
# =============================================================================
if not os.path.exists(MODEL_SCORES_FILE):
    raise FileNotFoundError(
        f"The model scores file '{MODEL_SCORES_FILE}' was not found. "
        "Please ensure it exists in the 'report/' directory."
    )

model_scores_df = pd.read_csv(MODEL_SCORES_FILE)
model_scores_df = model_scores_df[model_scores_df['Model Name'] != 'ANN']

required_columns = {'Model Name', 'Model Sensitivity', 'Model Specificity'}
if not required_columns.issubset(model_scores_df.columns):
    raise ValueError(
        f"The file '{MODEL_SCORES_FILE}' must contain the columns: "
        f"{required_columns}"
    )

model_performance = (
    model_scores_df
    .set_index('Model Name')[['Model Sensitivity', 'Model Specificity']]
    .T
    .to_dict()
)

# =============================================================================
# Load models
# =============================================================================
loaded_models = {}
for model_name, pickle_file in MODEL_PATHS.items():
    model_file_path = os.path.join("model", pickle_file)
    if os.path.exists(model_file_path):
        try:
            loaded_models[model_name] = joblib.load(model_file_path)
        except Exception as e:
            print(f"Error loading {model_name} from {pickle_file}: {e}")
    else:
        print(f"Model file for {model_name} not found at {model_file_path}.")

# =============================================================================
# Prediction function
# =============================================================================
def predict_cancer(age, gender, T, N, focality, response):
    """
    Predicts thyroid cancer recurrence risk using multiple loaded ML models.
    Returns an HTML table with each model's performance and prediction.
    """
    # Validate inputs
    if age is None or not (1 <= age <= 100):
        return "🔴 **Error:** Age must be a number between 1 and 100."
    if gender not in ["Female", "Male"]:
        return "🔴 **Error:** Please select a valid gender."
    if T is None:
        return "🔴 **Error:** Please select a valid T (Tumor Size) option."
    if N is None:
        return "🔴 **Error:** Please select a valid N (Lymph Node Spread) option."
    if focality is None:
        return "🔴 **Error:** Please select a valid Focality option."
    if response is None:
        return "🔴 **Error:** Please select a valid Response option."

    # Convert categorical inputs to numeric
    gender_val = 0 if gender == "Female" else 1
    response_val = int(response)
    T_val = int(T)
    N_val = int(N)
    focality_val = int(focality)

    # Prepare features for prediction
    features = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_val],
        'T': [T_val],
        'N': [N_val],
        'Focality': [focality_val],
        'Response': [response_val]
    })

    # Validate and load scaler
    if not os.path.exists(SCALER_FILE):
        return (
            "🔴 **Error:** Scaler file not found. "
            "Please contact the administrator."
        )
    scaler = joblib.load(SCALER_FILE)
    features[['Age']] = scaler.transform(features[['Age']])

    # Sort models by sensitivity (highest first)
    sorted_model_names = sorted(
        model_performance.keys(),
        key=lambda m: model_performance[m]['Model Sensitivity'],
        reverse=True
    )

    # Prepare HTML table header
    table_header = (
        "<table>"
        "<thead>"
        "<tr>"
        "<th>Model</th>"
        "<th style='color:#FBCEB1;'>Recurrence Accuracy (%)</th>"
        "<th style='color:green;'>Non-Recurrence Accuracy (%)</th>"
        "<th>Prediction</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
    )

    table_rows = []
    can_recur_emoji = "🔴"
    cannot_recur_emoji = "🟢"

    # Generate predictions for each model
    for model_name in sorted_model_names:
        model = loaded_models.get(model_name)
        if not model:
            row = (
                f"<tr>"
                f"<td>{model_name}</td><td>N/A</td><td>N/A</td>"
                f"<td>Error: Model not loaded</td>"
                f"</tr>"
            )
            table_rows.append(row)
            continue

        try:
            prediction = model.predict(features)
            pred_value = prediction[0]
            pred_text = (
                f"{can_recur_emoji} Can recur"
                if pred_value == 1
                else f"{cannot_recur_emoji} Cannot-recur"
            )

            sensitivity = model_performance[model_name]['Model Sensitivity']
            specificity = model_performance[model_name]['Model Specificity']

            row = (
                f"<tr>"
                f"<td>{model_name}</td>"
                f"<td>{sensitivity:.2f}%</td>"
                f"<td>{specificity:.2f}%</td>"
                f"<td>{pred_text}</td>"
                f"</tr>"
            )
            table_rows.append(row)
        except Exception as e:
            row = (
                f"<tr>"
                f"<td>{model_name}</td>"
                f"<td>N/A</td>"
                f"<td>N/A</td>"
                f"<td>Error: {str(e)}</td>"
                f"</tr>"
            )
            table_rows.append(row)

    table_footer = "</tbody></table>"
    html_table = table_header + "".join(table_rows) + table_footer
    success_message = "<br><br>✅ <strong>Prediction completed successfully.</strong>"

    return html_table + success_message

def clear_md():
    """Clears the Gradio Markdown output."""
    return ""

# =============================================================================
# Gradio Interface
# =============================================================================
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Thyroid Cancer Recurrence Predictor")

    with gr.Row():
        age_slider = gr.Number(
            label="Age",
            value=44,
            interactive=True,
            elem_id="age-box",
            step=1
        )
        gender_radio = gr.Radio(
            choices=["Female", "Male"],
            value="Female",
            label="Gender",
            interactive=True
        )

    with gr.Row():
        T_dropdown = gr.Dropdown(
            choices=[
                ("T1a (≤1 cm, confined to the thyroid)", "0"),
                ("T1b (>1 cm and ≤2 cm, confined to the thyroid)", "1"),
                ("T2 (>2 cm and ≤4 cm, confined to the thyroid)", "2"),
                ("T3a (>4 cm, confined to the thyroid)", "3"),
                ("T3b (Minimal extrathyroidal extension)", "4"),
                ("T4a (Moderate extrathyroidal extension, operable)", "5"),
                ("T4b (Extensive extrathyroidal extension, inoperable)", "6")
            ],
            value="0",
            label="T (Tumor Size)",
            interactive=True
        )

    with gr.Row():
        N_dropdown = gr.Dropdown(
            choices=[
                ("N0 (No spread to nearby lymph nodes)", "0"),
                ("N1a (Spread to lymph nodes in the neck close to the thyroid)", "1"),
                ("N1b (Spread to lymph nodes in the neck farther from the thyroid or upper chest)", "2")
            ],
            value="0",
            label="N (Lymph Node Spread)",
            interactive=True
        )

    with gr.Row():
        focality_dropdown = gr.Dropdown(
            choices=[
                ("Uni-focal (Single focus of thyroid cancer)", "1"),
                ("Multi-focal (Multiple foci of thyroid cancer)", "0")
            ],
            value="1",
            label="Focality",
            interactive=True
        )

    with gr.Row():
        response_dropdown = gr.Dropdown(
            choices=[
                (
                    "✅ Excellent Response - Negative imaging studies and Tg < 0.2 ng/mL "
                    "or stimulated Tg < 1 ng/mL",
                    "0"
                ),
                (
                    "❓ Indeterminate Response - Nonspecific findings; Tg potentially low",
                    "1"
                ),
                (
                    "⚠️ Biochemical Incomplete - Tg > 1 ng/mL or rising anti-Tg "
                    "antibody levels",
                    "2"
                ),
                (
                    "❌ Structural Incomplete - Identifiable structural disease on imaging",
                    "3"
                )
            ],
            value="0",
            label="Response",
            interactive=True
        )

    predict_button = gr.Button(value="Predict", variant="primary")
    prediction_output = gr.HTML(label="Prediction Results")

    # First clear the output, then make a prediction
    predict_button.click(fn=clear_md, outputs=prediction_output)
    predict_button.click(
        fn=predict_cancer,
        inputs=[
            age_slider,
            gender_radio,
            T_dropdown,
            N_dropdown,
            focality_dropdown,
            response_dropdown
        ],
        outputs=prediction_output
    )

demo.launch()