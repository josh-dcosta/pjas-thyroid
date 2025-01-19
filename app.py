import gradio as gr
import joblib
import os
import pandas as pd

# Dictionary containing the model names and corresponding pickle file names
model_paths = {
    'AdaBoost': 'pjas-thyroid-AdaBoost.pkl',
    'Decision Tree': 'pjas-thyroid-Decision Tree.pkl',
    'Gaussian Naive Bayes': 'pjas-thyroid-Gaussian Naive Bayes.pkl',
    'Gradient Boosting': 'pjas-thyroid-Gradient Boosting.pkl',
    'K-Nearest Neighbors': 'pjas-thyroid-K-Nearest Neighbors.pkl',
    'Logistic Regression': 'pjas-thyroid-Logistic Regression.pkl',
    'Random Forest': 'pjas-thyroid-Random Forest.pkl',
    'Support Vector Machine': 'pjas-thyroid-Support Vector Machine.pkl',
    'XGBoost': 'pjas-thyroid-XGBoost.pkl'
}

# Example: Hard-coded dictionary of accuracies per model.
model_accuracies = {
    'AdaBoost': 97.13,
    'Logistic Regression': 96.87,
    'Gaussian Naive Bayes': 95.3,
    'Gradient Boosting': 98.96,
    'Support Vector Machine': 96.34,
    'Decision Tree': 97.91,
    'K-Nearest Neighbors': 97.13,
    'Random Forest': 98.96,
    'XGBoost': 98.43
}

# We assume:
#   0 -> Cancer cannot recur
#   1 -> Cancer can recur

response_details = {
    "0": """**Excellent Response**  
Negative imaging studies and suppressed thyroglobulin levels  
(below 0.2 ng/mL or stimulated Tg below 1 ng/mL).""",
    "1": """**Indeterminate Response**  
Nonspecific findings on imaging studies, making it difficult  
to confidently classify as benign or malignant,  
with potentially low thyroglobulin levels.""",
    "2": """**Biochemical Incomplete**  
Negative imaging but elevated thyroglobulin levels  
(suppressed Tg above 1 ng/mL or stimulated Tg above 10 ng/mL)  
or rising anti-Tg antibody levels.""",
    "3": """**Structural Incomplete**  
Presence of identifiable structural disease on imaging,  
regardless of thyroglobulin level."""
}

def predict_cancer(age, gender, response, tumor_size, lymph_node_spread, focality):
    """
    Generates a Markdown table with emoji icons for each model's prediction,
    including 'TumorSize', 'LymphNodeSpread', and 'Focality' as additional features.
    """
    # 1. Load your pre-fitted scaler from disk
    scaler_file = "model/pjas-thyroid-scaler.pkl"
    if not os.path.exists(scaler_file):
        return "Error: Scaler file not found. Please check your path or name."

    scaler = joblib.load(scaler_file)

    # 2. Simple encodings for demonstration
    gender_val = 0 if gender == "Female" else 1
    response_val = int(response)
    tumor_val = int(tumor_size)          # Convert Tumor Size dropdown value to int
    lymph_val = int(lymph_node_spread)   # Convert Lymph Node Spread dropdown value to int
    focality_val = int(focality)         # Convert Focality dropdown value to int

    # 3. Create a DataFrame for the features
    features = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_val],
        'T': [tumor_val],
        'N': [lymph_val],
        'Focality': [focality_val],
        'Response': [response_val]
    })

    # 4. Remove any NaN
    features = features.dropna()

    # 5. Scale the 'Age' column
    #    If you have more features to scale, include them here accordingly
    features[['Age']] = scaler.transform(features[['Age']])

    # 6. Sort models by accuracy (descending)
    sorted_model_names = sorted(
        model_paths.keys(),
        key=lambda m: model_accuracies[m],
        reverse=True
    )

    # 7. Build a Markdown table
    table_header = (
        "| **Model**                   | **Accuracy** | **Prediction**                |\n"
        "|-----------------------------|--------------|--------------------------------|\n"
    )
    table_rows = []

    # 8. Emojis for predictions
    can_recur_emoji = "🔴"  # "Cancer can recur"
    cannot_recur_emoji = "🟢"  # "Cancer cannot-recur"

    # 9. Iterate through each model and make predictions
    for model_name in sorted_model_names:
        pickle_file = model_paths[model_name]
        model_file_path = os.path.join("model", pickle_file) 

        if not os.path.exists(model_file_path):
            row = f"| {model_name} | N/A | **Error**: file not found |"
            table_rows.append(row)
            continue

        model = joblib.load(model_file_path)
        prediction = model.predict(features)  # e.g., [0] or [1]
        pred_value = prediction[0]

        # 10. Convert numeric prediction to icon + text
        if pred_value == 1:
            pred_text = f"{can_recur_emoji} Sorry, Your Cancer can recur"
        else:
            pred_text = f"{cannot_recur_emoji} Great News! Your Cancer cannot-recur"

        accuracy = model_accuracies.get(model_name, "N/A")
        row = f"| {model_name} | {accuracy}% | {pred_text} |"
        table_rows.append(row)

    # 11. Combine into a single Markdown table
    md_table = table_header + "\n".join(table_rows)
    return md_table

def clear_md():
    """Clears the Markdown output."""
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# Thyroid Cancer Recurrence Predictor")

    # Existing inputs
    age_slider = gr.Slider(
        minimum=1,
        maximum=100,
        step=1,
        label="Age",
        value=44,
        interactive=True
    )
    
    gender_radio = gr.Radio(
        choices=["Female", "Male"],
        value="Female",
        label="Gender",
        interactive=True
    )

    # New Tumor Size dropdown with descriptive text
    tumor_size_dropdown = gr.Dropdown(
        choices=[
            ("T1a (≤1 cm, confined to the thyroid)", "0"),
            ("T1b (>1 cm and ≤2 cm, confined to the thyroid)", "1"),
            ("T2 (>2 cm and ≤4 cm, confined to the thyroid)", "2"),
            ("T3a (>4 cm, confined to the thyroid)", "3"),
            ("T3b (Minimal extrathyroidal extension)", "4"),
            ("T4a (Moderate extrathyroidal extension, operable)", "5"),
            ("T4b (Extensive extrathyroidal extension, inoperable)", "6")
        ],
        value="0",  # Default T1a
        label="Tumor Size",
        interactive=True
    )

    # New Lymph Node Spread dropdown with descriptive text
    lymph_node_dropdown = gr.Dropdown(
        choices=[
            ("N0 (No spread to nearby lymph nodes)", "0"),
            ("N1a (Spread to lymph nodes in the neck close to the thyroid)", "1"),
            ("N1b (Spread to lymph nodes in the neck farther from the thyroid or upper chest)", "2")
        ],
        value="0",  # Default N0
        label="Lymph Node Spread",
        interactive=True
    )

    # New Focality of Differential Thyroid Cancer dropdown with descriptive text
    focality_dropdown = gr.Dropdown(
        choices=[
            ("Uni-focal (Single focus of thyroid cancer)", "1"),
            ("Multi-focal (Multiple foci of thyroid cancer)", "0")
        ],
        value="1",  # Default Uni-focal
        label="Focality of Differential Thyroid Cancer",
        interactive=True
    )

    response_dropdown = gr.Dropdown(
        choices=[
            ("✅ Excellent Response - Negative imaging studies and suppressed thyroglobulin levels (below 0.2 ng/mL or stimulated Tg below 1 ng/mL)", "0"),
            ("❓ Indeterminate Response - Nonspecific findings on imaging studies with potentially low thyroglobulin levels", "1"),
            ("⚠️ Biochemical Incomplete - Negative imaging but elevated thyroglobulin levels or rising anti-Tg antibody levels", "2"),
            ("❌ Structural Incomplete - Presence of identifiable structural disease on imaging, regardless of thyroglobulin level", "3")
        ],
        value="0",  # Default
        label="Response",
        interactive=True
    )

    # Predict button
    predict_button = gr.Button(
        value="Predict",
        variant="primary"
    )
    prediction_output = gr.Markdown(label="Prediction Results")

    # Include all dropdowns as inputs for the predict function
    predict_button.click(
        fn=predict_cancer,
        inputs=[age_slider, gender_radio, response_dropdown, tumor_size_dropdown, lymph_node_dropdown, focality_dropdown],
        outputs=prediction_output
    )

demo.launch()