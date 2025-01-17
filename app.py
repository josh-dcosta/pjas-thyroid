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
    'AdaBoost': 96,
    'Logistic Regression': 94,
    'Gaussian Naive Bayes': 94,
    'Gradient Boosting': 97,
    'Support Vector Machine': 94,
    'Decision Tree': 98,
    'K-Nearest Neighbors': 95,
    'Random Forest': 98,
    'XGBoost': 97
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

def predict_cancer(age, gender, response):
    """
    Generates a Markdown table with emoji icons for each model's prediction.
    """
    # 1. Load your pre-fitted scaler from disk
    scaler_file = "model/pjas-thyroid-scaler.pkl"
    if not os.path.exists(scaler_file):
        return "Error: Scaler file not found. Please check your path or name."

    scaler = joblib.load(scaler_file)

    # Simple encodings for demonstration
    gender_val = 0 if gender == "Female" else 1
    response_val = int(response)
    
    # Create a DataFrame for the features
    features = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_val],
        'Response': [response_val]
    })

    # Remove any NaN
    features = features.dropna()

    # Scale the 'Age' column
    features[['Age']] = scaler.transform(features[['Age']])
    
    # Sort models by accuracy (descending)
    sorted_model_names = sorted(
        model_paths.keys(),
        key=lambda m: model_accuracies[m],
        reverse=True
    )

    # We'll build a Markdown table
    table_header = (
        "| **Model**                   | **Accuracy** | **Prediction**                |\n"
        "|-----------------------------|--------------|--------------------------------|\n"
    )
    table_rows = []

    # Emojis for predictions
    can_recur_emoji = "🔴"  # "Cancer can recur"
    cannot_recur_emoji = "🟢"  # "Cancer cannot-recur"

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

        # Convert numeric prediction to icon + text
        if pred_value == 1:
            pred_text = f"{can_recur_emoji} Cancer can recur"
        else:
            pred_text = f"{cannot_recur_emoji} Cancer cannot-recur"

        accuracy = model_accuracies.get(model_name, "N/A")
        row = f"| {model_name} | {accuracy}% | {pred_text} |"
        table_rows.append(row)

    # Combine into a single Markdown table
    md_table = table_header + "\n".join(table_rows)
    return md_table

def clear_md():
    """Clears the Markdown output."""
    return ""

def show_description(response_value):
    """
    Return the detailed Markdown text for the selected response.
    """
    return response_details.get(response_value, "No description available.")


with gr.Blocks() as demo:
    gr.Markdown("# Thyroid Cancer Recurrence Predictor")

    age_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Age", value=44)
    gender_radio = gr.Radio(choices=["Female", "Male"], value="Female", label="Gender")
    response_dropdown = gr.Dropdown(
        choices=[
            ("✅ Excellent Response", "0"),
            ("❓ Indeterminate Response", "1"),
            ("⚠️ Biochemical Incomplete", "2"),
            ("❌ Structural Incomplete", "3")
        ],
        value="0",  # Default selection
        label="Response"
    )

    description_output = gr.Markdown(
        value=show_description("0"),
        label="Details"
    )
    response_dropdown.change(
        fn=show_description,
        inputs=response_dropdown,
        outputs=description_output
    )

    # Note: busy_value sets the text on the button while the function is running
    predict_button = gr.Button(
        value="Predict",
        variant="primary"
    )

    prediction_output = gr.Markdown(label="Prediction Results")

    predict_button.click(fn=clear_md, outputs=prediction_output)
    predict_button.click(
        fn=predict_cancer,
        inputs=[age_slider, gender_radio, response_dropdown],
        outputs=prediction_output
    )


demo.launch()