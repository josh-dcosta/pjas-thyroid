import gradio as gr
import joblib
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Dictionary containing the model names and corresponding pickle file names
model_paths = {
    #'AdaBoost': 'pjas-thyroid-AdaBoost.pkl',
    'Decision Tree': 'pjas-thyroid-Decision Tree.pkl',
    #'Gaussian Naive Bayes': 'pjas-thyroid-Gaussian Naive Bayes.pkl',
    'Gradient Boosting': 'pjas-thyroid-Gradient Boosting.pkl',
    #'K-Nearest Neighbors': 'pjas-thyroid-K-Nearest Neighbors.pkl',
    #'Logistic Regression': 'pjas-thyroid-Logistic Regression.pkl',
    'Random Forest': 'pjas-thyroid-Random Forest.pkl',
    #'Support Vector Machine': 'pjas-thyroid-Support Vector Machine.pkl',
    #'XGBoost': 'pjas-thyroid-XGBoost.pkl'
}

# Example: Hard-coded dictionary of accuracies per model.
# Replace these with the actual accuracies (integers or floats) you want to display.
model_accuracies = {
    # 'AdaBoost': 100,
    # 'Logistic Regression': 97,
    # 'Gaussian Naive Bayes': 97,
    'Gradient Boosting': 98,
    # 'Support Vector Machine': 97,
    'Decision Tree': 97,
    # 'K-Nearest Neighbors': 96,
    'Random Forest': 97 #,'XGBoost': 96
}

# We assume:
#   0 -> Cancer cannot recur
#   1 -> Cancer can recur
#
# In your real code, make sure the model predicts with the correct shape/transform
# e.g., model.predict([[Age, Gender_encoded, Response_encoded]]) 
# and encode gender/response properly. 
# For demonstration, let's do a simplistic approach with minimal transformations.

# Data Wrangler
# import pandas as pd
# df = df[['Row', 'Age', 'Gender', 'Response', 'Recurred']]
# df = df[df["Recurred"] == "Yes"]
# df = df[df["Model Name"] == "Gradient Boosting"]

def predict_cancer(age, gender, response):
    """
    Generates a Markdown table with emoji icons for each model's prediction.
    """
    scaler = MinMaxScaler()
    age_scaled = scaler.fit_transform([[age]])

    # Simple encodings for demonstration
    gender_val = 0 if gender == "Female" else 1
    response_val = int(response)
    features = pd.DataFrame(
    [[age_scaled, gender_val, response_val]], 
    columns=["Age", "Gender", "Response"]
    )
    
    # Sort the model names based on accuracy (descending order)
    sorted_model_names = sorted(model_paths.keys(), key=lambda m: model_accuracies[m], reverse=True)

    # We'll build a Markdown table
    table_header = (
        "| **Model**                   | **Accuracy** | **Prediction**                |\n"
        "|-----------------------------|--------------|--------------------------------|\n"
    )
    table_rows = []

    # Emojis for predictions
    can_recur_emoji = "🔴"      # "Cancer can recur"
    cannot_recur_emoji = "🟢"   # "Cancer cannot-recur"

    for model_name in sorted_model_names:
        pickle_file = model_paths[model_name]
        model_file_path = os.path.join("model", pickle_file) 

        # If the file doesn't exist, show an error
        if not os.path.exists(model_file_path):
            row = f"| {model_name} | N/A | **Error**: file not found |"
            table_rows.append(row)
            continue

        loaded_obj = joblib.load(model_file_path)
        model = loaded_obj

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

with gr.Blocks() as demo:
    gr.Markdown("# Thyroid Cancer Recurrence Predictor")

    age_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Age", value=44)
    gender_radio = gr.Radio(choices=["Female", "Male"], value="Female", label="Gender")
    response_dropdown = gr.Dropdown(
        choices=[
            (
                "✅ Excellent Response: Negative imaging studies and suppressed thyroglobulin levels (below 0.2 ng/mL or stimulated Tg below 1 ng/mL).", 
                "0"
            ),
            (
                "❓ Indeterminate Response: Nonspecific imaging findings, making it difficult to classify as benign or malignant, with potentially low thyroglobulin levels.", 
                "1"
            ),
            (
                "⚠️ Biochemical Incomplete: Negative imaging but elevated thyroglobulin levels (suppressed Tg > 1 ng/mL or stimulated Tg > 10 ng/mL) or rising anti-Tg antibody levels.", 
                "2"
            ),
            (
                "❌ Structural Incomplete: Presence of identifiable structural disease on imaging, regardless of thyroglobulin level.", 
                "3"
            )
        ],
        value="0",  # default to 'Excellent' if desired
        label="Response"
    )

    predict_button = gr.Button("Predict")

    # Use gr.Markdown to display the table and emojis
    prediction_output = gr.Markdown(label="Prediction Results")

    predict_button.click(fn=clear_md, outputs=prediction_output)
    predict_button.click(
        fn=predict_cancer,
        inputs=[age_slider, gender_radio, response_dropdown],
        outputs=prediction_output
    )

demo.launch()
