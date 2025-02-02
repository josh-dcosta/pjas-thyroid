import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

# --------------------------------------
# PART 1: Load CSV Data and Generate Graphs
# --------------------------------------

# File paths for the 13-feature and 6-feature reports
csv_13 = "report/model_summary_report_13.csv"
csv_6  = "report/model_summary_report_6_smote.csv"

# Load the CSVs into dataframes
if os.path.exists(csv_13) and os.path.exists(csv_6):
    df13 = pd.read_csv(csv_13)
    df6 = pd.read_csv(csv_6)
else:
    raise FileNotFoundError("One or both CSV files were not found. Check the file paths.")

# --- Assume both dataframes have these columns:
# "Model Name", "Test Accuracy", "Model Sensitivity", "Model Specificity"
# (If your CSVs use different names, adjust accordingly.)

# Merge the dataframes on "Model Name" with suffixes for clarity
merged_df = pd.merge(
    df13[['Model Name', 'Test Accuracy', 'Model Sensitivity', 'Model Specificity']],
    df6[['Model Name', 'Test Accuracy', 'Model Sensitivity', 'Model Specificity']],
    on='Model Name', suffixes=('_13', '_6')
)

# Function to create a grouped bar chart comparing a given metric
def create_grouped_bar_chart(df, metric, ylabel, title, output_path):
    # df is merged_df with columns: Model Name, {metric}_13, {metric}_6
    models = df['Model Name']
    values_13 = df[f'{metric}_13']
    values_6 = df[f'{metric}_6']
    
    x = range(len(models))
    width = 0.35  # width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], values_13, width=width, color='orange', label='13 Features')
    plt.bar([i + width/2 for i in x], values_6, width=width, color='green', label='6 Features')
    
    plt.xlabel("Model Name", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Create output folder for graphs if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Generate grouped bar charts for Sensitivity, Specificity, and Accuracy
sensitivity_chart = "graphs/sensitivity_comparison.png"
specificity_chart = "graphs/specificity_comparison.png"
accuracy_chart    = "graphs/accuracy_comparison.png"

create_grouped_bar_chart(merged_df, "Model Sensitivity", "Sensitivity (%)",
                           "Comparison of Model Sensitivity", sensitivity_chart)
create_grouped_bar_chart(merged_df, "Model Specificity", "Specificity (%)",
                           "Comparison of Model Specificity", specificity_chart)
create_grouped_bar_chart(merged_df, "Test Accuracy", "Accuracy (%)",
                           "Comparison of Test Accuracy", accuracy_chart)

# --------------------------------------
# PART 2: Create the PowerPoint Presentation
# --------------------------------------

# Create a presentation object
prs = Presentation()

# Define slide layouts
title_slide_layout = prs.slide_layouts[0]
bullet_slide_layout = prs.slide_layouts[1]
blank_slide_layout = prs.slide_layouts[6]

# --- Slide 1: Title ---
slide = prs.slides.add_slide(title_slide_layout)
slide.shapes.title.text = "PREDICTING RECURRENCE OF DIFFERENTIAL THYROID CANCER"
slide.placeholders[1].text = "A Data-Driven Approach Using AI and Machine Learning\nBy [Your Name]"

# --- Slide 2: Research ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Research"
text = (
    "• The thyroid is a butterfly-shaped gland that produces hormones regulating metabolism.\n"
    "• Differential thyroid cancer (e.g., papillary, follicular) is common but can recur.\n"
    "• Early prediction of recurrence improves patient outcomes and reduces treatment costs.\n"
    "• Prior research used 13 features; our study explores using only 6 key features for improved privacy and efficiency."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 3: Problem / Purpose / Question ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Problem / Purpose / Question"
text = (
    "Problem: Recurrence of thyroid cancer is unpredictable, complicating patient management.\n\n"
    "Purpose: Develop a predictive model using fewer features.\n\n"
    "Question: Can a model using only 6 key features achieve high accuracy, sensitivity, and specificity compared to 13-feature models?"
)
slide.placeholders[1].text_frame.text = text

# --- Slide 4: Hypothesis ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Hypothesis"
text = (
    "If a machine learning model is built using the 6 most important features (Age, Gender, T, N, Focality, Response),\n"
    "then it will predict thyroid cancer recurrence with performance comparable to models using 13 features."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 5: Materials ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Materials"
text = (
    "• Data: 383 patient records (UCI Repository, 10+ years follow-up)\n"
    "• Software: Python, Jupyter Notebook, scikit-learn, imblearn, joblib\n"
    "• Tools: pandas, numpy, matplotlib, seaborn, plotly\n"
    "• Hardware: Standard computer\n"
    "• Gradio App: For real-time clinical predictions\n"
    "• References: Original study (Springer DOI:10.1007/s00405-023-08299-w)"
)
slide.placeholders[1].text_frame.text = text

# --- Slide 6: Procedure ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Procedure"
text = (
    "1. Data Acquisition: Load CSV data from UCI Repository.\n"
    "2. Preprocessing: Label encoding, one-hot encoding, and scaling.\n"
    "3. Feature Selection: Reduce from 13 to 6 key features based on importance.\n"
    "4. Balance Data: Apply SMOTE to address class imbalance.\n"
    "5. Model Training: Train various ML models using the 6 selected features.\n"
    "6. Evaluation: Assess models using accuracy, sensitivity, specificity, and AUC.\n"
    "7. Deployment: Export models and create a Gradio app for real-time prediction."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 7: Data Collected ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Data Collected"
text = (
    "• 383 patient records with clinicopathologic features.\n"
    "• Selected features: Age, Gender, T (Tumor Size), N (Lymph Node Spread), Focality, Response.\n"
    "• Outcome: Recurrence (Yes/No).\n\n"
    "Sample Table:\n"
    "Age | Gender | T | N | Focality | Response | Recurred\n"
    "0.27 | 0 | 0 | 0 | 1 | 1 | 0\n"
    "0.34 | 0 | 0 | 0 | 1 | 0 | 0\n"
    "..."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 8: Graphs of Results ---
slide = prs.slides.add_slide(blank_slide_layout)
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(1))
tf = title_box.text_frame
p = tf.add_paragraph()
p.text = "Graphs of Results"
p.font.size = Pt(28)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

# Insert previously generated graphs
# Position and size for the grouped bar charts
left = Inches(0.5)
top = Inches(1.5)
width = Inches(4)
height = Inches(3)

# Insert Sensitivity Comparison Chart (grouped bar)
slide.shapes.add_picture(sensitivity_chart, left, top, width=width, height=height)
# Insert Specificity Comparison Chart
slide.shapes.add_picture(specificity_chart, left+Inches(4.5), top, width=width, height=height)
# Insert Accuracy Comparison Chart
slide.shapes.add_picture(accuracy_chart, left, top+Inches(3.5), width=width, height=height)

# --- Slide 9: Analysis ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Analysis"
text = (
    "• Data Distribution: Majority of patients are female and mostly low risk.\n"
    "• Imbalance: Recurrence outcome is skewed towards non-recurrence.\n"
    "• Model Comparison:\n"
    "   - 13-feature models achieved very high sensitivity and AUC.\n"
    "   - Our 6-feature models (e.g., XGBoost, Random Forest) maintained test accuracy >98%.\n"
    "   - Some models (e.g., SVM) underperformed with fewer features.\n"
    "• Conclusion: Careful feature selection preserves high performance while reducing cost and protecting privacy."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 10: Conclusion ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Conclusion"
text = (
    "• Thyroid cancer recurrence can be accurately predicted using only 6 key features.\n"
    "• Benefits: Enhanced privacy, lower training costs, reduced energy consumption, and minimized overfitting.\n"
    "• Supports efficient, ethical, real-time clinical decision making."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 11: Weaknesses / Improvements ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Weaknesses / Improvements"
text = (
    "Weaknesses:\n"
    "• Heavy imbalance in recurrence outcomes.\n"
    "• Reduced features may omit subtle clinical nuances.\n"
    "• Some models (e.g., SVM) underperform with fewer features.\n\n"
    "Improvements:\n"
    "• Incorporate additional data or external validation.\n"
    "• Further feature engineering and model tuning.\n"
    "• Expand clinical integration and test on diverse populations."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 12: Future Studies / Research ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Future Studies / Research"
text = (
    "• Expand dataset and include multi-modal data (e.g., genetic, imaging).\n"
    "• Integrate model with electronic health records for real-time support.\n"
    "• Explore advanced ensemble and deep learning methods.\n"
    "• Conduct prospective clinical trials for external validation."
)
slide.placeholders[1].text_frame.text = text

# --- Slide 13: Works Cited ---
slide = prs.slides.add_slide(bullet_slide_layout)
slide.shapes.title.text = "Works Cited"
text = (
    "1. Borzooei, S. & Tarokhian, A. (2023). Differentiated Thyroid Cancer Recurrence [Dataset].\n"
    "   UCI Repository. https://doi.org/10.24432/C5632J\n\n"
    "2. Borzooei, S., Briganti, G., Golparian, M., Lechien, J.R., & Tarokhian, A. (2023).\n"
    "   Machine learning for risk stratification of thyroid cancer patients: a 15‐year cohort study.\n"
    "   Head and Neck. https://link.springer.com/article/10.1007/s00405-023-08299-w\n\n"
    "3. Additional References:\n"
    "   - PMC: Predictive factors for recurrence of papillary thyroid carcinoma\n"
    "   - Nature: Current practice in patients with differentiated thyroid cancer\n"
    "   - Frontiers in Oncology: AI for Personalized Medicine in Thyroid Cancer\n"
    "   - Tandfonline: Missing Data Preprocessing in Credit Classification"
)
slide.placeholders[1].text_frame.text = text

# Save the presentation
output_pptx = "presentation/Thyroid_Cancer_Recurrence_Presentation.pptx"
prs.save(output_pptx)
print(f"Presentation saved as '{output_pptx}'")