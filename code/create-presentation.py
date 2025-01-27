import pandas as pd
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches
import os
import numpy as np

# Load your data from CSV files
data_17_features = pd.read_csv('report/model_summary_report_17_v0.csv')  # Replace with your file path
data_6_smote = pd.read_csv('report/model_summary_report_6_smote.csv')  # Replace with your file path


# Sort the data by 'Model Name'
data_17_features = data_17_features.sort_values(by='Model Name')
data_6_smote = data_6_smote.sort_values(by='Model Name')

# Metrics for comparison
metrics = ['Model Sensitivity', 'Model Specificity', 'Model AUC']

# Ensure the directory for saving graphs and presentations exists
os.makedirs("presentation", exist_ok=True)

# Generate and save grouped bar charts for each metric
graph_image_paths = []
for metric in metrics:
    models = data_17_features['Model Name']  # Assuming all datasets have the same models
    x = np.arange(len(models))  # Positions for the groups
    width = 0.25  # Width of each bar

    plt.figure(figsize=(12, 6))

    # Plot grouped bars
    plt.bar(x - width / 2, data_6_smote[metric], width, label='6 Features', color='green')
    plt.bar(x + width / 2, data_17_features[metric], width, label='17 Features', color='orange')

    # Add values on the bars
    for group, data in zip([x - width / 2, x + width / 2], [data_6_smote, data_17_features]):
        for pos, value in zip(group, data[metric]):
            plt.text(pos, value, f'{value:.2f}', ha='center', va='bottom', fontsize=6)

    # Chart details
    ylabel = metric.replace('Model ', '')
    if ylabel == "Sensitivity":
        ylabel = "Recurrence Accuracy"
    elif ylabel == "Specificity":
        ylabel = "Non-Recurrence Accuracy"

    plt.title(f'Comparison of {ylabel} Across Feature Sets', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the chart
    graph_path = f"presentation/{metric}_Comparison.png"
    plt.savefig(graph_path)
    graph_image_paths.append(graph_path)
    plt.close()

# Create PowerPoint presentation
presentation = Presentation()

# Title slide
slide = presentation.slides.add_slide(presentation.slide_layouts[0])
title = slide.shapes.title
subtitle = slide.placeholders[1]

title.text = "Comparison of Feature Sets for Predicting Thyroid Cancer Recurrence"
subtitle.text = "Analyzing 6 Features and 17 Features"

# Overview slide
slide = presentation.slides.add_slide(presentation.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Overview of Feature Sets"
content.text = (
    "1. 6 Features: Models with 6 features and Synthetic Minority Oversampling Technique to balance data.\n"
    "2. 17 Features: Models using all available clinico-pathological features."
)

# Metrics slide
slide = presentation.slides.add_slide(presentation.slide_layouts[1])
title = slide.shapes.title
content = slide.placeholders[1]

title.text = "Key Metrics Compared"
content.text = "\n".join(metrics)

# Add graphs to slides
for graph_path, metric in zip(graph_image_paths, metrics):
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = f'{metric} Comparison'
    slide.shapes.add_picture(graph_path, Inches(0.5), Inches(1.5), width=Inches(9))

# Save the presentation
presentation.save("presentation/Feature_Set_Comparison_Presentation.pptx")
print("Presentation saved as 'presentation/Feature_Set_Comparison_Presentation.pptx'")
