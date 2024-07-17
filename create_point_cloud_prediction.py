import openpyxl
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
excel_file = "annotations/time_prediction.xlsx"
workbook = openpyxl.load_workbook(excel_file)
worksheet = workbook.active

# Collect durations and predictions
durations = []
predictions = []

for row in worksheet.iter_rows(min_row=2):  # Skip header row
    prediction = row[4].value
    duration = row[5].value
    if duration is not None and prediction is not None:
        durations.append(duration)
        predictions.append(prediction)

# Calculate the range for durations (optional for point cloud)
min_duration = min(durations)
max_duration =300 #500
y_lim = max(predictions)
# print(min_duration)
# print(max_duration)

# Create the point cloud (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(durations, predictions, alpha=0.7, edgecolors='black')

# Customize plot
plt.xlabel("Video Duration (frames)")
plt.ylabel("Infer Time (seconds)")
plt.title("Infer Time vs Video Duration")
# Optional: Set axis limits based on your data (if needed)
plt.xlim(min_duration, max_duration)
plt.ylim(0, 5)

plt.grid(axis='y')
plt.show()
