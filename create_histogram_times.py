import openpyxl
from openpyxl.chart import BarChart, Reference
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
excel_file = "annotations/time_prediction.xlsx"
workbook = openpyxl.load_workbook(excel_file)
worksheet = workbook.active

# Collect durations
durations = []
for row in worksheet.iter_rows(min_row=2):  # Skip header row
  duration = row[5].value  # Assuming duration is in the second column (index 1)
  if duration:  # Skip empty cells
    durations.append(duration)

# Create a histogram with specified bins based on min/max
bins = [0, 10, 20, 30, 40, 100, 200, 300, 400, 500, 1000, 2000, 3601]
num_bins = len(bins)-1

# Calculate the histogram data
counts, _ = np.histogram(durations, bins=bins)

# Calculate positions for the bars
bin_positions = np.arange(num_bins)

# Plot the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(bin_positions, counts, width=0.8, align='center', alpha=0.7, edgecolor='black')

# Annotate bar heights
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

# Set x-ticks to show the bins ranges
plt.xticks(bin_positions, [f"{bins[i]}-{bins[i+1]}" for i in range(num_bins)], rotation=45)

# Labels and Title
plt.xlabel("Video Duration (seconds)")
plt.ylabel("Number of videos")
plt.title("Distribution of Video Durations in " + excel_file)
plt.grid(axis='y')
plt.show()