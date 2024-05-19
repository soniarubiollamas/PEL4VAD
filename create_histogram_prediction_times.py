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
predictions = []

for row in worksheet.iter_rows(min_row=2):  # Skip header row
  prediction = row[3].value
  duration = row[5].value  
  if duration is not None and prediction is not None:
    durations.append(duration)
    predictions.append(prediction)

# Calculate the range for durations
min_duration = min(durations)
max_duration = max(durations)
print(min_duration)
print(max_duration)
# durations.remove(max_duration)
# second_max_duration = max(durations)

# Create a histogram with specified bins based on min/max
# bins = [min_duration + (i * (200 - min_duration) / (num_bins - 1)) for i in range(num_bins)]
# bins = np.linspace(min_duration,max_duration,num_bins+1).astype(int)
# breakpoint()
bins = [0, 10, 20, 30, 40, 100, 200, 300, 400, 500, 1000, 2000]
# bins = [500, 1000, 2000, 3000, 4000]
num_bins = len(bins)-1



# breakpoint()
# Annotations for bar heights
# counts, bins, patches = plt.hist(durations, bins=bins, edgecolor='black')  # Get bar details
# plt.bar_label(patches)  # Add labels using matplotlib function

# Initialize lists to store mean and standard deviation of predictions
bin_means = []
bin_stds = []

# Calculate mean and standar deviation for each bin
for i in range(num_bins):
  bin_min = bins[i]
  bin_max = bins[i+1]
  bin_indices = [index for index, duration in enumerate(durations) if bin_min <= duration < bin_max]

  if bin_indices:
    bin_predictions = [predictions[index] for index in bin_indices]
    bin_mean = np.mean(bin_predictions)
    bin_std = np.std(bin_predictions)
  else:
    bin_mean = 0
    bin_std = 0
  
  bin_means.append(bin_mean)
  bin_stds.append(bin_std)


# Calculate equal spacing for bin centers
bin_positions = np.arange(num_bins)
# Plot the histogram with error bars
plt.figure(figsize=(10, 6))
bars = plt.bar(bin_positions, bin_means, width=0.8, yerr=bin_stds, align='center', alpha=0.7, ecolor='black', capsize=10)

# Annotate mean and std on top of each bar
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + bin_stds[i], f'{bin_means[i]:.2f}\n±{bin_stds[i]:.2f}', ha='center', va='bottom')


plt.xlabel("Video Duration (seconds)")
plt.ylabel("Mean Prediction Time")
plt.title("Mean Prediction Time vs Video Duration")
plt.xticks(bin_positions, [f"{bins[i]}-{bins[i+1]}" for i in range(num_bins)], rotation=45)
plt.grid(axis='y')
plt.show()