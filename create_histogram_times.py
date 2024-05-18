import openpyxl
from openpyxl.chart import BarChart, Reference
import matplotlib.pyplot as plt

# Load the Excel file
excel_file = "annotations\histogram_times.xlsx"
workbook = openpyxl.load_workbook(excel_file)
worksheet = workbook.active

# Collect durations
durations = []
for row in worksheet.iter_rows(min_row=2):  # Skip header row
  duration = row[1].value  # Assuming duration is in the second column (index 1)
  if duration:  # Skip empty cells
    durations.append(duration)
# breakpoint()
min_duration = min(durations)
max_duration = max(durations)
#second max duration
durations.remove(max_duration)
second_max_duration = max(durations)
# breakpoint()

# Create a histogram with specified bins based on min/max
num_bins = 10
bins = [min_duration + (i * (200 - min_duration) / (num_bins - 1)) for i in range(num_bins)]
plt.hist(durations, bins=bins)

# Annotations for bar heights
counts, bins, patches = plt.hist(durations, bins=bins, edgecolor='black')  # Get bar details
plt.bar_label(patches)  # Add labels using matplotlib function

# Labels and Title
plt.xlabel("Video Duration (seconds)")
plt.ylabel("Number of videos")
plt.title("Distribution of Video Durations in " + excel_file)
plt.grid(axis='y')
plt.show()