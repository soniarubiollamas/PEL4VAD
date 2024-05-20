import os
import numpy as np
from openpyxl import Workbook


def compare_file_sizes(folder1, folder2, output_file):

  wb = Workbook()
  ws = wb.active
  ws.append(["Filename", f"Original GT size", f"Predicted size"])  # Header row

  for filename in os.listdir(folder1):
    if filename.endswith(".npy"):
      file_path1 = os.path.join(folder1, filename)
      data1 = np.load(file_path1)
      data1shape = data1.shape[0]

      # Check if the file exists in the second folder
      file_path2 = os.path.join(folder2, filename)
      if os.path.exists(file_path2):
        data2 = np.load(file_path2)
        data2 = np.repeat(data2, 16)
        data2shape = data2.shape[0]

        ws.append([filename, data1shape, data2shape])

  wb.save(output_file)


# Example usage
folder1 = "frame_label/gt"
folder2 = "frame_label"
output_file = "file_size_comparison.xlsx"

compare_file_sizes(folder1, folder2, output_file)

print(f"File size comparison saved to: {output_file}")
