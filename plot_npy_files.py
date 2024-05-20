import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(folder1, folder2, filename, label1, label2):
  
  file_path1 = os.path.join(folder1, filename)
  file_path2 = os.path.join(folder2, filename)

  data1 = np.load(file_path1)
  data2 = np.load(file_path2)
  data2 = np.repeat(data2, 16)

  plt.plot(data1, label=label1)
  plt.plot(data2, label=label2)


  plt.xlabel("Time (or Index)")
  plt.ylabel("Value")
  plt.title("Comparison of " + label1 + " and " + label2 + " for " + filename)

  plt.legend()

  plt.show()

folder1 = "frame_label/gt"  
folder2 = "frame_label"
filename = "Normal_Videos_033_x264_pred.npy"
label1 = "original gt"
label2 = "predicted gt"

plot_npy_files(folder1, folder2, filename, label1, label2)
