import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(folder1, folder2, folder3, filename, filename_opt, label1, label2, label3):
  
  file_path1 = os.path.join(folder1, filename)
  file_path2 = os.path.join(folder2, filename)
  file_path3 = os.path.join(folder3, filename_opt)

  data1 = np.load(file_path1)
  data2 = np.load(file_path2)
  data2 = np.repeat(data2, 16)

  data3 = np.load(file_path3)
  data3 = np.repeat(data3, 16)

  plt.plot(data1, label=label1)
  plt.plot(data2, label=label2)
  plt.plot(data3, label=label3)


  plt.xlabel("Time (or Index)")
  plt.ylabel("Value")
  plt.title("Comparison of " + label1 + ", " + label2 + " and " + label3 + " for " + filename)
  plt.ylim(0, 1)

  plt.legend()

  plt.show()
# not bad: Shooting037, RoadAccidents002, Explosion025, RoadAccidents010
# largos: Vandalism028, Stealing058 (muy larga anomalia)

folder1 = "frame_label/gt"  
folder2 = "frame_label/all_good"
folder3 = "frame_label/no_OPT"
name = "RoadAccidents021"
filename = f"{name}_x264_pred.npy"
filename_opt = f"{name}_x264_pred_no_OPT.npy"
label1 = "original gt"
label2 = "predicted value"
label3 = "predicted value with no OPT"

plot_npy_files(folder1, folder2, folder3, filename, filename_opt, label1, label2, label3)
