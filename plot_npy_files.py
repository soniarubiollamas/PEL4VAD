import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(folder1, folder2, filename, label1, label2):
  
  file_path1 = os.path.join(folder1, filename)
  filename2 = filename.split('.')[0] + ".npy"
  file_path2 = os.path.join(folder2, filename2)

  data1 = np.load(file_path1)
  data2 = np.load(file_path2)
  data2 = np.repeat(data2, 16)

  plt.plot(data1, label=label1)
  plt.plot(data2, label=label2)


  plt.xlabel("Frames")
  plt.ylabel("Value")
  plt.title("Comparison of " + label1 + " and " + label2 + " for " + filename)
  plt.ylim(0, 1)

  plt.legend()

  plt.show()
# not bad: Shooting037, RoadAccidents002, Explosion025, RoadAccidents010
# largos: Vandalism028, Stealing058 (muy larga anomalia)

folder1 = "frame_label/gt"  
folder2 = "frame_label/24June"
name = "Burglary005"
filename = f"{name}_x264_pred.npy"
label1 = "original gt"
label2 = "predicted value"

plot_npy_files(folder1, folder2, filename, label1, label2)
