import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(folder1, folder2, name, label1, label2):
  
  file_path1 = os.path.join(folder1, name + "pred.npy")
  file_path2 = os.path.join(folder2, name + "010_i3d_all_pred.npy")
  file_path3 = os.path.join(folder2, name + "10_i3d_pred.npy")


  label2_new = label2 + "total frames"
  label3 = label2 + "acumulated frames"

  data1 = np.load(file_path1)
  data2_temp = np.load(file_path2)
  data2_temp = np.repeat(data2_temp, 16)

  data3_temp = np.load(file_path3)
  data3_temp = np.repeat(data3_temp, 16)

  # search filename in txt lines
  # with open("G:/XONI MASTER/1 interships/UR-DMU/feature_extract/annotations/crop_videos.txt") as f:
  #   lines = f.readlines()
  #   for line in lines:
  #     if name in line:
  #       init_frame = int(line.split(" ")[1])
        # end_frame = int(line.split(" ")[2])
  init_frame = 1
  # add zeros from 0 to init_frame and end_frame to the end of the gt size
  # HAY 8 FRAMES FIJOS DE DIFERENCIA ENTRE EL GT Y EL PREDICTED
  # data2 = np.zeros(len(data1))
  data2 = np.zeros(len(data1))
  end_frame = init_frame + len(data2_temp)
  data2[init_frame-1:end_frame-1] = data2_temp

  data3 = np.zeros(len(data1))
  end_frame = init_frame + len(data3_temp)
  data3[init_frame-1:end_frame-1] = data3_temp

  plt.plot(data1, label=label1)
  plt.plot(data2, label=label2_new)
  plt.plot(data3, label=label3)
    
  plt.xlabel("Time (or Index)")
  plt.ylabel("Value")
  plt.title("Comparison of " + label1 + " and " + label2 + " for " + name)
  plt.ylim(0, 1)

  plt.legend()
 
  plt.show()
# not bad: Shooting037, RoadAccidents002, Explosion025, RoadAccidents010

folder1 = "frame_label/gt"  
folder2 = "frame_label/compare"
folder3 = "frame_label/compare"
name = "RoadAccidents002_x264_"
label1 = "original gt"
label2 = "predicted value "

plot_npy_files(folder1, folder2, name, label1, label2)
