import os
import numpy as np
import matplotlib.pyplot as plt

def plot_npy_files(folder1, folder2, name, label1, label2):
  
  file_path1 = os.path.join(folder1, "Shoplifting021_x264_pred.npy")

  for i in range(1,11):
    percentaje = i*10
    
    file_path2 = file_path2 = os.path.join(folder2, name + str(i) + "_i3d_pred_NoNorm_NoSmoothing.npy")
    label2_new = label2 + str(percentaje) + "%"

    data1 = np.load(file_path1)
    data_temp = np.load(file_path2)
    data_temp = np.repeat(data_temp, 16)

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
    data2 = np.zeros(len(data_temp))
    end_frame = init_frame + len(data_temp)
    data2[init_frame-1:end_frame-1] = data_temp

    
    plt.plot(data2, label=label2_new) # label=label2_new
    
  plt.xlabel("Frames")
  plt.ylabel("Value")
  plt.title("Comparison of " + label1 + " and " + label2 + " for " + name)
  plt.ylim(0, 1)

  plt.legend()
  plt.plot(data1, label=label1)
  plt.show()
# not bad: Shooting037, RoadAccidents002, Explosion025, RoadAccidents010

folder1 = "frame_label/gt"  
folder2 = "frame_label/all_good"
name = "Shoplifting021_x264_"
label1 = "original gt"
label2 = "pred"

plot_npy_files(folder1, folder2, name, label1, label2)
