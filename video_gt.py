from video_player import attach_video_player_to_figure
import matplotlib.pyplot as plt
import numpy as np

# (timestamp, value) pairs

# sample: big bunny scene cuts
fancy_data = [
    (0, 1),
    (11.875, 1),
    (11.917, 2),
    (15.75, 2),
    (15.792, 3),
    (23.042, 3),
    (23.083, 4),
    (47.708, 4),
    (47.75, 5),
    (56.083, 5),
    (56.125, 6),
    (60, 6)
]


gt_path = "frame_label/gt/Assault010_x264_pred.npy"
predictions_path = "frame_label/Assault010_x264_pred.npy"
video_path = "videos/Assault010_x264.mp4"

anomaly_gt = []
prediction = []
gt = np.load(gt_path)
predictions = np.load(predictions_path)
predictions = np.repeat(predictions, 16)
# Iterate through each element (assuming gt_data is frame-based)
for frame_num, gt_value in enumerate(gt):
    timestamp = frame_num / 30  # Calculate timestamp based on frame number and FPS
    anomaly_gt.append((timestamp, gt_value))

for frame_num, pred_value in enumerate(predictions):
    timestamp = frame_num / 30  # Calculate timestamp based on frame number and FPS
    prediction.append((timestamp, pred_value))

def on_frame(video_timestamp, line):
    timestamps_anomaly, y_anomaly = zip(*anomaly_gt)
    x_anomaly = [timestamp - video_timestamp for timestamp in timestamps_anomaly]

    timestamps_prediction, y_prediction = zip(*prediction)
    x_prediction = [timestamp - video_timestamp for timestamp in timestamps_prediction]

    # Combine data with markers (example: 'o' for anomaly, 's' for prediction)
    # all_x = np.concatenate((x_anomaly, x_prediction))
    # all_y = np.concatenate((y_anomaly, y_prediction))

    # line.set_data(all_x, all_y)
     # Set data for anomaly points
     # Clear previous scatter points
    ax.collections.clear()

    # line.set_data(x_anomaly, y_anomaly)
    # Plot anomaly points
    ax.scatter(x_anomaly, y_anomaly, color='blue', marker='o', s=10) 

    # Plot prediction points
    ax.scatter(x_prediction, y_prediction, color='red', marker='s', s=10)
    line.axes.relim()
    line.axes.autoscale_view()

    # Plot prediction points separately
    # ax.scatter(x_prediction, y_prediction, color='red', marker='s')

    
    line.axes.figure.canvas.draw()




    
fig, ax = plt.subplots()
plt.xlim(-5, 5)
plt.axvline(x=0, color='k', linestyle='--')


line, = ax.plot([], [], color='blue')
attach_video_player_to_figure(fig, video_path, on_frame, line=line)



plt.show()


# fig, ax = plt.subplots()
# plt.xlim(-15, 15)
# plt.axvline(x=0, color='k', linestyle='--')

# line, = ax.plot([], [], color='blue')

# attach_video_player_to_figure(fig, "BigBuckBunny.mp4", on_frame2, line=line)

plt.show()

