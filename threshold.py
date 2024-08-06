import numpy as np
from matplotlib import pyplot as plt
import os

def load_video_data(pred_folder, gt_folder, video_file):
    """
    Carga los datos de predicciones y ground truth para un video específico.
    """
    pred_path = os.path.join(pred_folder, video_file)
    gt_path = os.path.join(gt_folder, video_file)

    if os.path.isfile(pred_path) and os.path.isfile(gt_path):
        preds = np.load(pred_path)
        gt = np.load(gt_path)
        
        # Repeat each value of preds 16 times 
        preds = np.repeat(preds, 16)
        
        if len(preds) < len(gt):
            #repeat last value of preds until it has the same length as gt
            preds = np.append(preds, np.repeat(preds[-1], len(gt) - len(preds)))

        if len(preds) > len(gt):
            gt = np.append(gt, np.repeat(gt[-1], len(preds) - len(gt)))

        return preds, gt
    else:
        print(f"Pred file or GT file not found for {video_file}")
        return None, None

def plot_ground_truth_vs_predictions(preds, gt, threshold=0.3):
    """
    Representa gráficamente el ground truth y las predicciones clasificadas según un umbral.
    """
    binary_preds = (preds >= threshold).astype(int)

    plt.figure(figsize=(15, 5))
    plt.plot(gt, label='Ground Truth', color='navy')
    plt.plot(binary_preds, label='Predictions', color='darkorange', alpha=0.5)
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Ground Truth vs Predictions')
    plt.legend()
    plt.show()

# Define folders
gt_folder = "frame_label/gt"
pred_folder = "frame_label/all_good"
video_file = "Burglary005_x264_pred.npy"  # Cambia esto por el nombre del archivo de video que quieras

# Load data for specific video
preds, gt = load_video_data(pred_folder, gt_folder, video_file)

if preds is not None and gt is not None:
    # Plot ground truth vs predictions
    plot_ground_truth_vs_predictions(preds, gt)
