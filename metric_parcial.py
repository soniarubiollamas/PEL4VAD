import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt
import os

def cal_false_alarm(labels, preds, threshold=0.5):
    """
    Función para calcular la tasa de falsas alarmas.
    """
    binary_preds = (preds >= threshold).astype(int)
    false_alarms = np.sum((labels == 0) & (binary_preds == 1))
    total_negatives = np.sum(labels == 0)

    if total_negatives == 0:
        return 0
    return false_alarms / total_negatives

def adapt_pred(preds, gt, pred_file, suffix):
    # search filename in txt lines
    name = pred_file.split("_x264_")[0]
    with open("G:/XONI MASTER/1 interships/UR-DMU/feature_extract/annotations/crop_videos_metrics.txt") as f:
        lines = f.readlines()
        for line in lines:
            if name in line:
                print(f'Processing {name}')
                if(suffix == "_001_i3d_pred.npy"):
                    # breakpoint()
                    init_frame = int(line.split(",")[1].split(" ")[0])
                    preds = np.repeat(preds, 16)
                    end_frame = init_frame + len(preds)
                    if(end_frame > len(gt)):
                        preds = preds[:len(gt)-init_frame]
                    preds_aux = np.zeros(len(gt))
                    preds_aux[init_frame:end_frame] = preds
                    preds = preds_aux
                    break
                if(suffix == "_005_i3d_pred.npy"):
                    init_frame = int(line.split(",")[2].split(" ")[0])
                    preds = np.repeat(preds, 16)
                    end_frame = init_frame + len(preds)
                    if(end_frame > len(gt)):
                        preds = preds[:len(gt)-init_frame]
                    preds_aux = np.zeros(len(gt))
                    preds_aux[init_frame:end_frame] = preds
                    preds = preds_aux
                    break
                if(suffix == "_01_i3d_pred.npy"):
                    init_frame = int(line.split(",")[3].split(" ")[0])
                    preds = np.repeat(preds, 16)
                    end_frame = init_frame + len(preds)
                    if(end_frame > len(gt)):
                        preds = preds[:len(gt)-init_frame]
                    preds_aux = np.zeros(len(gt))
                    preds_aux[init_frame:end_frame] = preds
                    preds = preds_aux
                    break
                if(suffix == "_x264_pred.npy"):
                    preds = np.repeat(preds, 16)
                    if len(preds) > len(gt):
                        preds = preds[:len(gt)]
                    if len(preds) < len(gt):
                        preds = np.append(preds, np.repeat(preds[-1], len(gt) - len(preds)))
                    break
    return preds

def load_and_concatenate_files(pred_folder, gt_folder, suffix):
    preds_list = []
    gt_list = []
    video_titles = []

    for pred_file in os.listdir(pred_folder):
        if pred_file.endswith(suffix):
            filename = pred_file.split("_x264_")[0] + "_x264_pred.npy"
            pred_path = os.path.join(pred_folder, pred_file)
            gt_path = os.path.join(gt_folder, filename)

            if os.path.isfile(pred_path) and os.path.isfile(gt_path):
                preds = np.load(pred_path)
                gt = np.load(gt_path)
                preds = adapt_pred(preds, gt, pred_file, suffix)
                
                if len(preds) < len(gt):
                    preds = np.append(preds, np.repeat(preds[-1], len(gt) - len(preds)))

                if len(preds) > len(gt):
                    gt = np.append(gt, np.repeat(gt[-1], len(preds) - len(gt)))

                preds_list.append(preds)
                gt_list.append(gt)
                video_titles.append(pred_file.split("_x264_")[0])
            else:
                print(f"Pred file or GT file not found for {pred_file}")

    if preds_list and gt_list:
        concatenated_preds = np.concatenate(preds_list)
        concatenated_gt = np.concatenate(gt_list)
    else:
        concatenated_preds = np.array([])
        concatenated_gt = np.array([])

    return concatenated_preds, concatenated_gt, preds_list, gt_list, video_titles

def plot_individual_gt_vs_preds(preds_list, gt_list, video_titles, suffix):
    """
    Función para graficar las predicciones frente a las etiquetas verdaderas para cada archivo individual.
    """
    num_plots = len(preds_list)
    if num_plots == 0:
        print(f"No data to plot for {suffix}")
        return

    cols = 3
    rows = (num_plots + cols - 1) // cols  # Calcular el número de filas necesarias

    plt.figure(figsize=(15, 3 * rows))  # Hacer las gráficas más estrechas

    for idx, (preds, gt, title) in enumerate(zip(preds_list, gt_list, video_titles)):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.plot(gt, label='GT', alpha=0.6)
        ax.plot(preds, label='Prediction', alpha=0.6)
        ax.set_xticks([])  # Eliminar los ticks del eje x
        ax.set_yticks([])  # Eliminar los ticks del eje y
        # ax.set_title(title, fontsize=10)  # Añadir el título del video
        # set y axis to be between 0 and 1
        ax.set_ylim(-0.1, 1.1)

    # plt.suptitle(f'GT vs Prediction for {suffix}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def calculate_metrics_and_plot(preds, gt, group_name):
    if len(preds) == 0 or len(gt) == 0:
        print(f"No data for group {group_name}")
        return
    
    false_alarm_rate = cal_false_alarm(gt, preds)
    fpr, tpr, thresholds = roc_curve(gt, preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(gt, preds)
    pr_auc = auc(recall, precision)

    print(f"Group: {group_name}")
    print(f"Tasa de falsas alarmas: {false_alarm_rate}")
    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {group_name}')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve - {group_name}')
    plt.legend(loc="lower left")
    plt.show()

# Define folders
gt_folder = "frame_label/gt"
pred_folder = "frame_label/metrics"

# Define suffixes
suffixes = ["_001_i3d_pred.npy", "_005_i3d_pred.npy", "_01_i3d_pred.npy","_x264_pred.npy"]

# Process each group
for suffix in suffixes:
    preds, gt, preds_list, gt_list, video_titles = load_and_concatenate_files(pred_folder, gt_folder, suffix)
    calculate_metrics_and_plot(preds, gt, suffix)
    
    # Also plot individual GT vs Predictions for each file
    plot_individual_gt_vs_preds(preds_list, gt_list, video_titles, suffix)
