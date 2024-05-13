import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve, plot_confusion_matrix




results_path = os.path.join('test_results', 'results.pkl')  # Adapt the path if needed 
with open(results_path, 'rb') as f:
    results = pickle.load(f)
# breakpoint()
thresholds = results['thresholds']
fpr = results['fpr']
tpr = results['tpr']

# Find the index of the threshold closest to 0.5
best_threshold_idx = np.argmin(np.abs(thresholds - 0.5)) 
best_threshold = thresholds[best_threshold_idx]

y_pred = (results['all_preds'] >= best_threshold).astype(int)
cm = confusion_matrix(results['all_labels'], y_pred)

precision = cm[1,1] / (cm[0,1] + cm[1,1])
recall = cm[1,1] / (cm[1,1] + cm[1,0])
# [[950529  75023]
# [ 45023  39105]]
breakpoint()
print('At threshold {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(best_threshold, precision, recall))
# At threshold 0.500, Precision: 0.343, Recall: 0.465

########################
roc_auc = results['roc_auc']  

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Reference line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

pre = results['pre']
rec = results['rec']


#################

pre = results['pre']
rec = results['rec']
pr_auc = results['pr_auc']  

# Plot the Precision-Recall curve
plt.figure()
plt.plot(rec, pre, color='teal', lw=2, label='Precision-Recall Curve (area = %0.3f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")  # Lower left might be better for PR curves 
plt.show()



# Precision at Recall 1.0 
precision_at_recall_1 = None  # Initialize as None

if 1.0 in pre:  # Check if perfect precision is achievable
    idx = np.where(pre == 1.0)[0][0]  # Find the first index if multiple occurrences  
    precision_at_recall_1 = rec[idx] 
else:
    # Find the closest precision to 1.0 
    closest_precision_idx = np.argmin(np.abs(pre - 1.0))
    precision_at_recall_1 = rec[closest_precision_idx] 

print('Recall when Precision is 1.0 (or closest value): {:.3f}'.format(precision_at_recall_1))
