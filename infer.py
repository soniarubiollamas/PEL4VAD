
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval() 
        pred = torch.zeros(0)
        normal_preds = torch.zeros(0)
        normal_labels = torch.zeros(0)
        gt_tmp = torch.tensor(gt.copy()) 
        start_data_loading = time.time()  # Start timer for data loading
        time_pred_start = time.time()
        for i, (v_input, filename) in enumerate(dataloader):
            v_input = v_input.float()
            if v_input.shape[0] == 1:
                v_input = torch.squeeze(v_input, dim=0)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            
            logits, _ = model(v_input, seq_len)
            logits = torch.mean(logits, 0) 
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]
            time_pred_end = time.time()
            time_pred = time_pred_end - time_pred_start

            pred = torch.cat((pred, logits)) 
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]
        end_data_loading = time.time()  # End timer for data loading
        data_loading_time = end_data_loading - start_data_loading
        # logger.info(f"Data loading and preprocessing time: {data_loading_time:.4f} seconds")
        # n_far = cal_false_alarm(normal_labels, normal_preds)
        pred = list(pred.cpu().detach().numpy())  
        # far = cal_false_alarm(normal_labels, normal_preds)
        # fpr, tpr, thresholds = roc_curve(list(gt), np.repeat(pred, 16))
        # roc_auc = auc(fpr, tpr)
        # pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        # pr_auc = auc(rec, pre)
    filename_save = filename[0].split('/')[-1].split('.')[0]
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    np.save('frame_label/'+filename_save+timestamp+'_pred.npy', pred)
    time_elapsed = time.time() - st
    # logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
    #     roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
    # logger.info(' Complete in {:.0f}m {:.4f}s\n'.format(
        # time_elapsed // 60, time_elapsed % 60))
    return time_pred
    
    
    # Data Saving
    
    # output_dir = 'test_results'  # Set the desired output directory
    # os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist 
    # # print that the folder was created
    # print(f"Created folder {output_dir}")

    # results = {
    #     'all_preds': np.repeat(pred, 16),
    #     'all_labels': list(gt),
    #     'fpr': fpr,
    #     'tpr': tpr,
    #     'thresholds': thresholds,  # Assuming your roc_curve function returns thresholds
    #     'pre': pre,
    #     'rec': rec,
    #     'roc_auc': roc_auc,
    #     'pr_auc': pr_auc,
    #     'far': n_far  
    # }

    # with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
    #     pickle.dump(results, f)
    #     print(f"Saved results to {os.path.join(output_dir, 'results.pkl')}")

    # df = pd.DataFrame(results) 
    # df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    # with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    #     json.dump(results, f) 
