from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import pandas as pd
import tempfile
import random
from configs import build_config
from utils import setup_seed
from log import get_logger

from model import XModel
from dataset_ucf import *

from train import train_func
from test import test_func
from infer_metrics import infer_func
import argparse
import copy

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def create_gt(test_list):
    gt = []
    with open(test_list, 'r') as f:
        for line in f:
            # read that file in folder
            # breakpoint()
            filename = line.split("_")[0]
            path_gt = f'G:\XONI MASTER/1 interships/PEL4VAD/list/ucf/single_gt/{filename}_x264__9_gt.npy'
            # path_gt = f'G:/XONI MASTER/1 interships/PEL4VAD/frame_label/gt/{filename}_x264_pred.npy'
            read = np.load(path_gt)
            # breakpoint
            gt.append(read)
            
    # breakpoint()
    gt = np.concatenate(gt)
    return gt



def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        start_time = time.time()  # Start timer
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        print(torch.cuda.is_available())
        # weight_dict = torch.load(ckpt_path)
        weight_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))  # Force CPU
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
        end_time = time.time() # End timer
        load_time = end_time - start_time
        # logger.info(f'Checkpoint loaded in {load_time:.2f} seconds')
    else:
        logger.info('Not found pretrained checkpoint file.')
    # breakpoint()


def train(model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))

    initial_auc, n_far = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize {}:{:.4f} FAR:{:.5f}'.format(cfg.metrics, initial_auc, n_far))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_far = 0.0

    st = time.time()
    for epoch in range(cfg.max_epoch):
        loss1, loss2 = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
        scheduler.step()

        auc, far = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_far = far
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} | AUC:{:.4f} FAR:{:.5f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, auc, far))

    time_elapsed = time.time() - st
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | best {}:{:.4f} FAR:{:.5f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_far))


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    if cfg.dataset == 'ucf-crime':
        train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == 'xd-violence':
        train_data = XDataset(cfg, test_mode=False)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == 'shanghaiTech':
        train_data = SHDataset(cfg, test_mode=False)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    model = XModel(cfg)
    # gt = np.load(cfg.gt)
    gt = create_gt(cfg.test_list)
    device = torch.device("cpu")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    if args.mode == 'train':
        logger.info('Training Mode')
        train(model, train_loader, test_loader, gt, logger)

    elif args.mode == 'infer':
        logger.info('Test Mode')
        if cfg.ckpt_path is not None:
            # Read filenames from the config file
            with open(cfg.test_list, 'r') as f:
                dataset_files = [line.strip() for line in f]
                dataset_len = len(dataset_files)/1
            # replace the batch to 10 when using ucf dataset
            # convert dataset_len to int

            batch_size = 10 # 10  set to 10 when using UCF dataset and saving prediction times
            # batch_size = int(dataset_len) # when doing infer with gt

            # Create the .list file (e.g., 'batch_list.txt')
            list_file_path = 'batch_list.txt'
            with open(list_file_path, 'w') as list_file:
                pass
                   
            # create excel file if it doesn't exist
            if not os.path.exists('annotations/time_prediction_11_06.xlsx'):
                df = pd.DataFrame(columns=['File name', 'checkpoint time', 'load dataset time', 'pred time','complete infer time'])
                df.to_excel('annotations/time_prediction_11_06.xlsx', index=False)
            else:
                df = pd.read_excel('annotations/time_prediction_11_06.xlsx')
           
            
            # Results collection
            results = []

            files_processed = 0  # Counter for the number of files processed

            # Process files in batches
            for start_index in range(0, len(dataset_files), batch_size):
                end_index = min(start_index + batch_size, len(dataset_files))
                current_batch = dataset_files[start_index:end_index]
                skip_file = False
                # Append to the .list file
                with open(list_file_path, 'a') as list_file:
                    for filename in current_batch:
                        # Check if the filename is already in the excel file
                        if filename in df['File name'].values:
                            # skip this batch files, aka, go back to for start_index in range(0, len(dataset_files), batch_size):
                            logger.info(f"Skipping {filename} as it is already in the Excel file")
                            skip_file = True # CHANGE TO TRUE WHEN NOT DOING INFER   UCF
                            break
                        else:
                            list_file.write(filename + '\n')
                if skip_file:
                    # next batch
                    continue
                else: 
                    total_checkpoint_time = 0.0
                    total_complete_time = 0.0
                    total_load_time = 0.0
                    total_model_time = 0.0
                    num_files = 0  # Counter for the number of files
                    repeat = 1

                    # Experimental loop (assuming you need it)
                    for experiment_num in range(repeat):
                        # Reset random seed
                        setup_seed(cfg.seed)

                        # Load checkpoint
                        start_load_time = time.time()
                        load_checkpoint(model, cfg.ckpt_path, logger)
                        end_load_time = time.time()
                        checkpoint_time = end_load_time - start_load_time

                        # Create dataset for the current batch 
                        test_data = UCFDataset(cfg, test_mode=True, files=list_file_path) 
                        test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                                num_workers=cfg.workers, pin_memory=True)
                        

                        # Inference with timing
                        start_infer = time.time()
                        time_load_dataset, time_model = infer_func(model, test_loader, gt, logger, cfg)  # No 'index' needed here
                        complete_time = time.time() - start_infer
                        dataset_file = current_batch[0]

                        # Update counters
                        total_checkpoint_time += checkpoint_time
                        total_complete_time += complete_time
                        total_load_time += time_load_dataset
                        total_model_time += time_model

                        logger.info(f"Processed {num_files+1}/{repeat}")
                        num_files += 1

                    
                    # Calculate averages
                    if num_files > 0:  # Protection against zero files 
                        average_checkpoint_time = total_checkpoint_time / num_files
                        average_complete_time = total_complete_time / num_files
                        average_load_time = total_load_time / num_files
                        average_model_time = total_model_time / num_files
                    else:
                        average_checkpoint_time = 0.0
                        average_complete_time = 0.0
                        average_load_time = 0.0
                        average_model_time = 0.0 

                    # Store results
                    results.append({
                        'File name': dataset_file,  # Assuming you only process one 'dataset_file' here
                        'checkpoint time': average_checkpoint_time,
                        'load dataset time': average_load_time,
                        'pred time': average_model_time,
                        'complete infer time': average_complete_time
                    })
                
                    logger.info(f"Processed {files_processed+1} out of {dataset_len/batch_size} files")
                    files_processed += 1

                    # Remove the .list file
                    os.remove(list_file_path)

                    # Append the new data to the existing data
                    df_new = pd.DataFrame(results)
                    df_combined = pd.concat([df, df_new], ignore_index=True)

                    # Write the combined data back to the file
                    # df_combined.to_excel('annotations/time_prediction_11_06.xlsx', index=False)

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='model status: (train or infer)')
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    main(cfg)
