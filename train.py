import os
import os.path as osp
import time
import math
import numpy as np
import random
from datetime import timedelta
from argparse import ArgumentParser
import yaml
from dotmap import DotMap

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset.east_dataset import EASTDataset
from dataset.dataset import SceneTextDataset
from model.model import EAST
import wandb
import shutil


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the configuration YAML file")    
    args = parser.parse_args()
    return args


def initialize_directory(config):
    model_dir = config.data.model_dir
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = osp.join(model_dir, config.exp_name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir    


def update_log(log_dir, epoch, mean_loss, elapsed_time):
    log_path = osp.join(log_dir, 'training_log.txt')
    with open(log_path, 'a') as f:
        f.write(f"Epoch: {epoch + 1}, Mean Loss: {mean_loss:.4f}, Elapsed Time: {elapsed_time}\n")


def do_training(config):
    wandb.init(project=config.wandb.project_name)
    # 실행 이름 설정
    wandb.run.name = config.exp_name
    wandb.run.save()
    wandb_args = {
        "learning_rate": config.solver.lr,
        "max_epochs": config.solver.max_epoch,
        "batch_size": config.data.batch_size
    }
    wandb.config.update(wandb_args)

    log_dir = initialize_directory(config)

    dataset = SceneTextDataset(
        config.data.data_dir,
        split='train',
        image_size=config.data.image_size,
        crop_size=config.data.input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / config.data.batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.solver.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[config.solver.max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(config.solver.max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                wandb.log({"Cls loss": extra_info['cls_loss'],"Angle loss": extra_info['angle_loss'], "IoU loss": extra_info['iou_loss']})

        scheduler.step()

        mean_loss = epoch_loss / num_batches
        elapsed_time = timedelta(seconds=time.time() - epoch_start)
        print(f'Mean loss: {mean_loss:.4f} | Elapsed time: {elapsed_time}')
        wandb.log({"Mean loss": mean_loss})

        # Update log file
        update_log(log_dir, epoch, mean_loss, elapsed_time)

        # Save checkpoint at intervals
        if (epoch + 1) % config.data.save_interval == 0:
            ckpt_fpath = osp.join(log_dir, f"epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_fpath)

    # save confog file
    config_file_path = args.config
    shutil.copy(config_file_path, os.path.join(log_dir, os.path.basename(config_file_path)))

    latest_ckpt_fpath = osp.join(log_dir, f"latest.pt")
    torch.save(model.state_dict(), latest_ckpt_fpath)

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    # random seed
    SEED = config.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # check image size
    if config.data.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    do_training(config)

if __name__ == '__main__':
    args = parse_args()
    main(args)