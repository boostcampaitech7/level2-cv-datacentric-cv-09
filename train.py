import os
import os.path as osp
import time
import math
import numpy as np
import random
from datetime import timedelta
from argparse import ArgumentParser
import yaml
from  dotmap import DotMap

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset.east_dataset import EASTDataset
from dataset.dataset import SceneTextDataset
from model.model import EAST
import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the configuration YAML file")    
    args = parser.parse_args()
    return args


def do_training(config):
    wandb.init(project='OCR_receipt')
    # 실행 이름 설정
    wandb.run.name = 'baseline_250ep'
    wandb.run.save()
    wandb_args = {
        "learning_rate": config.solver.lr,
        "max_epochs": config.solver.max_epoch,
        "batch_size": config.data.batch_size
    }
    wandb.config.update(wandb_args)

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

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        wandb.log({"Mean loss": epoch_loss / num_batches})

        model_dir = config.data.model_dir
        if (epoch + 1) % config.data.save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


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