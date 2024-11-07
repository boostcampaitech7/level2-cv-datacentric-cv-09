import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import matplotlib.pyplot as plt

# def save_augmented_samples(data_loader, num_samples=30, save_path="/data/ephemeral/home/ES-datacentric-cv-09/code/data/augmented_samples.jpg"):
#     # DataLoader에서 첫 번째 배치 가져오기
#     for img, gt_score_map, gt_geo_map, roi_mask in data_loader:
#         img = img[:num_samples].cpu().numpy()  # 최대 30개 이미지 선택 (GPU에서 CPU로 변환)
        
#         # 이미지 그리기
#         fig, axes = plt.subplots(5, 6, figsize=(15, 10))  # 5x6 그리드 (최대 30장)
#         for i, ax in enumerate(axes.flat):
#             if i < len(img):  # `img`의 크기보다 인덱스가 작을 때만 이미지 표시
#                 ax.imshow(img[i].transpose(1, 2, 0))  # 채널 순서 조정
#             ax.axis('off')  # 축 숨기기
        
#         # 이미지 저장
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close(fig)
#         print(f"Augmented samples saved to {save_path}")
#         break  # 첫 번째 배치만 사용

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    import wandb
    wandb.init(project='OCR_receipt',
            entity='int-hyokwang-kaist'
            )
    # 실행 이름 설정
    wandb.run.name = 'ES_without_crop'
    wandb.run.save()
    wandb_args = {
    "learning_rate": learning_rate,
    "max_epochs": max_epoch,
    "batch_size": batch_size
    }
    wandb.config.update(wandb_args)
    
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # # Augmentation이 적용된 샘플 이미지 30장 저장
    # save_augmented_samples(train_loader)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
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
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)