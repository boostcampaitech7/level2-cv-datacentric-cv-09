import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import random
from utils.detect import detect
from dataset.dataset import crop_img, rotate_img, filter_vertices


LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--augmented_dir', type=str, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--crop_size', default=1024)
    args = parser.parse_args()

    return args


def crop(img, vertices, labels, length):
    region, new_vertices = crop_img(img, vertices, labels, length)

    if new_vertices.size > 0:
        # 이미지 경계로 넘어가는 vertices 제한
        new_vertices[:, [0, 2, 4, 6]] = np.clip(new_vertices[:, [0, 2, 4, 6]], 0, length)
        new_vertices[:, [1, 3, 5, 7]] = np.clip(new_vertices[:, [1, 3, 5, 7]], 0, length)

        # 매우 작은 영역의 vertices 제거
        new_vertices, _ = filter_vertices(new_vertices, np.ones_like(new_vertices), drop_under=10)
    
    return region, new_vertices


def rotate(img, vertices, angle_range=10):
    img, new_vertices = rotate_img(img, vertices)
    
    # 이미지 경계로 넘어가는 vertices 제한
    new_vertices[:, [0, 2, 4, 6]] = np.clip(new_vertices[:, [0, 2, 4, 6]], 0, img.width - 1)  # x 좌표 경계 처리
    new_vertices[:, [1, 3, 5, 7]] = np.clip(new_vertices[:, [1, 3, 5, 7]], 0, img.height - 1)  # y 좌표 경계 처리

    # 매우 작은 영역의 vertices 제거
    new_vertices, _ = filter_vertices(new_vertices, np.ones_like(new_vertices), drop_under=10)
    
    return img, new_vertices

    
def apply_augmentation(args):  
    os.makedirs(args.augmented_dir, exist_ok=True)
    
    for lang in LANGUAGE_LIST:
        augmented_annotations = {"images": {}}
        
        with open(osp.join(args.data_dir, f'{lang}_receipt/ufo/{args.split}.json'), 'r', encoding='utf-8') as f:
            anno = json.load(f)
        json_save_path = osp.join(args.augmented_dir, f'{lang}_receipt/ufo/{args.split}.json')
        os.makedirs(osp.dirname(json_save_path), exist_ok=True)
               
        for image_fpath in tqdm(glob(osp.join(args.data_dir, f'{lang}_receipt/img/{args.split}/*'))):
            image_fname = osp.basename(image_fpath)

            # 이미지와 bbox 로드
            image = Image.open(image_fpath).convert('RGB')
            image = ImageOps.exif_transpose(image)
            vertices, labels = [], []
            for word_info in anno['images'][image_fname]['words'].values():
                num_pts = np.array(word_info['points']).shape[0]
                if num_pts > 4:
                    continue
                vertices.append(np.array(word_info['points']).flatten())
                labels.append(1)
            
            vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)
            
            # Augmentation된 이미지 경로 설정
            aug_image_path = osp.join(args.augmented_dir, f'{lang}_receipt/img/{args.split}/{image_fname}')
            os.makedirs(osp.dirname(aug_image_path), exist_ok=True)
            
            # Augmentation 적용
            augmentations = [
                lambda img, verts: rotate(img, verts),
                lambda img, verts: crop(img, verts, labels, args.crop_size)
            ]
            
            # augmentations 중 하나를 무작위로 선택하여 적용
            augmentation_fn = random.choice(augmentations)
            image, vertices = augmentation_fn(image, vertices)
            
            # JSON 어노테이션 업데이트
            words = {}
            for i, bbox in enumerate(vertices.reshape(-1, 4, 2)):
                words[f"word_{i}"] = {
                    "transcription": "",
                    "points": bbox.tolist()
                }
            augmented_annotations["images"][image_fname] = {"words": words}
            image.save(aug_image_path)

        # Augmented JSON 파일 저장
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(augmented_annotations, f, ensure_ascii=False, indent=4)

    print(f"Augmented dataset 생성 완료: {args.augmented_dir}")



if __name__ == '__main__':
    args = parse_args()
    apply_augmentation(args)
