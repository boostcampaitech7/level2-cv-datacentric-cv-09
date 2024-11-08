import json
import random
import os
import shutil
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to the JSON file with OCR data')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the original images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for json files')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train set (default: 0.8)')
    args = parser.parse_args()
    
    return args


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def split_data_with_images(data, image_dir, output_dir, train_ratio=0.8):
    images = data["images"]
    image_keys = list(images.keys())
    
    # 데이터를 섞은 후 지정된 비율로 분할
    random.shuffle(image_keys)
    train_size = int(len(image_keys) * train_ratio)
    
    train_keys = image_keys[:train_size]
    val_keys = image_keys[train_size:]
    
    # 분할된 데이터로 각각의 JSON 구조를 생성
    train_data = {"images": {key: images[key] for key in train_keys}}
    val_data = {"images": {key: images[key] for key in val_keys}}
    
    # Output 경로 설정
    train_image_dir = os.path.join(output_dir, 'img', 'train')
    val_image_dir = os.path.join(output_dir, 'img', 'val')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    
    # 이미지 파일을 각 경로로 복사
    for key in train_keys:
        src_image_path = os.path.join(image_dir, key)
        dst_image_path = os.path.join(train_image_dir, key)
        shutil.copy(src_image_path, dst_image_path)
    
    for key in val_keys:
        src_image_path = os.path.join(image_dir, key)
        dst_image_path = os.path.join(val_image_dir, key)
        shutil.copy(src_image_path, dst_image_path)
    
    return train_data, val_data

# 데이터를 JSON 파일로 저장하는 함수
def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parse_args()

    # JSON 데이터 로드
    data = load_json(args.file_path)

    # train_ratio를 0.8로 설정해 train/val 데이터를 80:20으로 나눔
    train_data, val_data = split_data_with_images(data, args.image_dir, args.output_dir, train_ratio=args.train_ratio)

    # JSON 파일로 train과 val 데이터 저장
    save_json(train_data, os.path.join(args.output_dir, 'ufo', 'train.json'))
    save_json(val_data, os.path.join(args.output_dir, 'ufo', 'val.json'))
