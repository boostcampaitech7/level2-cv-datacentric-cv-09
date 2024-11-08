import json
import os
import numpy as np
from utils.deteval import default_evaluation_params, calc_deteval_metrics
from argparse import ArgumentParser 
import os.path as osp
from collections import namedtuple
from pathlib import Path
from PIL import Image, ImageDraw


LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'miscls_data'))
    args = parser.parse_args()
    
    return args


# 주어진 JSON 파일을 로드하는 함수
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# 이미지의 points를 rect 형식으로 변환
def convert_to_rect(bboxes_data):
    bboxes_rect = {}
    for image_id, image_data in bboxes_data.get("images", {}).items():
        bboxes_rect[image_id] = []
        for _, word_data in image_data.get("words", {}).items():
            points = np.array(word_data["points"])
            xmin, ymin = points[:, 0].min(), points[:, 1].min()
            xmax, ymax = points[:, 0].max(), points[:, 1].max()
            rect_bbox = [xmin, ymin, xmax, ymax]
            bboxes_rect[image_id].append(rect_bbox)
    return bboxes_rect


# ground truth와 prediction을 기반으로 precision, recall, hmean(f1)을 계산
def calculate_metrics(pred_bboxes_dict, gt_bboxes_dict):
    eval_hparams = default_evaluation_params()

    # calc_deteval_metrics 함수를 사용해 metric 계산
    results = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, eval_hparams, bbox_format='rect', verbose=True)

    # 결과 출력
    method_metrics = results['total']
    print("Precision:", method_metrics['precision'])
    print("Recall:", method_metrics['recall'])
    print("F1 Score (Hmean):", method_metrics['hmean'])

    return method_metrics


def area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, dx) * max(0, dy)


# 평가 메트릭 계산
def calc_metrics_per_image(pred_bboxes, gt_bboxes, eval_params):
    eval_params = default_evaluation_params()

    # 오분류로 판정할 리스트
    misclassified_images = []
    per_image_metrics = []

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    for image_id, gt_bboxes_list in gt_bboxes.items():
        pred_bboxes_list = pred_bboxes.get(image_id, [])
        
        gtRects = [Rectangle(*bbox) for bbox in gt_bboxes_list]
        detRects = [Rectangle(*bbox) for bbox in pred_bboxes_list]
        
        recallMat = np.zeros((len(gtRects), len(detRects)))
        precisionMat = np.zeros((len(gtRects), len(detRects)))
        
        # recall 및 precision 매트릭스 계산
        for i, gtRect in enumerate(gtRects):
            for j, detRect in enumerate(detRects):
                intersected_area = area(gtRect, detRect)
                gt_area = (gtRect.xmax - gtRect.xmin) * (gtRect.ymax - gtRect.ymin)
                pred_area = (detRect.xmax - detRect.xmin) * (detRect.ymax - detRect.ymin)
                
                recallMat[i, j] = intersected_area / gt_area if gt_area > 0 else 0
                precisionMat[i, j] = intersected_area / pred_area if pred_area > 0 else 0

        # Area recall, precision 평균 계산
        area_recall = np.mean(np.max(recallMat, axis=1)) if len(gtRects) > 0 else 0
        area_precision = np.mean(np.max(precisionMat, axis=0)) if len(detRects) > 0 else 0
        
        per_image_metrics.append({
            'image_id': image_id,
            'area_recall': area_recall,
            'area_precision': area_precision
        })
        
        # 오분류 기준에 따라 판정
        if area_recall < eval_params['AREA_RECALL_CONSTRAINT'] or area_precision < eval_params['AREA_PRECISION_CONSTRAINT']:
            misclassified_images.append(image_id)
            
    return per_image_metrics, misclassified_images


# 오분류된 이미지 시각화
def visualize_misclassifications(misclassified_images, pred_data, gt_data, data_dir, output_dir):
    nation_dict = {
        'vi': 'vietnamese_receipt',
        'th': 'thai_receipt',
        'zh': 'chinese_receipt',
        'ja': 'japanese_receipt',
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_id in misclassified_images:
        # 이미지 경로 설정
        im_path = im_path = Path(data_dir) / nation_dict[image_id.split('.')[1]] / 'img' / 'val' / image_id
        img = Image.open(im_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Ground Truth (빨간색) bounding boxes
        if image_id in gt_data["images"]:
            for obj in gt_data["images"][image_id]["words"].values():
                pts = [(int(p[0]), int(p[1])) for p in obj["points"]]
                draw.polygon(pts, outline=(0, 0, 255))  # 파란색 윤곽선
        
        # Prediction (파란색) bounding boxes
        if image_id in pred_data["images"]:
            for obj in pred_data["images"][image_id]["words"].values():
                pts = [(int(p[0]), int(p[1])) for p in obj["points"]]
                draw.polygon(pts, outline=(255, 0, 0))  # 빨간색 윤곽선

        # 결과 이미지 저장
        img.save(os.path.join(output_dir, image_id))


def main(args):
    eval_params = default_evaluation_params()
    
    # ground-truth / prediction json 파일 로드
    gt_data = dict(images=dict())
    for lang in LANGUAGE_LIST:
        data = load_json(osp.join(args.data_dir, '{}_receipt/ufo/{}.json'.format(lang, 'val')))
        for im in data['images']:
            gt_data['images'][im] = data['images'][im]
    pred_data = load_json(args.pred_path)

    # bounding box 데이터를 rect 형식으로 변환
    gt_bboxes_dict = convert_to_rect(gt_data)
    pred_bboxes_dict = convert_to_rect(pred_data)
    
    metrics = calculate_metrics(pred_bboxes_dict, gt_bboxes_dict)
    per_image_metrics, misclassified_images = calc_metrics_per_image(pred_bboxes_dict, gt_bboxes_dict, eval_params)

    visualize_misclassifications(misclassified_images, pred_data, gt_data, args.data_dir, args.output_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
