import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# JSON 파일이 있는 디렉토리 경로
dir_directory = '/data/ephemeral/home/ES-datacentric-cv-09/code/data/cosmetics_dataset'
output_directory = '/data/ephemeral/home/ES-datacentric-cv-09/code/data/visual_cosmetics'  # 저장할 디렉토리

directory = os.path.join(dir_directory, "info")
imgdir_path = os.path.join(dir_directory, "img")
# JSON 데이터 읽기
json_data = []

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            json_data.append(data)

# 시각화 함수
def visualize_image_with_annotations(data):
    # 이미지 경로
    img_path = os.path.join(imgdir_path, f"{data['name']}")
    img = Image.open(img_path)

    # 새로운 그림과 축 만들기
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # 주석과 바운딩 박스 그리기
    for annotation in data['annotations']:
        for bbox in annotation['bbox']:
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']),  # (x,y)
                bbox['width'], bbox['height'],  # width, height
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.axis('off')  # 축 숨기기

    # 이미지 저장
    output_path = os.path.join(output_directory, f"{data['Identifier']}.jpg")  # 파일 이름 설정
    plt.savefig(output_path, bbox_inches='tight')  # 이미지 저장
    plt.close()  # 플롯 닫기

# 모든 JSON 데이터에 대해 시각화
for data in json_data:
    visualize_image_with_annotations(data)
