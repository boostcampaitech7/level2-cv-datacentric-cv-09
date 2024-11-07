import json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

# JSON 파일 경로와 이미지 경로 지정
json_path = "/data/ephemeral/home/ES-datacentric-cv-09/code/data/additional_dataset/train/info_data/*.json"
base_dir = "/data/ephemeral/home/ES-datacentric-cv-09/code/data/additional_dataset/train"
output_dir = "/data/ephemeral/home/ES-datacentric-cv-09/code/data/visual_additional_dataset"

# 결과를 저장할 디렉토리 생성 (존재하지 않으면 생성)
os.makedirs(output_dir, exist_ok=True)

# 모든 JSON 파일 읽어서 bbox 시각화 후 이미지 저장
for file_path in glob.glob(json_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # 기존 이미지 경로에서 파일명만 추출
    original_image_path = data["image_path"]
    image_filename = os.path.basename(original_image_path)  # "000.jpg" 형태로 파일명 가져오기

    # 새로운 이미지 경로 생성
    image_path = os.path.join(base_dir, "images", image_filename)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        continue  # 이미지가 없으면 다음 파일로 넘어감

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 읽기 때문에 RGB로 변환

    # 이미지 시각화 설정
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # bbox 시각화 (important_words의 Coordinates 기준)
    for item in data["important_words"]:
        for word in item["words"]:
            coords = word["Coordinates"]
            xmin, ymin = coords["xmin"], coords["ymin"]
            xmax, ymax = coords["xmax"], coords["ymax"]

            # 사각형 그리기
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

    # 저장할 파일명 설정
    base_name = os.path.basename(file_path).replace(".json", ".png")
    output_path = os.path.join(output_dir, base_name)

    # 이미지 저장
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)  # 메모리 해제를 위해 figure 닫기

print(f"이미지 저장이 완료되었습니다! 저장 위치: {output_dir}")
