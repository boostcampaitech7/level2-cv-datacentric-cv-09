import json

# JSON 파일을 읽어오는 함수
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# JSON 데이터를 파일에 저장하는 함수
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# 메인 실행 코드
if __name__ == "__main__":
    # 입력 파일 경로
    input_file = "./code/data/cosmetics_dataset/info/medicine_00001.json"
    
    # 출력 파일 경로
    output_file = "./code/data/cosmetics_dataset/info/formatted_medicine_00001.json"
    
    # JSON 파일 읽기
    data = read_json_file(input_file)
    
    # 데이터를 보기 좋게 정리하여 저장
    save_json_file(data, output_file)
    
    print(f"정리된 JSON 파일이 {output_file}로 저장되었습니다.")