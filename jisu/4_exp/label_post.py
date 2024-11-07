import csv

# 입력 파일 경로 설정 (기존 파일을 덮어쓸 예정)
file_path = '10_generated_raw.csv'


# 데이터 처리 함수
def process_csv(file_path):
    # CSV 파일 읽기
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = []
        
        for row in reader:
            # 마지막 두 값 중 하나는 삭제, 나머지는 새로운 열로 이동
            main_data = row[:-2]  # 앞의 주요 데이터 (마지막 두 값을 제외)
            new_column_value = row[-2]  # 두 번째 마지막 값
            rows.append(main_data + [new_column_value])  # 새로운 열에 추가

    # 기존 파일에 덮어쓰기
    with open(file_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

# CSV 처리 실행
process_csv(file_path)