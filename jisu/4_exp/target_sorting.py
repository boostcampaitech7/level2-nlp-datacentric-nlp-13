"""
주어진 csv 파일(ID,text,target)에서 target을 기준으로 분리하여 각각 csv 파일로 저장하는 코드
파일 경로 수정이 필요합니다.
"""
import pandas as pd

# CSV 파일 불러오기
file_path = './recent_output.csv'  # 실제 파일 경로로 수정하세요
data = pd.read_csv(file_path)

# target 값에 따라 데이터를 분리하여 각각 CSV 파일로 저장
for target_value in range(7):  # target 값이 0부터 6까지이므로 range(7)
    subset = data[data['target'] == target_value]  # 특정 target 값의 데이터 선택
    output_file = f'target_{target_value}.csv'  # 저장할 파일 이름 설정
    subset.to_csv(output_file, index=False)  # CSV 파일로 저장
    print(f"{output_file}로 저장 완료")  # 저장 완료 메시지 출력