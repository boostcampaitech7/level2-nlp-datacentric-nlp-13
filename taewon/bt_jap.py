import pandas as pd
import re

# CSV 파일 읽기
df = pd.read_csv('result_deepl.csv')

# ID 열의 모든 값에 "_BT_jap" 추가
df['ID'] = df['ID'] + "_BT_jap"

# 텍스트 길이가 8자 이상인 행만 유지
df = df[df['text'].str.len() >= 8]

# 결과를 새 CSV 파일로 저장
df.to_csv('result_deepl_proceeded.csv', index=False)

print("처리가 완료되었습니다. 결과는 'result_deepl_proceeded.csv' 파일에 저장되었습니다.")