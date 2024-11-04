"""
cleaning.py를 통해 생성한 cleaned_data.csv를 후처리하는 코드입니다.
후처리 결과는 text_cleaned.csv, text_cleaned_nida.csv로 저장됩니다.
"""
import pandas as pd
import re

# cleaned_data.csv 파일을 읽어옵니다.
df = pd.read_csv("cleaned_data.csv")

# 1. 컬럼 제거
df.drop(columns=['Original Text'], inplace=True)  # Original Text 컬럼 제거
df.dropna(subset=['Cleaned Text'], inplace=True)  # Cleaned Text 컬럼의 결측치 제거

# 2. 특수문자 처리
df['Cleaned Text'] = df['Cleaned Text'].str.replace(', ', '·', regex=False)
df['Cleaned Text'] = df['Cleaned Text'].str.replace('"', '', regex=False)
df['Cleaned Text'] = df['Cleaned Text'].str.replace('#', '', regex=False)
df['Cleaned Text'] = df['Cleaned Text'].str.replace('*', '', regex=False)

# 3. 응답 오류 처리: '->'이 포함된 문장에서 '->' 앞의 문장 제거 (예시: '문장 A -> 문장 B' ~> '문장 B')
df['Cleaned Text'] = df['Cleaned Text'].apply(
    lambda x: x.split('->', 1)[1].strip() if '->' in x else x
)

# 4. 응답 오류 처리: '뉴스 제목:'이 포함된 문장에서 '뉴스 제목:' 앞의 문장 제거 및 실제 뉴스 제목 추출
df['Cleaned Text'] = df['Cleaned Text'].str.extract(
    r'뉴스 제목:\s*(.*)', expand=False
).fillna(df['Cleaned Text'])

# 5. 앞 뒤 공백 제거
df['Cleaned Text'] = df['Cleaned Text'].str.strip()

# 6. 응답 오류 처리: 답변에 "니다."를 포함하는 행을 분리해 text_cleaned_nida.csv에 저장
contains_nida = df[df['Cleaned Text'].str.contains("니다.", regex=False)]
df = df[~df['Cleaned Text'].str.contains("니다.", regex=False)]
contains_nida.to_csv("text_cleaned_nida.csv", index=False, encoding='utf-8-sig')

# 7. 'Cleaned Text' 열 이름을 'text'로 변경
df.rename(columns={'Cleaned Text': 'text'}, inplace=True)

# 8. 'ID' 열을 기준으로 오름차순 정렬
df.sort_values(by='ID', ascending=True, inplace=True)

# 9. 결과를 새로운 CSV 파일로 저장
df.to_csv("text_cleaned.csv", index=False, encoding='utf-8-sig')
print("후처리가 완료되었습니다. 결과는 'text_cleaned.csv'에 저장되었습니다.")
