"""
llm_gemini.py를 통해 생성한 cleaned_data.csv를 후처리하는 코드입니다.
후처리 결과는 postprocessed.csv, postproessed_nida.csv로 저장됩니다.
"""
import pandas as pd

df = pd.read_csv("cleaned_data.csv")

# 1. 컬럼 제거
df.drop(columns=['Original Text'], inplace=True) # Original Text 컬럼 제거
df.dropna(subset=['Cleaned Text'], inplace=True) # 결측치 제거

# 2. [", #, *] 제거
df['Cleaned Text'] = df['Cleaned Text'].str.replace('"', '', regex=False)
df['Cleaned Text'] = df['Cleaned Text'].str.replace('#', '', regex=False)
df['Cleaned Text'] = df['Cleaned Text'].str.replace('*', '', regex=False)

# 3. '->'이 포함된 문장에서 '->' 앞의 문장 제거 (예시: '문장 A -> 문장 B' ~> 문장 B)
df['Cleaned Text'] = df['Cleaned Text'].apply(lambda x: x.split('->', 1)[1].strip() if '->' in x else x)

# 4. 앞 뒤 공백 제거
df['Cleaned Text'] = df['Cleaned Text'].str.strip()

# 4. 답변에 "니다."를 포함하는 행을 분리해 postprocessed_nida.csv에 저장
contains_nida = df[df['Cleaned Text'].str.contains("니다.", regex=False)]
df = df[~df['Cleaned Text'].str.contains("니다.", regex=False)]
contains_nida.to_csv("postprocessed_nida.csv", index=False)

# 6. 결과를 새로운 CSV 파일로 저장
df.to_csv("postprocessed.csv", index=False)
print("후처리가 완료되었습니다. 결과는 'postprocessed.csv'에 저장되었습니다.")
