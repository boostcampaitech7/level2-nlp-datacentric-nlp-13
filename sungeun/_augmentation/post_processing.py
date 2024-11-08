import pandas as pd
 
df = pd.read_csv("train_augmented.csv")
print(df.head())
print(len(df))

# 'augmented_text'가 비어있는 경우 'text'의 값을 가져오기
df['augmented_text'] = df['augmented_text'].fillna(df['text'])

# text 열 삭제
df = df.drop(columns=["text"])

# 열 이름 변경: augmented_text -> text
df = df.rename(columns={"augmented_text": "text"})

# 열 순서 변경: ID, text, target
df = df[["ID", "text", "target"]]

df['text'] = df['text'].str.replace('"', '', regex=False)
df['text'] = df['text'].str.replace("'", '', regex=False)
df['text'] = df['text'].str.replace('*', '', regex=False)
df['text'] = df['text'].str.replace(',', ' ', regex=False)

df['text'] = df['text'].str.replace('...', '…', regex=False)
df['text'] = df['text'].str.replace('….', '…', regex=False)

df['text'] = df['text'].str.replace('-', '·', regex=False)
df['text'] = df['text'].str.replace(' · ', '·', regex=False)
df['text'] = df['text'].str.replace('· ', '·', regex=False)
df['text'] = df['text'].str.replace(' ·', '·', regex=False)
df['text'] = df['text'].str.replace('  ', ' ', regex=False)
df['text'] = df['text'].str.replace('  ', ' ', regex=False)

contains_nida = df[df['text'].str.contains("어렵습니다.", regex=False)]
df = df[~df['text'].str.contains("어렵습니다.", regex=False)]
contains_nida.to_csv("nida.csv", index=False, encoding='utf-8-sig')

df['text'] = df['text'].str.strip()


# ID 열에 "_syn_rep" 접미사 추가 --> 유의어 교체
df["ID"] = df["ID"] + "_syn_rep"

# 결과 확인
print(df.head())
print(len(df))
# 수정된 데이터프레임을 새로운 CSV 파일로 저장 (필요할 경우)
df.to_csv("train_augmented_processed4.csv", index=False, encoding="utf-8")