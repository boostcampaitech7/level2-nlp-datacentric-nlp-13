"""
생성한 generated_raw.csv를 후처리하는 코드입니다.
후처리 결과는 generated.csv로 저장됩니다.
"""

TARGET_NUM = 3
INPUT_FILE = f"{TARGET_NUM}_generated_raw.csv"
OUTPUT_FILE = f"{TARGET_NUM}_generated.csv"

import pandas as pd

df = pd.read_csv(INPUT_FILE)

# 1. 특수 문자 처리
df['text'] = (
    df['text']
    .str.replace('"', '', regex=False)
    .str.replace("'", '', regex=False)
    .str.replace('#', '', regex=False)
    .str.replace('*', '', regex=False)
    .str.replace(', ', ' ', regex=False)
    .str.replace('-', ' ', regex=False)
    .str.replace('...', '…', regex=False)
    .str.replace('….', '…', regex=False)
    .str.replace(' · ', '·', regex=False)
    .str.replace('· ', '·', regex=False)
    .str.replace(' ·', '·', regex=False)
    .str.replace('  ', ' ', regex=False)
    .str.replace('  ', ' ', regex=False)
    .str.strip()
)

# 2. 후처리 결과 저장
df.to_csv(OUTPUT_FILE, index=False)
print(f"후처리가 완료되었습니다. 결과는 {OUTPUT_FILE}에 저장되었습니다.")