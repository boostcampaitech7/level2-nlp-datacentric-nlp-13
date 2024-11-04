"""
backtranslate_DeepL_JP.py를 통해 생성한 backtranslated_DeepL_JP_raw.csv를 후처리하는 코드입니다.
후처리 결과는 backtranslated_DeepL_JP_processed.csv로 저장됩니다.
"""
import pandas as pd

df = pd.read_csv("backtranslated_DeepL_JP.csv")

df['text'] = df['text'].str.replace('"', '', regex=False)
df['text'] = df['text'].str.replace(',', '·', regex=False)

df['text'] = df['text'].str.replace('...', '…', regex=False)
df['text'] = df['text'].str.replace('….', '…', regex=False)

df['text'] = df['text'].str.replace('-', '·', regex=False)
df['text'] = df['text'].str.replace(' · ', '·', regex=False)
df['text'] = df['text'].str.replace('· ', '·', regex=False)
df['text'] = df['text'].str.replace(' ·', '·', regex=False)


df.to_csv("backtranslated_DeepL_JP_processed.csv", index=False)