import pandas as pd

df = pd.read_csv("train.csv")

df['text'] = df['text'].str.replace('"', '', regex=False)
df['text'] = df['text'].str.replace(',', ' ', regex=False)

df['text'] = df['text'].str.replace('...', '…', regex=False)
df['text'] = df['text'].str.replace('……', '…', regex=False)

df['text'] = df['text'].str.replace('-', '·', regex=False)
df['text'] = df['text'].str.replace(' · ', '·', regex=False)
df['text'] = df['text'].str.replace('· ', '·', regex=False)
df['text'] = df['text'].str.replace(' ·', '·', regex=False)
df['text'] = df['text'].str.replace('  ', ' ', regex=False)
df['text'] = df['text'].str.replace('*', ' ', regex=False)
df['text'] = df['text'].str.strip()

df.to_csv("train_processed.csv", index=False)