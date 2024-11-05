import pandas as pd
import glob

# CSV 파일들의 경로를 지정합니다. 현재 디렉토리에 있다고 가정합니다.
csv_files = glob.glob('*.csv')

# 빈 리스트를 생성하여 각 DataFrame을 저장할 준비를 합니다.
df_list = []

# 각 CSV 파일을 읽어 DataFrame으로 변환하고 리스트에 추가합니다.
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# 모든 DataFrame을 하나로 병합합니다.
merged_df = pd.concat(df_list, ignore_index=True)

# text 열에 clean_text 함수 적용
merged_df['text'] = merged_df['text'].str.strip().str.replace(r"(^['\"]|['\"]$)", '', regex=True).str.strip()
# ID 열을 기준으로 정렬합니다.
merged_df = merged_df.sort_values('ID')

# 병합된 DataFrame을 새로운 CSV 파일로 저장합니다.
merged_df.to_csv('train.csv', index=False)

print("파일 병합 완료. 'train.csv'로 저장되었습니다.")