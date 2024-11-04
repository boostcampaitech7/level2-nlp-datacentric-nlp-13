from googletrans import Translator
import pandas as pd
import time

train_df = pd.read_csv("../_llm_data/2_label_cleaned_train_025_040.csv")
df = pd.DataFrame(train_df)
# df = df.head()
# print(df.head())

translator = Translator()

def back_translate(text, src='ko', mid='ja', dest='ko'):
    """
    역번역 함수: 한국어 -> 일본어 -> 한국어
    src: 원본 언어 (한국어)
    mid: 중간 번역 언어 (일본어)
    dest: 최종 언어 (한국어)
    """
    try:
        # 한국어에서 일본어로 번역
        translated_to_mid = translator.translate(text, src=src, dest=mid).text
        time.sleep(1)  # 너무 많은 요청을 한 번에 보내지 않도록 대기
        # 일본어에서 다시 한국어로 번역
        back_translated = translator.translate(translated_to_mid, src=mid, dest=dest).text
        time.sleep(1)  # 대기
        return back_translated
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # 오류 발생 시 원본 텍스트 반환

# 역번역 적용하기
for idx, row in df.iterrows():
    df.at[idx, 'text'] = back_translate(row['text'])

    if idx % 10 == 0:  # 매 10번째 행마다 상태 출력
        print(f"{idx} rows processed...")

# 결과 출력
# print(df)
df.to_csv('result_jap2.csv', index=False, encoding='utf-8-sig')