from googletrans import Translator
import pandas as pd
import time

train_df = pd.read_csv("../_llm_data/1_label_cleanded_train_0.00_0.60.csv")
df = pd.DataFrame(train_df)

# print(df.head())

translator = Translator()

def back_translate(text, src='ko', mid='en', dest='ko'):
    """
    역번역 함수: 한국어 -> 영어 -> 한국어
    src: 원본 언어 (한국어)
    mid: 중간 번역 언어 (영어)
    dest: 최종 언어 (한국어)
    """
    try:
        # 한국어에서 영어로 번역
        translated_to_mid = translator.translate(text, src=src, dest=mid).text
        time.sleep(1)  # 너무 많은 요청을 한 번에 보내지 않도록 대기
        # 영어에서 다시 한국어로 번역
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
print(df)
df.to_csv('result_eng.csv', index=False, encoding='utf-8-sig')