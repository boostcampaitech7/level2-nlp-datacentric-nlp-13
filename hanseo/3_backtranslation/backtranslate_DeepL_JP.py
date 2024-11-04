"""
label noise, text noise를 해결한 train.csv를 
DeepL API를 이용해 일본어로 back-translation 하는 코드입니다.
이후 후처리 코드인 post_process_JP.py를 실행해야 합니다. 
"""
import pandas as pd
import requests
import time
import os

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_KEY = " "

file_path = 'train.csv'
output_path = 'backtranslated_DeepL_JP_raw.csv'
repeat = 1  # 각 데이터에 대해 역번역을 수행하는 횟수
SAVE_INTERVAL = 10  # 중간 저장할 데이터 개수

def back_translate(text):
    params_ko_ja = {
        'auth_key': DEEPL_API_KEY,
        'text': text,
        'source_lang': 'KO',
        'target_lang': 'JA',
    }
    try:
        response = requests.post(DEEPL_API_URL, data=params_ko_ja)
        response.raise_for_status()
        result = response.json()
        japanese_text = result['translations'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"번역 오류 (한국어 -> 일본어): {e}")
        return text  # 오류 발생 시 원본 텍스트 반환

    # 일본어 -> 한국어 번역
    params_ja_ko = {
        'auth_key': DEEPL_API_KEY,
        'text': japanese_text,
        'source_lang': 'JA',
        'target_lang': 'KO',
    }
    try:
        response = requests.post(DEEPL_API_URL, data=params_ja_ko)
        response.raise_for_status()
        result = response.json()
        korean_text = result['translations'][0]['text']
        return korean_text
    except requests.exceptions.RequestException as e:
        print(f"번역 오류 (일본어 -> 한국어): {e}")
        return text  # 오류 발생 시 원본 텍스트 반환

# 데이터 불러오기
data = pd.read_csv(file_path)
data = data[['ID', 'text', 'target']]
total_characters = data['text'].str.len().sum()

print(f"원본 데이터 개수: {len(data)}")
print(f"총 글자 수: {total_characters}")

texts_to_translate = data['text'].tolist()
REQUEST_DELAY = 1  # API 요청 사이에 지연 추가 (단위: 초)

print("back-translation을 시작합니다...")

# 기존 출력 파일이 있으면 덮어쓰기
if os.path.exists(output_path):
    os.remove(output_path)

for r in range(repeat):
    print(f"{r + 1}번째 증강을 시작합니다...")
    translated_texts = []
    batch = []  # 중간 저장을 위한 배치 리스트

    for idx, text in enumerate(texts_to_translate):
        translated = back_translate(text)
        translated_texts.append(translated)
        batch.append({
            'ID': data.at[idx, 'ID'],
            'text': translated,
            'target': data.at[idx, 'target']
        })
        print(f"번역 완료: {idx + 1}/{len(texts_to_translate)}")
        time.sleep(REQUEST_DELAY)  # API 요청 사이에 지연 추가

        # SAVE_INTERVAL마다 중간 저장
        if (idx + 1) % SAVE_INTERVAL == 0:
            batch_df = pd.DataFrame(batch)
            if not os.path.exists(output_path):
                # 첫 저장 시 헤더 포함
                batch_df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='w')
            else:
                # 이후 저장 시 헤더 제외
                batch_df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='a', header=False)
            print(f"{idx + 1}개 데이터가 '{output_path}'에 저장되었습니다.")
            batch = []  # 배치 초기화

    # 마지막에 남은 데이터 저장
    if batch:
        batch_df = pd.DataFrame(batch)
        if not os.path.exists(output_path):
            batch_df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='w')
        else:
            batch_df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='a', header=False)
        print(f"남은 {len(batch)}개 데이터가 '{output_path}'에 저장되었습니다.")

print(f"'{output_path}'로 저장이 완료되었습니다.")
