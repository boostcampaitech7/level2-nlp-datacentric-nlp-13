import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
import time

# Gemini API 키 설정
genai.configure(api_key='AIzaSyBfY3CLZhrFdVFGZwaRlEipxpByG7nx46o')

# Gemini 모델 설정
model = genai.GenerativeModel('gemini-1.5-flash-latest')


def evolve_title(title):
    # 캐시에서 제목 확인
    prompt = f"""당신은 뉴스 헤드라인을 개선하는 전문 에디터 어시스턴트입니다. 다음 뉴스 제목을 더 복잡하고 상세하게 발전시켜주세요. 
    원래 제목의 핵심 의미는 유지하세요. 
    이는 뉴스 데이터셋을 위한 것임을 명심하세요.
    다른 문구 없이 변환된 제목만 출력해주세요.
    원본 제목: '{title}'

    변형된 제목:"""
    
    response = model.generate_content(prompt)
    
    # 변형된 제목 저장
    evolved_title = response.text.strip()
    
    return evolved_title

# CSV 파일 읽기
df = pd.read_csv('different_rows.csv')

# 결과를 저장할 리스트
evolved_data = []

# 배치 크기 설정
BATCH_SIZE = 1

# tqdm을 사용하여 진행 상황 표시
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Evolving titles"):
    batch = df.iloc[i:i + BATCH_SIZE]
    
    for _, row in batch.iterrows():
        try:
            evolved_title = evolve_title(row['text'])
            evolved_data.append({
                'ID': f"{row['ID']}_ev",
                'text': evolved_title,
                'target': row['target']
            })
        except Exception as e:
            print(f"Error processing row {row['ID']}: {e}")
            # 에러 발생 시 원본 데이터 유지하되 ID는 변경
            #evolved_data.append({'ID': f"{row['ID']}_ev", 'text': row['text'], 'target': row['target']})

    time.sleep(4)  # API 호출 사이에 짧은 대기 시간 추가

# 결과를 DataFrame으로 변환
evolved_df = pd.DataFrame(evolved_data)

# 결과를 CSV 파일로 저장
output_file = 'evolved_train1.csv'
evolved_df.to_csv(output_file, index=False)
print(f"Evolved data saved to {output_file}")