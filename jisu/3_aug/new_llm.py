import pandas as pd
import os
import time
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_fixed, stop_after_attempt

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyBsT-yVjamCVUdVcHQ8hps_T0WGMWbX-N4"

# 모델 초기화
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["noisy_text"],
    template="""
    다음은 뉴스 제목입니다:
    "{noisy_text}"
    이 제목의 의미를 유지하면서 약간 다른 표현으로 뉴스 제목을 새롭게 만들어줘.
    다른 표현을 사용하면서도 뉴스처럼 간결하고 핵심이 잘 드러나게 만들어줘.
    다른 문구 없이 변환된 제목만 출력해줘.
    """
)

# 데이터 불러오기
data = pd.read_csv('noisy_train.csv')

# 데이터 증강 함수 정의
@retry(wait=wait_fixed(3), stop=stop_after_attempt(5))  # 3초 대기, 최대 5회 재시도
def augment_text(text):
    prompt = prompt_template.format(noisy_text=text)
    response = model.invoke(input=prompt)
    return response.content.strip()

# 증강된 텍스트를 저장할 리스트
augmented_texts = []

# 데이터 증강 수행 (tqdm을 사용하여 진행 상태 표시)
for idx, row in tqdm(data.iterrows(), total=len(data), desc="데이터 증강 진행"):
    noisy_text = row['text']
    try:
        clean_text = augment_text(noisy_text)
        augmented_texts.append(clean_text)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        augmented_texts.append(noisy_text)  # 오류 발생 시 원본 텍스트 사용

    # API 사용 제한 대기 시간을 충분히 확보
    time.sleep(5)

# 증강된 텍스트를 데이터프레임에 추가
data['augmented_text'] = augmented_texts

# 증강된 데이터 저장
output_path = 'train_augmented.csv'
data.to_csv(output_path, index=False)
print(f"데이터 증강 완료. '{output_path}'에 저장되었습니다.")
data