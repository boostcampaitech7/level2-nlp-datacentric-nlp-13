"""
2024-11-05
gemini 1.5 flash LLM을 이용하여, 가상의 기사 제목 데이터를 생성하는 코드입니다.
"""

import os
import time
import re
import string

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = "AIzaSyBUoSfX70G0b5tew51YU8wPfijEkYvU7Oc"
TARGET_NUM = 6
OUTPUT_FILE = f"{TARGET_NUM}_generated_raw2.csv"

TOTAL_TITLES = 300  # 총 생성할 뉴스 기사 제목 수
NUM_TITLES = 10     # 한 번에 생성할 뉴스 기사 제목 수
MAX_RETRIES = 3     # 최대 재시도 횟수
F1_THRESHOLD = 0.3  # 중복 판단을 위한 F1 Score 임계값

# API 키 설정
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# 모델 초기화
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["num_titles"],
    template=f"""
- 다음 요구사항에 맞게 "{{num_titles}}개의 가상의 뉴스 기사 제목을 생성하세요.

- **제목 생성 지침**:
  - 일반적인 뉴스 기사에서 사용되는 표현과 어휘를 사용하세요.
  - 각 제목은 간결하고, 명확하며, 문법적으로 정확해야 합니다.
  - 구체적인 인물, 지역, 단체 이름을 사용하되, 다양한 이름을 사용해주세요.
  - 예시로 제공한 기사 제목과 유사한 주제의 제목을 생성하세요.

- **출력 형식**:
  - 제목은 아래와 같은 형식으로 출력해주세요:
    1. 첫 번째 제목
    2. 두 번째 제목
    3. 세 번째 제목
    …

- **기사 제목 예시**:
獨 反이스라엘 테러 팔레스타인인 강연 차단…비자도 취소
日도쿄 주택가 공원 연못서 토막시신 발견
폴란드서 이란 겨냥 美주도 60여개국 중동문제회의 시작
美성인 6명 중 1명꼴 배우자·연인 빚 떠안은 적 있다
유엔 리비아 내전 악화·국제적 확산 우려…아랍연맹 긴급회의종합
이스라엘 역사 녹은 희비극 말 한 마리…
북아프리카 스페인령 세우타 이슬람교회당에 괴한들 총격
트럼프가 한사코 부인했는데…멀베이니가 우크라 대가성 인정종합


- **주의사항**:
  - 예시와 동일한 데이터를 절대로 출력하지 마세요.
  - 생성된 제목은 독창적이어야 합니다.
"""
)

def preprocess_text(text):
    """
    텍스트 전처리 함수: 소문자 변환, 구두점 제거, 공백 정리
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(title1, title2):
    """
    두 텍스트 간의 F1 Score 계산 함수
    """
    tokens1 = set(preprocess_text(title1).split())
    tokens2 = set(preprocess_text(title2).split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    common_tokens = tokens1.intersection(tokens2)
    precision = len(common_tokens) / len(tokens1)
    recall = len(common_tokens) / len(tokens2)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def generate_news_titles(num_titles, existing_titles):
    titles = []
    retries = 0
    while len(titles) < num_titles and retries < MAX_RETRIES:
        remaining = num_titles - len(titles)
        current_prompt = prompt_template.format(num_titles=remaining)
        try:
            # 모델을 사용하여 텍스트 생성
            response = model.invoke(current_prompt)
            response_text = response.content.strip()

            # 생성된 제목을 리스트로 분리
            for line in response_text.split('\n'):
                # 번호와 점을 제거하여 실제 제목 추출
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    title = match.group(1).strip()
                    if not title:
                        continue
                    # 중복 검사
                    is_duplicate = False
                    for existing_title in existing_titles + titles:
                        f1 = compute_f1(title, existing_title['text'])
                        if f1 >= F1_THRESHOLD:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        titles.append({"text": title, "target": TARGET_NUM})
                        if len(titles) == num_titles:
                            break
                else:
                    # 예기치 않은 형식의 응답 처리
                    continue

            if len(titles) < num_titles:
                retries += 1
                time.sleep(2)  # 잠시 대기 후 재시도
        except Exception as e:
            retries += 1
            time.sleep(5)  # 잠시 대기 후 재시도

    return titles

def save_to_csv(titles, output_file, start_index):
    data = []
    for i, item in enumerate(titles):
        data.append({
            "id": f"gemini-{start_index + i:05d}",
            "text": item["text"],
            "target": item["target"]
        })
    
    df = pd.DataFrame(data)
    
    # 파일이 이미 존재하면 헤더 없이 추가, 아니면 헤더와 함께 생성
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False, encoding='utf-8-sig')
    else:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"생성된 뉴스 제목 {len(data)}개가 '{output_file}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    print("뉴스 제목 생성을 시작합니다.")
    
    # 기존 OUTPUT_FILE이 존재하면 삭제
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"{OUTPUT_FILE} 기존 파일을 삭제하고 새로 생성합니다.")
    
    existing_titles = []  # 기존 제목 목록 초기화
    total_generated = 0  # 총 생성된 제목 수 초기화
    
    total_batches = (TOTAL_TITLES + NUM_TITLES - 1) // NUM_TITLES  # 총 배치 수 계산
    
    for batch_num in range(total_batches):
        remaining_titles = TOTAL_TITLES - total_generated
        current_batch_size = NUM_TITLES if remaining_titles >= NUM_TITLES else remaining_titles
        
        print(f"배치 {batch_num + 1}/{total_batches}: {current_batch_size}개의 뉴스 제목 생성 중…")
        
        new_titles = generate_news_titles(current_batch_size, existing_titles)
        
        if new_titles:
            start_index = total_generated
            save_to_csv(new_titles, OUTPUT_FILE, start_index)
            existing_titles.extend(new_titles)
            total_generated += len(new_titles)
        else:
            print(f"배치 {batch_num + 1}에서 뉴스 제목 생성에 실패했습니다.")
        
        time.sleep(4)  # 각 배치 사이에 잠시 대기
    
    print("모든 뉴스 제목 생성이 완료되었습니다.")