"""
2024-11-05
gemini 1.5 flash LLM을 이용하여, 가상의 기사 제목 데이터를 생성하는 코드입니다.
"""

import os
import time
import re
import string
import random
import numpy as np

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# 시드 고정 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GEMINI_API_KEY = "AIzaSyAI5-uNfqsa1Br5-uaWBYtgQSkuoXalVag"
TARGET_NUM = 13
OUTPUT_FILE = f"{TARGET_NUM}_generated_raw.csv"

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
    - 다음 요구사항에 맞게 "{"{num_titles}"}개의 가상의 뉴스 기사 제목과 레이블을 생성하세요.
    
    **제목 생성 지침**:
    - 예시 데이터와 유사한 형태의 제목을 생성해주세요.
    - 구체적이고 다양한 이름을 사용하세요.
    
    **레이블 생성 지침**
    - 예시 데이터에서 각 레이블에 해당하는 주제를 파악하세요. 
    - 파악한 주제를 바탕으로, 생성한 제목의 주제에 알맞는 정수 레이블(2, 4, 5, 6 중 하나)을 할당하세요.
    
    **출력 형식**:
    - 제목과 레이블은 쉼표로 구분하여 아래와 같은 형식으로 출력해주세요:
    1. 첫 번째 제목,레이블
    2. 두 번째 제목,레이블
    3. 세 번째 제목,레이블
    ...
    
    **예시 데이터**:
    한국형 발사체 75t 엔진 첫 연소시험 성공,4
    삼성·애플·LG·구글 스마트폰 가을대전 임박,4
    아이폰11 프로 써보니…프로다운 카메라 성능 기대 이상,4
    모바일 컴퓨팅 미래는…서울서 국제학회 ACM 모비시스 개막,4
    줄기세포 이식 파킨슨병에 효과 연구 모식도,4
    스마트 이산화탄소로 화학원료를…잎의 선물 인공광합성,4
    갤럭시S8 덱스 지원하는 최적화 오피스앱 출시,4
    네이버 이미지 검색 고도화…DB 2배 이상 확대,4

    美금융사 가상화폐 경계령…비자 CEO 거래처리 안할 것,5
    코스닥 진입 수월해진다…자본잠식 등 상장요건 개편,5
    KB증권 2분기 영업익 1천5억원…2.21% 증가,5
    테슬라 4분기 연속 흑자…국내 2차전지 업체 수혜 기대,5
    중도금 대출제한도 무색한 청약열기…1순위 마감 줄이어,5
    거래소 ELS 상반기 상장…안정성·환금성 높인다,5
    그리스 조만간 10년물 국고채 발행…구제금융 이후 처음,5
    미래에셋 GS건설 내년 영업익 대폭 증가할 것,5

    네덜란드 정부 보스니아 무슬림 학살사건에 10% 책임,6
    유럽 최악 한파에 난민들 피해속출…폐렴·저체온증 극심,6
    대선출마 앞둔 바이든의 나쁜손 또 폭로…코 비비려고 했다,6
    김정은 베이징 경제기술개발구 제약회사 동인당 공장 방문속보,6
    이란 美 드론의 영공 침범 부인할 수 없는 증거 있어,6
    이탈리아서 전통적 가족가치 옹호 회의…시대착오적 맞불 집회,6
    프랑스 공무원 대규모 감축목표서 후퇴…감원폭 줄이기로,6
    오키나와 민심 달래러 간 아베…돌아가라 야유 당해종합,6
    
    지소미아 종료까지 8일…美 압박 속 고민 깊어지는 文대통령,2
    큰절하는 새누리당 당직자들,2
    민주 5·18 모독 한국당 압박 지속…헌정질서 파괴 옹호,2
    홍준표 황교안 선거권 없어…전대출마 자격 운운 난센스,2
    추미애 검언유착 책임론 정면돌파…윤석열 고립·이성윤 신임종합,2
    문 대통령·여야 지도부 지방선거 사전투표,2
    朴대통령 오바마·아베와 연쇄통화…北미사일 대응책 논의,2
    황교안·나경원 패스트트랙 사건 경찰 출석통보에 불응 입장,2
    
    **주의사항**
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