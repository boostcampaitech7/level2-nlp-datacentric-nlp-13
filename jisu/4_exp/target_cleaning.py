import pandas as pd
import torch
from transformers import pipeline

# 데이터 불러오기
df = pd.read_csv("recent_output.csv")

# GPU 설정 (CUDA가 사용 가능한지 확인)
device = 0 if torch.cuda.is_available() else -1

# 텍스트 생성 모델 초기화 (한국어 생성 모델 사용, GPU 사용 설정)
generator = pipeline("text-generation", model="skt/kogpt2-base-v2", max_new_tokens=30, device=device)

# 타겟 별 주제 예측
target_summary = {}

for target in sorted(df['target'].unique()):
    # 해당 target의 텍스트 데이터 추출
    texts = df[df['target'] == target]['text'].tolist()
    
    # 첫 번째 텍스트 데이터만 샘플로 사용
    text_sample = texts[0]
    
    # 역할 부여 프롬프트 구성
    prompt = (
        f"다음 조건에 맞게 '{text_sample}'라는 뉴스 제목의 주제를 추론하시오.\n"
        "    조건1: text는 뉴스 기사 제목입니다.\n"
        "    조건2: target은 0, 1, 2, 3, 4, 5, 6 중 하나입니다.\n"
        "    조건3: target이 같으면 뉴스 제목의 주제가 같습니다.\n"
        "주제:"
    )
    
    # 텍스트 생성
    response = generator(prompt, num_return_sequences=1)
    predicted_topic = response[0]['generated_text'].split("주제:")[-1].strip()
    
    target_summary[target] = predicted_topic

# 최종 결과 출력
print("타겟별 예측 주제 요약:")
for target, topic in target_summary.items():
    print(f"Target {target}: {topic}")


