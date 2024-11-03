"""
gemini 1.5 flash LLM을 이용하여, filtering 결과로 생성된 text_noise.csv의 text를 정제하는 코드입니다.
정제 결과는 cleaned_data.csv에 저장되며, post_process.py를 통해 후처리해야 합니다.
"""
import pandas as pd
import os
import time
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# API 키 설정
os.environ["GOOGLE_API_KEY"] = " "

# 데이터 로드
data_file = 'text_noise.csv'
data = pd.read_csv(data_file)

# 모델 초기화
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["noisy_text"],
    template="""
    다음은 노이즈가 포함된 텍스트입니다:
    "{noisy_text}"
    이 텍스트에서 의미를 유지하면서 노이즈를 제거하고, 자연스럽고 뉴스 제목처럼 간결하게 변환해줘.
    다른 문구 말고 오직 변환된 텍스트만을 출력해줘.
    """
)

# 결과를 저장할 리스트
results = []

# 배치 크기 설정
batch_size = 10
total_rows = len(data)

# 저장을 위한 파일명
output_file = 'cleaned_data.csv'

# 기존 결과가 있으면 불러오기
if os.path.exists(output_file):
    existing_results = pd.read_csv(output_file)
    processed_ids = set(existing_results['ID'])
    results = existing_results.to_dict('records')
    print(f"기존에 {len(processed_ids)}개의 데이터를 처리했습니다. 이어서 진행합니다.")
else:
    processed_ids = set()
    print("처음부터 시작합니다.")

# 배치 처리
for start_idx in range(0, total_rows, batch_size):
    end_idx = min(start_idx + batch_size, total_rows)
    batch = data.iloc[start_idx:end_idx]

    for index, row in batch.iterrows():
        id = row['ID']
        if id in processed_ids:
            continue  # 이미 처리된 데이터는 건너뜁니다

        noisy_text = row['text']
        target = row['target']

        # 프롬프트 생성
        prompt = prompt_template.format(noisy_text=noisy_text)

        success = False
        retry_count = 0
        max_retries = 5

        while not success and retry_count < max_retries:
            try:
                # 모델 예측
                response = model.invoke(prompt)

                # AIMessage 객체에서 실제 텍스트 추출
                cleaned_text = response.content.strip()

                # 결과 저장
                results.append({
                    "ID": id,
                    "Original Text": noisy_text,
                    "Cleaned Text": cleaned_text,
                    "target": target
                })

                processed_ids.add(id)
                print(f"ID {id} 처리 완료.")
                success = True

                # 요청 사이에 지연 시간 추가
                time.sleep(2)  # 지연 시간을 늘려서 API 부하를 줄입니다

            except Exception as e:
                retry_count += 1
                print(f"ID {id} 처리 중 오류 발생: {e}")
                if 'ResourceExhausted' in str(e):
                    wait_time = 30  # 쿼터 초과 시 1분 대기
                    print(f"{wait_time}초 후에 재시도합니다.")
                    time.sleep(wait_time)
                else:
                    print(f"10초 후에 재시도합니다.")
                    time.sleep(10)

        if not success:
            print(f"ID {id} 처리를 건너뜁니다.")
            # 실패한 경우에도 결과에 추가 (빈 문자열로)
            results.append({
                "ID": id,
                "Original Text": noisy_text,
                "Cleaned Text": "",
                "target": target
            })
            processed_ids.add(id)

    # 중간 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{end_idx}/{total_rows}개 데이터 처리 완료. 중간 결과를 저장했습니다.")

print(f"모든 데이터 처리가 완료되었습니다. 결과는 '{output_file}'에 저장되었습니다.")
