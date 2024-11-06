import pandas as pd

# CSV 파일을 읽어옵니다. 파일 경로는 'file_path' 변수에 입력하세요.
file_path = 'train_augmented_processed.csv'  # 파일 경로를 실제 파일 위치로 수정하세요.
data = pd.read_csv(file_path)

# 텍스트 내의 콤마를 공백으로 변경하는 함수
def replace_commas_with_space(text):
    return text.replace(',', ' ')

# 맨 앞과 맨 뒤의 쌍따옴표를 제거하는 함수
def remove_surrounding_quotes(text):
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text

# 각 함수를 차례대로 적용
data['text'] = data['text'].apply(replace_commas_with_space)
data['text'] = data['text'].apply(remove_surrounding_quotes)

# 결과를 확인하고 파일로 저장
data.to_csv('cleaned_file.csv', index=False)
