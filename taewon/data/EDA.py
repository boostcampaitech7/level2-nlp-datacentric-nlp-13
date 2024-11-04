import pandas as pd
import random
from konlpy.tag import Mecab
from tqdm import tqdm

# train.csv 파일 읽기
df = pd.read_csv('train.csv')

mecab = Mecab(dicpath='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic')


# 무작위 교환 함수
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            return new_words
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

# 무작위 삭제 함수
def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    if len(new_words) == 0:
        return [random.choice(words)]
    return new_words

# EDA 적용 함수
def eda(sentence, alpha_rs=0.1, p_rd=0.1, num_aug=1):
    words = mecab.morphs(sentence)
    num_words = len(words)
    
    augmented_sentences = []
    
    for _ in range(num_aug):
        a_words = words.copy()
        
        # 무작위 교환
        n_rs = max(1, int(alpha_rs * num_words))
        rs_words = random_swap(a_words, n_rs)
        augmented_sentences.append(' '.join(rs_words))
        
        # 무작위 삭제
        rd_words = random_deletion(a_words, p_rd)
        augmented_sentences.append(' '.join(rd_words))
    
    return augmented_sentences

# 데이터 증강 적용
augmented_data = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="데이터 증강 중"):
    augmented_sentences = eda(row['text'])
    for aug_sentence in augmented_sentences:
        augmented_data.append({
            'ID': row['ID'],
            'text': aug_sentence,
            'target': row['target']
        })

print("데이터 증강 완료")

# 증강된 데이터를 DataFrame으로 변환
augmented_df = pd.DataFrame(augmented_data)

# 원본 데이터와 증강된 데이터 합치기
final_df = pd.concat([df, augmented_df], ignore_index=True)

# 결과 저장
final_df.to_csv('augmented_train.csv', index=False)
print("증강된 데이터가 'augmented_train.csv'에 저장되었습니다.")