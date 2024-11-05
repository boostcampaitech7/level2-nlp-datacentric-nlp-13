import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import os
import evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=7)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, '../outputs')

# 데이터 로드
clean_data = pd.read_csv(os.path.join('text_cleaned.csv'))
noise_data = pd.read_csv(os.path.join('label_noise.csv'))

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)

# 데이터 분할 (clean 데이터만 사용)
dataset_train, dataset_valid = train_test_split(clean_data, test_size=0.2, random_state=SEED)

# 데이터셋 및 데이터로더 생성
data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

os.environ['WANDB_DISABLED'] = 'true'

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    logging_strategy='steps',
    eval_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-05,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()
print("Fine-tuning 완료")

# 임베딩 추출 함수
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    model.eval()
    model.config.output_hidden_states = True  # hidden states 활성화
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="텍스트 임베딩 중"):
        batch = texts[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=256)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # 마지막 은닉층에서 [CLS] 토큰 벡터 추출
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
    # 임베딩 벡터 정규화 (L2 정규화)
    normalized_embeddings = normalize(np.vstack(embeddings), norm='l2')
    return normalized_embeddings

# 클린 데이터와 노이즈 데이터의 임베딩 추출
clean_embeddings = get_embeddings(clean_data['text'].tolist(), model, tokenizer, device)
noise_embeddings = get_embeddings(noise_data['text'].tolist(), model, tokenizer, device)

# RandomForest 분류기 학습 (전체 clean 데이터 사용)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_classifier.fit(clean_embeddings, clean_data['target'])

###
# 노이즈 데이터에 대한 예측 확률 계산
predicted_probs = rf_classifier.predict_proba(noise_embeddings)

# 최대 확률이 alpha 미만인 샘플 필터링
alpha = 0.7  # 예를 들어, alpha를 0.7로 설정
max_probs = np.max(predicted_probs, axis=1)  # 각 샘플의 최대 확률 계산

# 최대 확률을 계산하여 max_probs 열로 추가
noise_data['max_probs'] = np.max(predicted_probs, axis=1)
noise_data.to_csv('max_probs.csv', index=False)

# filtered_indices = np.where(max_probs >= alpha)[0]  # alpha 이상인 샘플의 인덱스

# # 노이즈 데이터에서 필터링된 샘플만 남기기
# filtered_noise_data = noise_data.iloc[filtered_indices].copy()

# # 필터링된 데이터의 라벨을 모델의 예측 라벨로 교체
# filtered_noise_data['corrected_target'] = rf_classifier.predict(noise_embeddings[filtered_indices])

# # 클린 데이터와 필터링된 노이즈 데이터 결합
# combined_data = pd.concat([
#     clean_data[['ID', 'text', 'target']], 
#     filtered_noise_data[['ID', 'text', 'corrected_target']].rename(columns={'corrected_target': 'target'})
# ], ignore_index=True)

# # 결과 저장
# output_path = 'train_filtered.csv'
# combined_data.to_csv(output_path, index=False, columns=['ID', 'text', 'target'])
# print(f"\n결합된 데이터가 {output_path}에 저장되었습니다.")

###

# 노이즈 데이터에 대한 예측
predicted_labels = rf_classifier.predict(noise_embeddings)

# 노이즈 데이터의 라벨 교정
noise_data['corrected_target'] = predicted_labels

# 클린 데이터와 교정된 노이즈 데이터 합치기
combined_data = pd.concat([
    clean_data[['ID', 'text', 'target']], 
    noise_data[['ID', 'text', 'corrected_target']].rename(columns={'corrected_target': 'target'})
], ignore_index=True)

# 결과 저장
output_path = 'train.csv'
combined_data.to_csv(output_path, index=False, columns=['ID', 'text', 'target'])
print(f"\n결합된 데이터가 {output_path}에 저장되었습니다.")