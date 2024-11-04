from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
model_name = "lcw99/t5-large-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 테스트할 문장
test_text = "삼성증권 1분기 영업이익 1천496억원…17% 감소"

# Paraphrasing 함수 정의
def generate_paraphrase(text, model, tokenizer, max_length=50):
    input_text = f"뉴스 기사 제목처럼 간결하게 표현해 주세요: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

# Paraphrasing 수행
paraphrased_text = generate_paraphrase(test_text, model, tokenizer)
print("원문:", test_text)
print("Paraphrased:", paraphrased_text)
