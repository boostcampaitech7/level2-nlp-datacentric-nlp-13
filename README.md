# Level 2 Project :: 주제 분류 프로젝트

### 📝 Abstract
- 이 프로젝트는 네이버 부스트 캠프 AI-Tech 7기 NLP Level 2 기초 프로젝트 경진대회로, Dacon과 Kaggle과 유사한 대회형 방식으로 진행되었다.
- Data-Centric Topic Classification는 주어진 뉴스 헤드라인으로부터 해당 뉴스의 주제를 `0`~`6`의 정수 레이블로 분류하는 것으로, 모델 구조의 변경 없이 Data-Centric 관점으로 해결해보았다.

<br>

## Project Leader Board 
- Public Leader Board
<img width="700" alt="public_leader_board" src="https://github.com/user-attachments/assets/d66e3034-0f85-4128-badd-efa221d68436">

- Private Leader Board 
<img width="700" alt="private_leader_board" src="https://github.com/user-attachments/assets/0fe1d855-1ac2-4eb2-b7b0-121dc5f13ce9">

- [📈 NLP 13조 Project Wrap-Up report 살펴보기](https://github.com/user-attachments/files/17536172/NLP_13.Wrap-Up.pdf)

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 스빈라킨스배 [ NLP 13조 ]

### 👨🏼‍💻 Members
권지수|김성은|김태원|이한서|정주현|
:-:|:-:|:-:|:-:|:-:
 <img src='https://github.com/user-attachments/assets/ab4b7189-ec53-41be-8569-f40619b596ce' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/49dc0e59-93ee-4e08-9126-4a3deca9d530' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/a15b0f0b-cd89-412b-9b3d-f59eb9787613' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/11b2ed88-bf94-4741-9df5-5eb2b9641a9b' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/3e2d2a7e-1c64-4cb7-97f6-a2865de0c594' height=125 width=100></img>
[Github](https://github.com/Kwon-Jisu)|[Github](https://github.com/ssungni)|[Github](https://github.com/chris40461)|[Github](https://github.com/beaver-zip)|[Github](https://github.com/peter520416)
<a href="mailto:wltn80609@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:sunny020111@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:chris40461@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:beaver.zip@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:peter520416@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

### 🧑🏻‍🔧 Members' Role

| 이름 | 역할 |
| :---: | --- |
| **`권지수`** | **Text/Label noise split**, **Text cleaning**, **Prompt-based Generation** |
| **`김성은`** | **Text/Label Noise split**, **Back-translation**(Google Translate), **Prompt-based Generation** |
| **`김태원`** | **EDA**, **Re-labeling**, **Evol-Instruct LLM for Augmentation**  |
| **`이한서`** | **Text/Label noise split**, **Text cleaning**, **Back-translation**(DeepL), **Prompt-based Generation** |
| **`정주현`** | **Re-labeling**, **ML model searching** |

<br>

## 🖥️ Project Introduction 


| **프로젝트 주제** | 주어진 뉴스 헤드라인으로부터 해당 뉴스의 주제를 분류하되, 모델 구조의 변경 없이 Data-Centric 관점으로 해결|
| :---: | --- |
| **프로젝트 구현내용** | LLM을 활용해 데이터의 텍스트 노이즈와 라벨 노이즈를 분류한 뒤, LLM으로 텍스트를 정제하고 Cleanlab으로 라벨을 정제했다. 그 후 한국어-일본어로 역번역하여 데이터를 증강하고, 중복 데이터를 제거하여 최적의 학습 데이터를 생성한 뒤 분류를 진행했다.|
| **개발 환경** |**• `GPU` :** Tesla V100(32G RAM) 서버 4개 <br> **• `개발 Tool` :** Jupyter notebook, VS Code [서버 SSH연결]|
| **협업 환경** |**• `Github Repository` :** 데이터, 코드 버전 관리 및 작업 상황 공유 <br>**• `Notion` :** 역할 분담, 실험 가설 및 검증 결과 공유 <br>**• `SLACK, Zoom` :** 실시간 비대면 회의|

<br>

## 📁 Project Structure

### 🗂️ 디렉토리 구조 설명

**Code**
- Text/Label Noise Split: `filtering.ipynb`
- Text/Label Cleaning, Back-Translation, Baseline: `/code`

**Data**
- Noise/Cleaned: `/split_train_data`
- Back-Translation: `/backtranslation_data`
- train: `/data`

### 📄 코드 설명

- `filtering.ipynb`: `train.csv`에서 Text/Label Noise를 구분해 `text_noise.csv`, `labe_noise.csv`를 생성
- `text_clean.py`: `text_noise.csv`의 text를 정제한 `text_cleaned.csv`를 생성
- `correct_label.py`: `label

- filtering -> text_clean -> correct_label -> total_clean -> backtranslate_DeepL_JP -> postprocess_and_merge -> baseline_code

```
📂 backtranslation_data
┃ ┗  backtranslated_DeepL_JP.csv
📂 code
┃ ┣  backtranslate_DeepL_JP.py
┃ ┣  baseline_code.ipynb
┃ ┣  correct_label.py
┃ ┣  postprocess_and_merge.py
┃ ┣  text_clean.py
┃ ┗  total_clean.py
📂 data
┃ ┗  train.csv
📂 split_train_data
┃ ┣  label_noise.csv
┃ ┣  merge_text_label_cleaned.csv
┃ ┣  text_cleaned.csv
┃ ┣  text_noise.csv
┃ ┗  total_cleaned.csv
┣ filtering.ipynb
┗ requirements.txt
 ```
<br>

## 📐 Project Ground Rule
>팀 협업을 위해 프로젝트 관련 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 규칙을 정했으며, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획하여 진행했다.

**- a. `Server 관련`** : 권지수, 김성은, 이한서, 정주현 캠퍼는 각자 서버를 생성해 모델 실험을 진행하고, 김태원 캠퍼는 서버가 유휴 상태인 서버에서 실험을 진행했다.

**- b. `Git 관련`** : exp branch에 각자 폴더를 생성해 작업하였고, 공통으로 사용할 파일은 main에 push 하는 방법으로 협업했다.

**- c. `Submission 관련`** : 대회 마감 5일 전까지는 자유롭게 제출했고, 5일 전부터는 인당 2회씩 분배했다.

**- d. `Notion 관련`** : 원활한 아이디어 브레인스토밍과 분업을 위해 회의를 할 경우 노션에 기록하며, 연구 및 실험결과의 기록을 공유했다.

<br>

## 🗓 Project Procedure: 총 2주 진행
- **`(1~5 일차)`**: 기본 Baseline format 해석 및 script 구현
- **`(6~12 일차)`**: 데이터 EDA 및 구조 파악, 데이터 전처리, MRC 모델 탐색
- **`(12~20 일차)`** : MRC 모델 하이퍼 파라미터 튜닝(wandb), Dense Retriever 구현
- **`(20~25 일차)`** : Dense Retriever 과 Sparse Retriever 을 사용한 Hybrid Retriever 구현
- **`(26~28 일차)`** : 앙상블 진행 

<br>

## **MRC**
* 우리는 먼저 Retriever - Reader 모델을 구현하기에 앞서, KorQuad data 에 대해서 pre-trained 된 모델을 사용해, 부족한 Train dataset 을 보강하여 학습하기로 하였다.
<br>

## **Retriever**
* Retriever 모델의 경우, KorQuad data 를 통해 question, passage embedding 을 미리 학습하는 과정을 가진 Dense Retriever 과, BM25 를 사용한 Sparse Retriever 을 결합한 Hybrid Retriever 을 사용했다.
* Hybrid Retriever 을 하는 방식은 크게 3가지 인데, 이 3가지를 모두 활용하여 passage 와 question 의 연관성을 최대로 학습하고, 사용하고자 하였다.

| **Score Function Type**                       | **Description**                                                                                        |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **1. λ x Dense Similarity + BM25 Score** | Dense Retriever와 BM25 점수를 결합하여 질문과 문서 간의 관계를 강화합니다. λ는 두 점수 간의 가중치를 조정합니다. |
| **2. Reciprocal Rank Fusion (RRF)**      | 여러 Retriever의 랭킹을 기반으로 각 문서의 reciprocal rank를 합산하여 최종 점수를 계산합니다. |
| **3. (λ x Dense Similarity + BM25 Score) → ko-reranker** | 1번의 Score 를 한국어 re-ranker에 입력하여 최종 랭킹을 정제하고 성능을 향상시킵니다.     |
<br>

## **Ensemble Model**

* 최종적으로 3개의 json 파일을 softvoting 기법을 활용하여 사용했다.

| **File Name**                             | **Score Function**                                   |
|-------------------------------------------|-----------------------------------------------------|
| nbest_predictions_op1.json               | λ x Dense Similarity + BM25 Score                   |
| nbest_predictions_op2.json               | Reciprocal Rank Fusion (RRF)                        |
| nbest_predictions_op3.json               | (λ x Dense Similarity + BM25 Score) → ko-reranker   |

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
# 필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️ How To Train & Test
```bash
# label_noise.csv, text_noise.csv 파일 생성
filtering.ipynb

# train.sh 코드 실행 : MRC 를 위한 Train dataset 에 대한 script file 실행
chmod +x train.sh # 권한 추가
./train.sh

# test.sh 코드 실행 : Retriever 을 사용해서 ODQA task 수행
chmod +x test.sh # 권한 추가
./test.sh

# 이후, nbest_prediction.json 들이 ./nbest 에 저장됨
```

### ⌨️ How To Ensemble
```bash
# ./nbest 에 생성된 json 파일들을 모두 확률 값에 따라서 soft voting 하여 ensemble
python3 softvoting.py
```
