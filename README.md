# TFP-Bert-Hate

## 오픈소스 팀 과제- BERT 활용한 혐오표현 탐지 및 분류<박서은>
[Try in colab](https://colab.research.google.com/drive/11xnWLBdngWq77dVlcZqNI4rDrdVnM4a4#scrollTo=958f2K1st_pG)
### 1.1 사용한 데이터
- A Multi-label Hate Speech Detection 데이터셋(K-MHaS)
- https://github.com/adlnlp/K-MHaS
### 1.2 데이터셋 구조
- 한국 온라인 뉴스 댓글 의 109,692개 발화로 구성되며 , 8개의 세분화된 라벨로 분류됩니다.
- (a) 이진 분류 및 (b) 다중 라벨(1~4) 분류를 제공합니다.

#### 다중라벨 주석

two layers: 
- **(a) 이진** classification (`Hate Speech` or `Not Hate Speech`) and 
- **(b) 세부라벨** classification. 
    - `Politics`(정치성향차별)
    - `Origin`(출신차별)
    - `Physical`(외모차별)
    - `Age`(연령차별)
    - `Gender`(성차별)
    - `Religion`(종교차별)
    - `Race`(인종차별)
    - `Profanity`(혐오욕설)
 
#### Dataset Details

label types  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8
----------|-----------|-----------|-----------|----------|----------|---------|---------|---------|-----------
`En` |  `Origin` | `Physical` | `Politics` | `Profanity` | `Age` | `Gender` | `Race` | `Religion` | `Not Hate Speech`
`Kr` |  `출신차별` | `외모차별` | `정치성향차별` | `혐오욕설` | `연령차별` | `성차별` | `인종차별` | `종교차별` | `해당사항없음`


```
document	label
좌좀 홍어들이 일베보다 더 극혐이다	0,2
와 이거 완전 재앙이네. 깨끗한 나라 만든다면서 왜 이런 사건이 터지냐??	2
몬생긴 여자인간들은 찌그러져있어라 확그냥 막그냥	1,5
오면 화염병 던진다 돼지새키야	1,3
수꼴틀딱시키들이 다 디져야 나라가 똑바로 될것같다..답이 없는 종자들ㅠ	2,3,4
ㅇㅇ? 양놈이건 깜씨건 뭔 아방가르드하네	0,6
OOO천지 개독교 새끼들아 다 착해져라!	3,7
다들 오지랖 좀...되게 할 짓 없어보여요	8
```


## 2. 모델학습
### 2.1 Dataset loading
- huggingface에서 제공하는 transformer 라이브러리 사용
  
### 2.2 Data preparation- 
- 학습, 검증, 테스트 데이터 세트에서 데이터 준비

### 2.3 Prep for Pytorch
- tokenizing

### 2.4 Multi-BERT model
- GPU setting
- Model setting
- Model training
- Evaluation
- Break down evaluation
- test

## 실험
- 6가지 지표(Accuracy, F1-[macro, micro, weighted], AUC, Hamming Loss) 사용하여 평가

### BERT 전반적 성능
![BERT전반적성능(표)](https://github.com/TwitFilterProject/TFP-Bert-Hate/assets/165137301/93354dd3-eb0d-4e9d-b0b7-9fadda823587)

### Multi-label 분류 성능(label 1~4)
![Multilabel분류성능(표)](https://github.com/TwitFilterProject/TFP-Bert-Hate/assets/165137301/a1776eb0-edb6-4c7e-95df-135f9c6f4a71)


## 테스트
- precision: 정밀도, 모델이 양성으로 예측한 것 중 실제로 양성인 비율
- recall: 재현율, 실제 양성인 것 중 모델이 양성으로 예측한 비율
- f1-score: F1 점수, 정밀도와 재현율의 조화 평균
- support: 각 클래스에 속하는 실제 샘플 수
#### 테스트결과
![test결과](https://github.com/TwitFilterProject/TFP-Bert-Hate/assets/165137301/c411b859-fa5a-4164-97a9-95c61c33f85f)
