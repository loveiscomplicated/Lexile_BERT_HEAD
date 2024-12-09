1. data preparing
2. data preprocessing
3. build a model
4. train a model
5. test the accuracy
    statictical method


1. data preparing
    카글에서 가져옴
    https://www.kaggle.com/datasets/yuewang123/dataset-from-commonlit-website-with-full-text

    나중에 이것도 추가해봐야할듯
    https://github.com/isaacsarver/Lexile-Score-Predictor/blob/master/Book%20Files/19.txt 




2. data preprocessing
data source = commonlit, eannefawcett

commonlit
특징 - 결측치, 너무 짧은 글 포함하고 있음

진행 과정


commonlit 데이터랑 eannefawcett 데이터 합치기
    commonlit 데이터 결측치 제거
    eannefawcett의 구성이랑 commonlit의 구성을 같게 만들기
    이 둘이 합치기

1. 글들이 짧은 것들이 있고, 긴 것들이 있음 - 이것들의 길이를 조정할 예정
    처음에 단어 개수를 기준으로 조정하려고 했으나
    어차피 모델은 토큰의 관점에서 볼 것이라
    토큰화 후 토큰의 개수를 기준으로 데이터 조정할 예정


-- 토큰의 길이가 너무 긴 경우와 너무 짧은 경우로 나눌 수 있음


2. 너무 긴 경우 - 512개 토큰을 최대 길이로 하고 텍스트 분할 
    - 코랩 Lexile data preprocessing.ipynb에서 수행
    - 512개 토큰 넘어가면 쪼개기 
3. 너무 짧은 경우 
    2번 이후 토큰 길이의 분포 파악, 최소 토큰 개수를 얼마로 할지 정하기
    최소 토큰 개수를 정해서 미달인 데이터 제거
4. 512개 기준으로 패딩

5. 렉사일 지수 분포 파악
6. 최대한 균등하게 배치 후 렉사일 지수 레이블을 범주 개념으로 바꾸기
    6개 구간으로 percentile 기준으로 나누니까 좋은 듯
    

하면 끝...!!


3. build a model
BERT의 Head 방식에 대한 접근
Head 활용의 기본 개념

BERT는 여러 Attention Head로 구성된 Self-Attention 레이어를 사용합니다.
각 Head는 입력 문장에서 다른 종류의 관계나 특징(예: 문맥적 연관성, 문장 구조)을 학습합니다.
특정 Head의 출력을 활용하거나, 여러 Head를 결합하여 문장 표현을 생성할 수 있습니다.
Head 활용 방식

**전체 Head 결합: 모든 Head의 출력을 결합(평균 또는 가중합)하여 문장의 총체적인 표현으로 사용.
**특정 Head 선택: 특정 Attention Head가 중요한 정보를 잘 캡처한다면 해당 Head의 출력을 선택적으로 사용.
Multi-Head Attention 추가: 기존 BERT 출력에 새로운 Attention Head를 추가로 학습시켜 특정 작업에 맞춘 문장 표현을 생성.


장점

BERT의 각 Head가 학습한 다양한 문맥적 관계를 효과적으로 활용할 수 있음.
문장의 세부 정보와 전체 문맥 모두를 반영 가능.


적용 시 고려 사항

Head의 출력 크기와 가중치 결합 방식은 실험적으로 최적화해야 함.
계산 비용이 다소 증가할 수 있으므로 효율적인 결합 방식이 필요.
--> GPT4의 답변


우리가 지금 진행하고자 하는 방식은 BERT의 파라미터를 고정시키고, FC layer만 학습하는 방식이므로
BERT에 데이터를 통과시킨 후 결과 데이터를 가지고 FC layer를 학습시키는 데 집중할 것임

원래 하던 방식은, 토큰화된 데이터를 BERT와 FC layer 모두 통과시키는 방법으로 학습했음
각 에포크마다 데이터를 BERT에 다시 통과시켜야 하기 때문에 효율이 떨어짐

1. 토큰화된 데이터를 BERT에 입력한다
2. BERT에서 출력된 결과물을 가지고 FC layer를 학습시킨다.
3. FC layer에만 집중할 수 있으므로, 
    에포크 수를 늘리던가(BERT가 포함되면 에포크 수는 3~5정도까지밖에 안 됨) 여러가지 방법을 가지고
    정확도와 효율을 동시에 잡을 것
4. 코랩의 Lexile data preprocessing.ipynb에서 진행할 예정



----------------------------------------------------------------
3. build a model
4. train the model
5. test the model

사용한 방법론들

1. 데이터 정규화 - BERT에서 출력된 결과물을 FC layer에 입력하기 전에 먼저 정규화를 시킴
                정규화는 (편차/표준편차)로 계산 
                표준편차가 0이 되거나, 너무 심하게 compression되는 것을 막기 위해 분모에 미세한 수 (1e-8) 더함

2. train_test_split

3. FC layer에 입력하기 위해 데이터 탠서화

4. TensorDataset으로 데이터 묶기 + DataLoader(batch_size = 16)로 TensorDataset을 불러오기

5. 모델 설계
    입력 레이어 - 768차원

    첫 번째 FC Layer - 768차원 to 512차원
    활성화 함수 1 (Relu)
    드롭아웃 1 Dropout(p=0.1)

    두 번째 FC Layer - 512차원 to 256차원
    활성화 함수 2 (Relu)
    드롭아웃 2 Dropout(p=0.1)

    세 번째 FC Layer - 256차원 to 128차원
    활성화 함수 3 (Relu)
    드롭아웃 3 Dropout(p=0.1)

    네 번째 FC Layer - 128차원 to 6차원 (실제로 분류할 범주의 개수)
    LayerNorm 추가
        네 번째 FC Layer의 결과물인 Logits 값의 최댓값과 최솟값의 차이가 20을 넘어가면
        softmax 함수 통과할 때 비정상적인 값 출력
        따라서 비정상적으로 큰 loss 값을 가지게 됨
        최댓값과 최솟값의 차이를 줄이기 위해 마지막에 LayerNorm 추가하는 것

    맨 끝에 softmax를 추가해야 하는 것 아닌가요?
        crossEntropy loss는 이렇게만 놔둬도 내재된 softmax 활용하여 알아서 loss 값 계산
        나중에 실제로 사용할 때에는 softmax를 뒤에 덧붙여야 함

6. 모델 훈련
    learning rate = 0.0001
    epochs = 150
    early stopping 적용 (patience = 10)
    loss function = crossEntropy
    sceduler 활용하여 learning rate를 loss에 따라 자동적으로 적절히 적용
    Xavier initialization 적용하여 initialization도 최대한 잘 되게 적용
    중요 !!!! loss 계산이 평균으로 되어 있는지 확인해야 함 !!!! loss를 단순합으로 계산하고 있었음
        그러니 loss가 엄청 크게 나오지...
        

7. 모델 평가 및 저장
    accuracy = correct / total로 계산

    




