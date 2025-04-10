[Google Colab](https://colab.research.google.com/drive/1HMTmibRVkgkuNoz4Vy2h_rARdEIlgXd3?usp=sharing)

![image.png](attachment:ef4f9f47-0a10-426b-9dd0-a1eb2c46e41a:image.png)

### 📌 6.1. 이미지 분류를 위한 신경망

**6.1.4. GoogLeNet**

- 하드웨어 자원 효율적 이용, 학습 능력 극대화
- **인셉션(Inception) 모듈**
    - 1x1 합성곱
    - 1x1 합성곱 + 3x3 합성곱
    - 1x1 합성곱 + 5x5 합성곱
    - 3x3 최대 풀링 + 1x1 합성곱
    
    ![image.png](attachment:5a2b3ffd-c1ad-4740-8371-97ba682cbf35:image.png)
    
- **희소 연결(Sprase Connectivity)** : 합성곱, 풀링, 완전연결층이 밀집 → 관련성 높은 노드만 연결 (적은 연산량, 과적합 해결)

**6.1.5. ResNet**

- 신경망은 깊이가 깊어질수록 성능이 좋아지다가 일정한 단계부터 오히려 성능이 나빠짐 → VGG19 구조 + 합성곱층, 숏컷 추가
- **레지듀얼 블록(Residual Block)** : 깊어진 신경망을 효과적으로 학습, 기울기가 잘 전파될 수 있도록 일종의 **숏컷(Shortcut, Skip Connection)**을 만듦
    - **병목 블록(Bottleneck Block)** : 깊이가 깊어짐에 따라 파라미터가 무제한으로 커지는 것을 방지 → **1x1 합성곱층**을 추가하고 채널 수를 조절
    - **아이덴티티 매핑(Identity Mapping,** Shortcut, Skip Connection) : 입력 형태 그대로 (기존/병목 블록 모두에서 사용됨)
    - **다운샘플(Downsample)** : 특성 맵의 크기를 줄임 (풀링과 같은 역할, 아이덴티티 매핑을 수행하기 위해 형태를 맞추기)
        - 아이덴티티 블록 : 입력 차원 = 출력 차원
        - 프로젝션 숏컷, 합성곱 블록 : 입력 차원 ≠ 출력 차원
    - 각 레지듀얼 분기(Residual Branch)에 있는 마지막 BN(Batch Normalization)을 0으로 초기화 → 성능 향상
    
    ![image.png](attachment:2cbec51e-2846-48fd-b457-5ae73c4117dc:image.png)
    
    ![image.png](attachment:67be59c3-0736-4bc3-83c0-f23315013691:image.png)
    

### 📌 6.2. 객체 인식을 위한 신경망

**6.2.1. R-CNN**

- **객체 인식(Object Detection)** : 이미지/영상 내의 객체 식별
    - ① 객체가 무엇인지 분류
    - ② 객체의 위치 정보 검출(Localization) (객체의 위치를 Bounding Box로 표시)
    - **1단계 객체 인식(1-Stage Detector)** : 분류 + 위치 검출 동시 (빠르지만 낮은 정확도, YOLO 계열과 SSD 계열)
    - **2단계 객체 인식(2-Stage Detector)** : 위치 검출 → 분류 (느리지만 높은 정확도, R-CNN 계열)
- **슬라이딩 윈도우(Sliding Window) 방식** : 일정한 크기의 윈도우(Window)로 이미지의 모든 영역을 탐색하며 객체를 검출 → 비효율적
- **선택적 탐색(Selective Search) 알고리즘** : **후보 영역(Region Proposal)**을 알아내는 방법
    - 초기 영역 생성(이미지를 다수의 영역으로 분할) → 작은 영역 통합(Greedy 알고리즘, 비슷한 영역으로 통합) → 후보 영역 생성(Bounding Box 추출)
    - 시드(Seed, 특정 기준점) 선정과 시드에 대한 완전 탐색(Exhaustive Search, 크기 및 비율을 고려하여 후보 영역 찾기)
- **R-CNN(Region-based CNN)** : CNN(이미지 분류) + 후보 영역 알고리즘 (이미지에서 객체가 있을 만한 영역 제안)
    - 이미지 삽입 → 후보 영역 추출(Cropping + Warping, 잘라서 크기 통일) → CNN 특성 계산 → 영역 분류
    - 복잡하고 긴 학습 과정, 대용량 저장 공간, 객체 검출 속도 문제

**6.2.2. 공간 피라미드 풀링**

- 이미지 자르기(Crop), 비율 조정(Warp) → 물체 왜곡 문제점
- **공간 피라미드 풀링(Spatial Pyramid Pooling)** : 입력 이미지의 크기에 관계없이 합성곱층 통과 → 완전연결층 전달 전 특성 맵들을 동일한 크기로 조절해 주는 풀링층 적용
    - 원본 이미지 특징 훼손 X, 여러 작업에 적용 가능
    
    ![image.png](attachment:232b3fee-7163-4f52-affb-f22e95a35234:image.png)
    

**6.2.3. Fast R-CNN**

- **Fast R-CNN(Fast Region-based CNN)** : R-CNN 속도 문제 개선(Bounding Box마다 CNN과 분류) → **RoI 풀링**
    - Bounding Box 정보 → CNN → RoI 풀링 (크기 조정) → 완전연결층 (Bounding Box마다 CNN 돌리는 시간 단축)
- **RoI 풀링** : 크기가 다른 특성 맵의 영역마다 스트라이드를 다르게 최대 풀링을 적용하여 결과값 크기를 동일하게 맞추는 방법
    
    ![image.png](attachment:097eb154-c858-4b06-8c70-0c6bc4be4ac8:image.png)
    

**6.2.4. Faster R-CNN**

- **Faster R-CNN** : Fast R-CNN에 **후보 영역 추출 네트워크(RPN, Regional Proposal Network)**를 추가 (후보 영역 생성을 CNN 내부 네트워크에서 진행)
    - 내부의 빠른 RPN(GPU 계산) > 외부의 느린 선택적 탐색(CPU 계산)
    - 마지막 합성곱층 → RPN → RoI 풀링 → 분류기, 바운딩 박스 회귀
- **후보 영역 추출 네트워크(RPN)** : 작은 윈도우 영역 입력 → 슬라이딩 윈도우 방식으로 객체 탐색 → 이진 분류(객체 존재 유무 판단) 작은 네트워크 & 바운딩 박스 회귀(위치 보정, 좌표점 추론)
    - 객체의 크기와 비율이 다양하다는 문제 (고정 입력 → 다양한 이미지 수용 어려움)
    - **앵커(Anchor)** : 레퍼런스 박스(Reference Box) k개 미리 정의(다양한 크기/비율) → 각각의 슬라이딩 윈도우 위치마다 박스 k개 출력
    - 모든 앵커 위치에 대해 → 분류(객체/배경 판단) & 회귀(위치 보정 `x` `y` `w` `h` 값)
    
    ![image.png](attachment:718596ee-c02b-488c-9305-dc24cb86c049:image.png)
    
    ![image.png](attachment:7de32924-e345-43d9-8f2f-d6c068857904:image.png)
    

### **📌 6.3. 이미지 분할을 위한 신경망**

**6.3.1. 완전 합성곱 네트워크**

- **이미지 분할(Image Segmentation)** : 이미지를 픽셀 단위로 분할하여 이미지에 포함된 객체 추출
- **완전 합성곱 네트워크(Fully Convolutional Network, FCN)** : 이미지 분류 우수 성능 CNN 기반 모델 → 이미지 분할에 적합하도록 변형
    - 완전연결층(고정된 크기의 입력, 위치 정보 사라짐) → **1x1 합성곱**으로 대체
    - BUT 여러 단계의 합성곱층과 풀링층 → 해상도 저하 → **업샘플링**으로 해결 → 이미지 세부 정보 잃어버림

**6.3.2. 합성곱 & 역합성곱 네트워크**

- **합성곱 & 역합성곱 네트워크(Convolutional & Deconvolutional Network)** : CNN 최종 출력 결과를 입력 이미지와 같은 크기로 만듦
    - **역합성곱 = 업샘플링(Upsampling)** : 픽셀 주위 제로 패딩(Zero Padding) 추가 → 합성곱 연산 수행
    - 시멘틱 분할(Semantic Segmentation, 물체들을 의미 있는 단위로 분할)
        
        ![image.png](attachment:c1a10466-6d3a-4b24-ba34-e665d8539d61:image.png)
        

**6.3.3. U-Net**

- **U-Net** : 바이오 메디컬 이미지 분할을 위한 합성곱 신경망
    - 빠른 속도 (이미 검증이 끝난 패치(patch, 이미지 인식 단위) 건너뛰기) ↔ 슬라이딩 윈도우 방식 (재검증)
    - 트레이드오프(Trade-off) 빠지지 않음 (넓은 범위 이미지, 컨텍스트(Context) 인식 ↔ 지역화 문제를 개선)
        - **지역화(Localization)** : 이미지 안에 객체 위치 정보를 출력 (바운딩 박스 - Left Top & Right Bottom 좌표)
    - **수축 경로(Contracting Path)** : 컨텍스트 포착
    - **확장 경로(Expansive Path)** : 특성 맵 업 샘플링, 수축 경로에서 포착한 특성 맵의 컨텍스트와 결합하여 정확한 지역화 수행
    - 합성곱 블록 = 3x3 합성곱 + 드롭아웃(Dropout) + 3x3 합성곱
        
        ![image.png](attachment:f5e552c6-d4df-46fe-95bc-864a325b7d91:image.png)
        

**6.3.4. PSPNet**

- **PSPNet(Pyramid Scene Parsing Network)** : 시멘틱 분할 알고리즘, 피라미드 풀링 모듈 추가
    - 1x1(광범위), 2x2, 3x3, 6x6 크기 풀링으로 서로 다른 크기의 이미지 출력 → 1x1 합성곱으로 채널 수 조정 (출력 채널 수 = 입력 채널 수 / 풀링층 개수) → 특성 맵 업 샘플링 + **양선형 보간법(Bilinear Interpolation)** → 원래 특성 맵과 생성된 새로운 특성 맵 병합
    - **보간법(Interpolation)** : 빈 화소에 값을 할당하여 좋은 품질의 영상을 만듦 → 선형 보간법 (Linear Interpolation, 화소 값 2개 사용하여 새로운 화소 값 계산), 양선형 보간법 (Bilinear Interpolation, 화소당 선형 보간 3번, 가장 가까운 화소 4개에 가중치 곱한 값을 합함)

**6.3.5. DeepLabv3 / DeepLabv3+**

- **DeepLabv3 / DeepLabv3+** : Atrous 합성곱 사용, 인코더-디코더 구조
    - 인코더에서 추출된 특성 맵 해상도를 Atrous 합성곱으로 제어
    - **Atrous 합성곱** : 필터 내부 빈 공간(rate 파라미터) → 수용 영역(Receptive Field, 특정 범위 한정적이고 효과적인 처리)을 확대하여 특성을 찾는 범위를 넓게 함
    
    ![image.png](attachment:bda45d40-7a0b-4b95-b558-d41de34314e9:image.png)
    

### **📌 7.1. 시계열 문제**

**7.1. 시계열 문제**

- 시**계열 분석** : 시간에 따라 변하는 데이터를 사용하여 추이를 분석 (ex. 주가, 환율, 기온, 습도)
    - 독립 변수가 시간!
- **시계열 형태(Components of Time Series)** - 데이터 변동 유형에 따라
    - **불규칙 변동(Irregular Variation)** : 규칙성 X, 예측 불가능, 우연적 발생 (ex. 전쟁, 홍수, 화재, 지진, 파업)
    - **추세 변동(Trend Variation)** : 장기적 변화 추세 - 지속적 증가/감소 혹은 일정한 상태 유지 (ex. 국내총생산, 인구증가율)
    - **순환 변동(Cyclical Variation)** : 일정한 기간을 주기로 순환 (ex. 경기 변동)
    - **계절 변동(Seasonal Variation)** : 계절적 영향, 사회적 관습에 따라 1년 주기로 발생
- 시계열 데이터는 트렌드 혹은 분산의 변화 여부에 따라 규칙적 시계열, 불규칙적 시계열로 구분
    - 시계열 데이터를 잘 분석 = 불규칙적 시계열 데이터 + 특정 기법, 모델 → 규칙적 패턴을 찾거나 예측

### **📌 7.2. AR, MA, ARMA, ARIMA**

**7.2.1. AR 모델**

- **AR(AutoRegressive, 자기 회귀) 모델** : 이전 관측 값이 이후 관측 값에 영향
    - $Z_t = Φ_1Z_{t-1} + Φ_2Z_{t-2} + ... + Φ_pZ_{t-p} + a_t$
    - 현재 시점 = 과거가 현재에 미치는 영향(Φ) × 과거 시점 + 오차 항(백색 잡음)
    - 이전 데이터의 상태에서 현재 데이터의 상태를 추론

**7.2.2. MA 모델**

- **MA(Moving Average, 이동 평균) 모델** : 트렌드(평균 혹은 시계열 그래프 y값)이 변화하는 상황에 적합한 회귀 모델
    - 윈도우 → 시계열을 따라 윈도우 크기만큼 슬라이딩(Moving)
    - $Z_t = θ_1a_{t-1} + θ_2a_{t-2} + ... + θ_pa_{t-p} + a_t$
    - 현재 시점 = 매개변수(θ) × 과거 시점 오차 + 오차 항
    - 이전 데이터의 오차에서 현재 데이터의 상태를 추론

**7.2.3. ARMA 모델**

- **ARMA(AutoRegressive Moving Average, 자기 회귀 이동 평균) 모델** : AR 모델 + MA 모델 두 가지 관점에서 과거 데이터 사용
    - $Z_t = a + Φ_1Z_{t-1} + Φ_2Z_{t-2} + ... + Φ_pZ_{t-p} + θ_1a_{t-1} + θ_2a_{t-2} + ... + θ_pa_{t-p} + a_t$

**7.2.4. ARIMA 모델**

- **ARIMA(AutoRegressive Integrated Moving Average, 자기 회귀 누적 이동 평균) 모델** : AR 모델 + MA 모델 + 추세(Cointegration)까지 고려한 모델
    - `ARIMA(p, d, q)` → `fit()` 데이터 적용, 훈련 → `predict()` 미래 추세, 동향 예측
        - `p` : 자기 회귀 차수
        - `d` : 차분 차수
        - `q` : 이동 평균 차수

### **📌 7.3. 순환 신경망(RNN)**

- **RNN(Recurrent Neural Network)** : 시간적으로 연속성이 있는 데이터를 처리하는 인공 신경망
    - 이전 은닉층이 현재 은닉층의 입력이 되며 반복(Recurrent)
    - 기억(Memory, 현재까지 입력 데이터 요약한 정보) → h_t
    - 외부 입력과 자신의 이전 상태를 받아 현재 상태를 갱신
        
        ![image.png](attachment:8371589b-65fa-4107-93fe-cfb455430fd4:image.png)
        
- **RNN의 입출력 유형**
    - 일대일 : 순환 X → ex. 순방향 네트워크
    - 일대다 : 입력 하나, 출력 다수 → ex. 이미지 캡션(Image Captioning, 이미지 설명 문장 출력)
    - 다대일 : 입력 다수, 출력 하나 → ex. 감성 분석(문장 긍정/부정 출력) + 적층 가능
        
        ![image.png](attachment:6b56c21d-9639-47f0-be83-800952391e01:image.png)
        
    - 다대다 : 입력 다수, 출력 다수 → ex. 언어 번역기
        - 텐서플로 : `keras.layers.SimpleRNN`에서 `return_sequences=True` 설정 (시퀀스 리턴)
        - 파이토치 : 시퀀스-투-시퀀스(Seq2Seq) 이용
        
        ![image.png](attachment:f237ac64-3b99-4e7f-ab66-18384b1a4727:image.png)
        
    - 동기화 다대다 : 입력 다수, 출력 다수 → ex. 문장 다음 단어 예측 모델, 프레임 수준의 비디오 분류
    
    ![image.png](attachment:3dd7ee11-af88-4cc9-af85-00abdcc8d467:image.png)
    

**7.3.1. RNN 계층과 셀**

- **RNN 계층(Layer)** : 입력된 배치 순서대로 모두 처리
    - 셀을 래핑하여 동일한 셀을 여러 단계에 적용
    - 파이토치는 계층과 셀을 분리하여 구현 가능
- **RNN 셀(Cell)** : 오직 하나의 단계(Time Step)만 처리 (RNN 계층의 for loop)
    - 단일 입력, 과거 상태(State) → 출력, 새로운 상태
    - 셀 유형 : `nn.RNNCell` `nn.GRUCell` `nn.LSTMCell`
    
    ![image.png](attachment:0cf51f18-c56f-4292-8fcd-baf51a6e1f59:image.png)
    
- RNN 활용 : 자연어 처리 (음성 인식, 단어 의미 판단, 대화 처리 등), 손글씨, 센서 데이터 등
    
    ```python
    self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)  # 임베딩 처리
    self.rnn = nn.RNNCell(input_dim, hidden_size)  # RNN 적용
    self.fc1 = nn.Linear(hidden_size, 256)  # 완전연결층
    self.fc2 = nn.Linear(256, 3)  # 출력층
    
    Seq2Seq(
      (encoder): Encoder(
        (embedding): Embedding(7855, 256)
        (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
        (dropout): Dropout(p=0.5, inplace=False)
      )
      (decoder): Decoder(
        (embedding): Embedding(5893, 256)
        (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
        (fc_out): Linear(in_features=512, out_features=5893, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
      )
    )
    ```
    

### **📌 7.4. RNN 구조**

- **RNN 구조** : 이전 단계 정보 → 은닉층 노드 저장 → 과거 정보와 현재 정보 모두 반영
    - **가중치** : $W_{xh}$ (입력층 → 은닉층), $W_{hh}$ (은닉층 → 다음 은닉층), $W_{hy}$ (은닉층 → 출력층)
    - **은닉층 계산** : 은닉층 가중치 × 이전 은닉층 + 입력층 가중치 × 현재 입력 값
        - $h_t = tanh(\hat{y_t})=tanh(W_{hh}×h_{t-1}+W_{xh}×x_t)$
    - **출력층 계산** : 출력층 가중치 × 현재 은닉층
        - $\hat{y_i} = softmax(W_{hy}×h_t)$
    
    ![image.png](attachment:6bc9debf-c079-44cc-889d-32a9b1396887:image.png)
    
    - **오차(E)** : 각 단계마다 실제 값과 예측 값으로 오차 측정 (평균 제곱 오차) → 이전 단계로 전달(BPTT)
    - **역전파** : BPTT(BackPropagation Through Time) 이용, 모든 단계마다 처음부터 끝까지 역전파 진행
        - $W_{xh}$, $W_{hh}$, $W_{hy}$, 바이어스(bias) 업데이트
        - 기울기 소멸 문제(Vanishing Gradient Problem) : 멀리 전파되면 전파되는 양이 점차 적어짐 → 생략된-BPTT(Truncated BPTT, 일정 시점까지만 오류 역전파), LSTM, GRU
    - 계층 자체 개수(비선형 문제 학습) vs 계층의 유닛 개수(가중치, 바이어스 계산) : 계층 자체 개수 늘리기
    
    ![image.png](attachment:886a8b21-9094-44b7-9e54-ddfb4e7c6bba:image.png)
    
    - 입력층 입력 : `[시퀀스 길이, 배치 크기, 은닉층 뉴런 개수]`
    - 은닉층 입력 : `[은닉층 개수, 배치 크기, 은닉층 뉴런 개수]`
    
    ![image.png](attachment:a022be7f-4803-40c5-9dda-f7039e90dde8:image.png)
    

### **🤔 더 알아보기**

1. **GoogLeNet은 어떻게 구현하는가?**
    - https://github.com/teddylee777/pytorch-tutorial/blob/main/14-GoogleNet-Inception-Module.ipynb
    - 1x1 Convolution을 통해 파라미터 수를 획기적으로 줄일 수 있다!

1. YOLO vs R-CNN
    
    
    | **R-CNN** | **공간 피라미드** | **Fast R-CNN** | **Faster R-CNN** |
    | --- | --- | --- | --- |
    |  | 합성곱층 | 합성곱층 | 합성곱층 |
    | 후보 영역 추출
    (선택적 탐색
    알고리즘) | 후보 영역 추출
    (선택적 탐색
    알고리즘) | 후보 영역 추출
    (선택적 탐색
    알고리즘) | 후보 영역 추출
    (후보 영역 추출
    네트워크) |
    | Crop + Warp | 최대 풀링/연결 | RoI 풀링 | RoI 풀링 |
    | 합성곱층 |  |  |  |
    | 완전연결층
    (분류/회귀) | 완전연결층
    (분류/회귀) | 완전연결층
    (분류/회귀) | 완전연결층
    (분류/회귀) |
    - YOLO : 단일 단계 객체 탐지 알고리즘
        
        ![image.png](attachment:daacd0e0-f90b-489d-b83e-cd9152cbfbca:image.png)
        
        - 원본 이미지 → 그리드(Grid, 동일 크기) → Anchor Boxes(미리 정의된 형태를 가진 경계 박스, K-평균 알고리즘 생성) → 서로 다른 크기와 형태의 객체 탐지
    - Faster R-CNN : 이단계 방식 객체 탐지 알고리즘
        - CNN으로 Feature Map 추출 → RPN(Region Proposal Network)으로 개체를 포함할 가능성이 높은 윈도우 생성(이진 분류) → 각 윈도우/관심 영역(RoI)에 대한 피처 추출 계산