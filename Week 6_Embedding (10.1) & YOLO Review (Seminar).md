[Google Colab](https://colab.research.google.com/drive/1BqRIsAG5Y3XKfdbzYz3ye267H7iCzmuV?usp=sharing)

### 📌 10.1. 임베딩

- **임베딩(Embedding)** : 언어 → 벡터(Vector, 컴퓨터가 이해할 수 있는 숫자 형태)
    - 단어 및 문장 간 관련성 계산
    - 의미적, 문법적 정보 함축

**10.1.1. 희소 표현 기반 임베딩**

- **희소 표현(Sparse Representation) 기반 임베딩** : 대부분의 값이 0으로 채워져 있는 경우
- **원-핫 인코딩(One-Hot Encoding)** : 단어 N개를 N차원의 벡터로 표현, 단어가 포함되어 있는 위치에 1을 넣고 나머지는 0 값으로 채움
    - `sklearn.preprocessing.LabelEncoder()` `sklearn.preprocessing.OneHotEncoder()`
    - 단점 1. 단어끼리의 관계성 포착 불가
        - 벡터의 내적(Inner Product)은 언제나 0 = 직교(Orthogonal) → 독립적(Independent)인 관계
    - 단점 2. 차원의 저주(Curse of Dimensionality) 문제
        - 말뭉치(Corpus)의 단어 종류만큼 차원이 존재
    
    ![image.png](attachment:510fc64d-d49b-409f-a211-cb985938fff5:image.png)
    

**10.1.2. 횟수 기반 임베딩**

- **카운터 벡터(Counter Vector)** : 각 단어의 출현 빈도수를 이용하여 인코딩해서 벡터를 만드는 방법 (토크나이징 + 벡터화)
    - `sklearn.feature_extraction.text.CountVectorizer()`
        - 문서 → 토큰 리스트(토큰의 출현 빈도) → 인코딩(벡터 변환)
- **TF-IDF(Term Frequency-Inverse Document Frequency)**
    - 정보 검색론(Information Retrieval, IR)에서 가중치를 구할 때 사용되는 알고리즘
    - 특정 문서 내에서 단어의 출현 빈도가 높을수록, 전체 문서에서 특정 단어가 포함된 문서가 적을수록 높음 → 흔한 단어를 걸러 내고 특정 단어에 대한 중요도 찾을 수 있음
    - **TF(Term Frequency)** : 단어 빈도 (문서 내에서 특정 단어가 출현한 빈도)
        
        $$
        tf_{t,d} = \begin{cases} 1 + log\ count(t,d) & \text{if } count(t,d) > 0 \\ 0 & \text{otherwise} \end{cases}
        $$
        
    - **IDF(Inverse Document Frequency)** : 역문서 빈도
        - DF(Document Frequency) : 문서 빈도 (특정 단어가 나타난 문서 개수, 전체 문서에서 얼마나 공통적으로 많이 등장하는지)
        - 스무딩(Smoothing) : 분모에 1을 더함 (빈도가 0이면 분모가 0이 되는 상황 방지)
        
        $$
        idf_t = log(\frac{N}{1+df_t})
        $$
        
    - `sklearn.feature_extraction.text.TfidVectorizer()`

**10.1.3. 예측 기반 임베딩**

- 예측 기반 임베딩 : 신경망 구조(모델)을 이용하여 특정 문맥에서 어떤 단어가 나올지 예측하여 단어를 벡터로 만듦
- **워드투벡터(Word2Vec)**
    - 주어진 텍스트에서 텍스트의 각 단어마다 하나씩 일련의 벡터를 출력
        - 일정한 크기의 윈도우(Window)로 텍스트 분할 → 대상 단어 + 컨텍스트로 신경망의 입력 사용 → 은닉층에는 각 단어에 대한 가중치 포함
        
        ![image.png](attachment:7291550b-2ffe-44eb-a818-7edd072cbdf1:image.png)
        
    - 의미론적으로 유사한 단어의 벡터는 서로 가깝게(코사인 유사도, 단어 간의 거리) 표현됨
- **CBOW(Continuous Bag of Words)** : 단어를 여러 개 나열한 후 다음에 등장할 단어 예측 (Context Word ⇒ Central Word)
    - 은닉층의 크기 N = 입력 텍스트를 임베딩한 벡터 크기
        
        ![image.png](attachment:475e4eb4-aead-4554-be31-edf2f521455f:image.png)
        
        ![image.png](attachment:e571d686-816f-4efe-823d-d0f76e99b0f4:image.png)
        
    - `gensim.models.Word2Vec(data, min_count, vector_size, window, sg)`
        - `min_count` : 단어 최소 빈도수 제한 (빈도가 적은 단어들은 학습 X)
        - `vector_size` : 임베딩된 벡터의 차원
        - `window` : 컨텍스트 윈도우 크기
        - `sg` : `0` CBOW (기본값), `1` Skip-gram
- **Skip-Gram** : 특정한 단어에서 문맥이 될 수 있는 단어 예측 (Central Word ⇒ Context Word)
    
    ![image.png](attachment:8b91811b-29a8-46e9-821d-b043604dcb3d:image.png)
    
    ⇒ 데이터 성격, 분석에 대한 접근 방법 및 도출하고자 하는 결론 등을 종합적으로 고려!
    
- **패스트텍스트(FastText)**
    - 워드투벡터는 사전에 없는 단어의 벡터 표현 불가, 자주 사용되지 않는 단어의 학습 불안정
    - 워드투벡터 분산 표현(Distributed Representation) ↔ 패스트텍스트 단어 표현(Word Representation)
    - 노이즈에 강함, 새로운 단어에 대해 형태적 유사성을 고려하여 벡터 값을 얻음
        - 모든 단어를 n-그램(n-gram)에 대해 임베딩 (사전에 없는 단어는 n-그램으로 분리된 부분 단어와 유사도를 계산하여 의미 유추)
        - n-그램(n-gram) : n개의 어절/음절을 연쇄적으로 분류하여 빈도를 계산 (n=1 유니그램, n=2 바이그램, n=3 트라이그램)
        - 등장 빈도 수가 적더라도 n-그램으로 임베딩 → 높은 정확도
    - 사전 학습된 패스트텍스트 : `gensim.models.KeyedVectors`
        
        

**10.1.4. 횟수/예측 기반 임베딩**

- **글로브(GloVe, Global Vectors for Word Representation)**
    - 횟수 기반의 LSA(Latent Semantic Analysis, 잠재 의미 분석), 예측 기반의 워드투벡터 단점 보완
    - 단어에 대한 글로벌 동시 발생 확률(Global Co-occurence Statistics) 정보 포함
    - Skip-Gram + 단어에 대한 통계 정보

### 📌 You Only Look Once: Unified, Real-Time Object Detection

[프미나 → 5/7 21시 진행] CNN 배경지식 + YOLO 논문 리뷰

**1️⃣ Introduction**

- 기존의 Object Detection : Classifier(분류기) 사용, 다양한 위치에서 물체가 존재하는지 판단
    - **Deformable Parts Models (DPM)** : Sliding Window Approach (전체 이미지를 훑으며 Classifer 활용)
    - **R-CNN** : Region Proposal Method (Bounding Box를 만들고 Classifer 실행, 이후 후처리)
    
    ⇒ 문제점 : 각각의 요소들을 개별적으로 훈련해야 하므로 느리다!
    
- **YOLO (YOU ONLY LOOK ONCE)** : 단일 Regression으로 동시에 여러 개의 Bounding Box와 Class Probability 예측
    - 장점1) **빠른 속도** : 45 frames/second (배치 연산 없이 Titan X GPU 기준), 다른 실시간 시스템과 비교했을 때 평균 정확도 두 배 이상
    - 장점2) **전역적인 이미지 추론** : 전체 이미지의 정보 활용 (R-CNN은 큰 맥락을 보지 못함), Background Error 절반 이상 줄임
    - 장점3) **사물의 Generalizable Representation을 배움** : 새로운 영역이나 예상하지 못한 입력에 대해서도 적용 가능

**2️⃣ Unified Detection**

- **Global Reasoning** : 하나의 신경망으로 전체 이미지의 특성으로 모든 물체(Bounding Box)를 동시에 예측
- 입력 이미지 → S × S 그리드 → 각 그리드 셀이 B개의 Bounding Box와 Confidence Score, C개의 Pr(Class_i | Object) 확률 예측
    - **Bounding Box [B개]** : 물체의 정중앙이 들어있는 그리드 셀이 해당 물체를 탐지
        - **`(x, y, w, h)`** → `(x, y)`는 박스의 중심, `(w, h)`는 전체 이미지 대비 너비와 높이
    - **Confidence Score [B개]** : 예측한 박스와 Ground Truth Box 사이의 IOU
        - **`Confidence`** → Pr(Object) × IOU_truth_pred  (물체가 없다면 0)
    - **Pr(Class_i | Object) [C개]** : 물체가 있는 경우 예측 (Bounding Box 개수와 무관하게 그리드 셀마다 한 세트 예측)
        - 테스트 시 → Pr(class_i | Object) × Pr(Object) × IOU
    
    ⇒ S × S × (B * 5 + C) tensor 출력 ⇒ S = `7`, B = `2`, C = `20` ⇒ 7 × 7 × 30 tensor 출력
    

**2️⃣.1️⃣ Design**

- **GoogLeNet 모델 기반** : 합성곱층 (24개) → 특성 추출 / 완전연결층 (2개) → 확률 출력
- **차이점**
    - 1 × 1 Reduction Layers & 3 × 3 합성곱층 사용
    - 입력 이미지 224 x 224 → 448 x 448로 변경
    

**2️⃣.2️⃣ Training**

- **사전 훈련 및 입출력**
    - 사전 훈련 (합성곱층 20개 + 평균 풀링층 + 완전연결층) → ImageNet 1000-class Competition Dataset
    - 4개의 합성곱층, 2개의 완전연결층을 랜덤하게 초기화된 가중치로 추가
    - 마지막 완전연결층 → Bounding Box 위치(`w`, `h`, `x`, `y` 0~1 사이로 정규화)와 Class 확률 출력
- **활성화 함수**
    - 마지막 층은 선형 활성화 함수
    - 나머지는 Leaky ReLU 활성화 함수 (양수일 때 x, 그 외 0.1x 출력)
- 손실 함수/최적화 : **Sum-Squared Error**
    - **Localization Error** : ① (x, y), ② (w, h)
    - **Classification Error** : ③/④ Confidence (obj/noobj), ⑤ Class Probability
    - 문제점1) Localization Error와 Classification Error의 동일한 가중치
        - Localization → Bounding Box 위치 예측의 loss는 증가 (λ_coord = 5)
        - Confidence → 물체가 없는 셀에서 Confidence 예측의 loss는 감소 (λ_noobj = 0.5)
    - 문제점2) Localization Error : 큰 박스와 작은 박스의 동일한 가중치
        - Localization → 박스의 너비, 높이의 루트값을 예측
    - 문제점3) Classification Error : 물체가 없는 그리드 셀의 Confidence Score는 0이 됨
        - Class Proability → 물체가 그리드 셀에 있을 때만 계산
    - 물체는 하나의 Predictor가 탐지(IOU가 가장 높은 것) → 책임 할당
        - Predictor에게 책임이 있는 물체만 Bounding Box 위치 예측의 Loss 계산
- **학습 설정**
    - PASCAL VOC 2007, 2012
    - Batch size `64`, Momentum `0.9`, Decay `0.0005`
    - 135 epochs / Learning Rate : `1e-3` → `1e-2` (75+ epochs) → `1e-3`  (30 epochs) → `1e-4` (30 epochs)
    - Dropout : 첫 번째 완전연결층 이후 `0.5`
    - Extensive Data Augmentation : 랜덤 스케일링과 이동 (~20%), 밝기와 채도 조정 (1.5 HSV)

**2️⃣.3️⃣. Inference**

- 빠른 속도(Single Network Evaluation) → 이미지당 98개의 Bounding Box 예측
- 사물마다 하나의 박스만 예측 ← Non-maximal Suppression (큰 사물, 경계에 걸친 사물의 중복 검출 방지), mAP 2~3% 향상

**2️⃣.4️⃣. Limitations of YOLO**

- 인접한 여러 물체를 동시에 탐지하는 데 제약 (ex. 새 떼와 같은 작은 물체 문제)
- 새롭거나 전형적이지 않은 비율로 나타난 데이터 적용 어려움
- Downsampling Layers 여러 개 → Bounding Box 예측할 때 Coarse Features 사용
- Bounding Box 크기에 상관없이 에러를 처리 (작은 박스에서의 작은 오차가 큰 영향을 미침)

**3️⃣ Comparison to Other Detection Systems**

- **DPM (Deformable Parts Model)** : 슬라이딩 윈도우 기반 (정적 특성 추출 → 분류 → 박스 예측)
- **R-CNN** : Selective Search 기반 (영역 제안 → 특성 추출 → 점수화/후처리 → 중복 제거)
    - R-CNN은 속도가 느림(약 40초), YOLO는 더 적은 Bounding Box 제안(98개 vs 2000개)
- **Fast/Faster R-CNN** : 속도 개선, 그러나 여전히 실시간 처리는 어려운 속도
    
    ⇒ YOLO는 Object Detection의 모든 과정을 **하나의 Neural Network로 통합하여 더 빠르다**!
    
- **Deep Multibox** : CNN 기반 영역 제안, 그러나 일반적인 객체 탐지는 불가 (단일 클래스 예측 최적화)
- **OverFeat** : 슬라이딩 윈도우 기반, 지역 정보만 활용(Localization 적합)
- **MultiGrasp** : Grid Approach를 사용하여 단일 객체의 그리드 영역 예측
    
    ⇒ YOLO는 **전역(Global) 정보를 활용**하여 **일반적인(General) 다중 클래스 객체를 탐지**할 수 있다!
    

**4️⃣ Experiments**

**4️⃣.1️⃣. Comparison to Other Real-Time Systems**

- 테스트 기준 : PASCAL VOC 2007
- Fast R-CNN : Selective Search 2초
- Faster R-CNN VGG-16 : RPN(Regional Proposal Network) 기반 속도 단축, 그러나 YOLO 대비 6배 느린 속도
- Faster R-CNN ZF : YOLO 대비 2.5배 느린 속도, 적은 정확도
- **Fast YOLO** : 기존 실시간 모델 대비 **2배 높은 정확도, 빠른 속도**
- **YOLO** : Fast YOLO 대비 mAP 10% 증가
    
    ⇒ YOLO는 **높은 정확도와 빠른 속도**를 동시에 보인다!
    

**4️⃣.2️⃣. VOC 2007 Error Analysis ~ 4️⃣.3️⃣. Combining Fast R-CNN and YOLO**

- Fast R-CNN **Background Error** / YOLO **Localization Error**
- Fast R-CNN과 YOLO 합치기
    - R-CNN이 예측한 Bounding Box → YOLO가 유사한 박스를 예측하는지 확인
    - 정확도 증가, 그러나 YOLO의 빠른 속도를 이점으로 활용 X
    
    ⇒ YOLO와 Fast R-CNN을 합쳐 **Background Error를 줄일 수 있다**!
    

**4️⃣.4️⃣. VOC 2012 Results ~ 4️⃣.5️⃣. Generalizability**

- 테스트 기준 : PASCAL VOC 2012
- Fast R-CNN + YOLO의 높은 성능 vs. 작은 물체 탐지의 어려움
    
    ⇒ YOLO는 **작은 물체 탐지에 어려움**을 겪는다!
    
- R-CNN은 PASCAL VOC에 과적합 (작은 지역만 봄)
- DPM은 성능 유지 (물체의 모양에 대한 공간적 모델)
    
    ⇒ YOLO는 이미지 전체 정보를 활용하여 **일반적인 객체 탐지에 적합**하다!
    

**5️⃣ Real-Time Detection In The Wild ~ 6️⃣ Conclusion**

- YOLO는 Object Detection을 위한 **Unified Model**
- YOLO는 **높은 정확도와 빠른 속도**를 갖기에 **실시간 처리**가 가능 (Tracking System 사용)
- YOLO는 **일반적인 객체 탐지**에도 적합

https://github.com/sangnekim/YOLO-v1-for-studying

[Machine-Learning-Collection/ML/Pytorch/object_detection/YOLO at master · aladdinpersson/Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO)

[YOLO모델 및 버전별 차이점 분석](https://velog.io/@juneten/YOLO-v1v9#yolo-v3)

[[Object Detection] YOLO모델의 발전 과정 정리해보기 V1~12](https://c0mputermaster.tistory.com/30)

YOLO v2 (2017) : Global Average Pooling, Anchor Box 도입

YOLO v3 (2018) : Feature Pyramid Network → **작은 객체 탐지 개선**

YOLO v4 (2020) : SPP + PANet, Data Augmentation, Quantization

YOLO v5 (2020) : Pytorch 변환, Mosaic Augmentation

YOLOX    (2021) : SPP, Anchor-free (중심점만 사용), SimOTA

YOLO v6 (2022) : Rep-PAN, PTQ와 QAT → 양자화 체계 최적화

YOLO v7 (2022) : E-ELAN, Soft Label 생성 → 파라미터 효율성 개선

YOLO v8 (2023) : SPP + PANet + SAM → **경량화된 모델**

YOLO v9 (2024) : PGI, Gradient Path Planning → 정보 병목 문제 해결

YOLO v10 (2024) : Dual Label Assignments (NMS 제거), Rank-guided Block Design

YOLO v11 (2024) : C3k2 Block + SPPF + C2PSA → **다양한 태스크 지원**

YOLO v12 (2025) : Area Attention, R-ELAN, 주의 집중 아키텍처 최적화 (플래시어텐션)