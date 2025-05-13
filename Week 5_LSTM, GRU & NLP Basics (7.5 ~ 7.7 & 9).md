[Google Colab](https://colab.research.google.com/drive/1g7yct5jL2EBKqB4idB23irK9V02xz622?usp=sharing)

### 📌 7.5. LSTM

**7.5.1. LSTM 구조**

- **LSTM 순전파** (기울기 소멸 문제 해결을 위해 각 뉴런마다 망각/입력/출력 게이트)
    
    ![image.png](attachment:dc759b0b-7db9-4305-8392-9b0540a4d6bf:image.png)
    
    - **망각 게이트(Forget Gate)** : 과거 정보를 어느 정도 기억할지 결정
        - 시그모이드 출력이 0이면 과거 정보 버리기, 1이면 보존하기
        - $c_t = \sigma(w_f [h_{t-1}, x_t])\ ·\ c_{t-1}$
            
            ![image.png](attachment:634b9937-a5c6-4677-955c-d55292474eb7:image.png)
            
    - **입력 게이트(Input Gate)** : 현재 정보를 기억
        - 계산값이 0이면 입력 차단하기, 1이면 허용하기
        - $c_t = c_{t-1} + \sigma(w_i [h_{t-1}, x_t])\ ·\ tanh(w_c [h_{t-1}, x_t])$
            
            ![image.png](attachment:e907dc00-0d72-467c-b7fe-2234ba123e18:image.png)
            
    - **메모리 셀** : 각 단계에 대한 은닉 노드(Hidden Node)
        - 셀 상태(Cell State) 업데이트 : 망각 게이트와 입력 게이트의 이전 단계 셀 정보를 계산하여 업데이트
            
            ![image.png](attachment:127853bc-5de4-42d8-ad7e-9cb4864c3e3d:image.png)
            
    - **출력 게이트(Output Gate)** : 과거 정보와 현재 데이터를 사용하여 뉴런의 출력 결정
        - 계산값이 0이면 출력하지 않기, 1이면 출력하기
        - $h_t = \sigma(w_o [h_{t-1}, x_t])\ ·\ tanh(c_{t-1})$
            
            ![image.png](attachment:4d20f66e-7e7a-4370-b9ca-f9eff52d280c:image.png)
            
- **LSTM 역전파**
    - **중단 없는 기울기(Uninterrupted Gradient Flow)** : 최종 오차는 셀을 통해 중단 없이 모든 노드에 전파 (입력 방향으로도 전파)
        - $t_t = tanh(w_{hh}h_{t-1} + w_{xh}x_t)$

### 📌 7.6. GRU

**7.6.1. GRU 구조**

- LSTM의 망각 게이트와 입력 게이트를 하나로 합침(업데이트 게이트)
- 하나의 게이트 컨트롤러(Gae Controller)가 망각 게이트(출력 1)와 입력 게이트(출력 0)를 모두 제어
- 출력 게이트가 없어 전체 상태 벡터가 매 단계마다 출력됨, 이전 상태의 어느 부분이 출력될지 제어하는 새로운 게이트 컨트롤러 별도 존재
    
    ![image.png](attachment:814cace6-2008-4f7d-b9b3-b1df93312c7f:image.png)
    
    - **망각 게이트(Reset Gate)** : 과거 정보 초기화 목적
        - $r_t = \sigma(W_r\ ·\ [h_{t-1}, x_t])$
        
        ![image.png](attachment:40584850-e749-4f0f-8d3d-558b3001b27d:image.png)
        
    - **업데이트 게이트(Update Gate)** : 과거와 현재 정보의 최신화 비율 결정
        - $z_t = \sigma(W_z\ ·\ [h_{t-1}, x_t])$
        
        ![image.png](attachment:e4ba14e7-b7fd-4b33-b689-b2542b8b208d:image.png)
        
    - **후보군(Candidate)** : 현 시점의 정보에 대한 후보군 계산 (망각 게이트의 결과를 이용)
        - $\~{h}_t = tanh(W\ ·\ [r_t * h_{t-1}, x_t])$
    - **은닉층 계산** : 업데이트 게이트 결과와 후보군 결과 종합
        - $h_t = (1 - z_t) * h_{t-1} + z_t × \~h_t$

### 📌 7.7. 양방향 RNN

**7.7.1. 양방향 RNN 구조**

- **양방향 RNN(Bidirectional RNN)** : 이전+이후 시점의 데이터를 함께 활용하여 출력 값을 예측
- 하나의 출력 값을 예측하는 데 메모리 셀 2개 사용
    - 메모리셀1 : 이전 시점 은닉 상태 (Forward States) → 현재 은닉 상태 계산
    - 메모리셀2 : 다음 시점 은닉 상태 (Backward States) → 현재 은닉 상태 계산

### 📌 9.1. 자연어 처리란

- **자연어 처리** : 언어 의미 분석 → 컴퓨터가 처리
    - ex. 스팸 처리, 맞춤법 검사, 단어 검색, 객체 인식, 질의응답, 요약, 유사 단어 바꾸어 쓰기, 대화

**9.1.1. 자연어 처리 용어 및 과정**

- **말뭉치(Corpus, 코퍼스)** : 모델을 학습시키기 위한 데이터 (특정한 목적에서 표본을 추출한 집합)
- **토큰(Token)** : 문서를 나누는 단위
    - **토큰 생성(Tokenizing)** : 문자열을 토큰으로 나누는 작업 (← 토큰 생성 함수)
- **토큰화(Tokenization)** : 텍스트를 문장이나 단어로 분리
    - `nltk.word_tokenize()`
- **불용어(Stop Words)** : 문장 내에서 많이 등장하는 단어 (분석과 무관) → 제거
- **어간 추출(Stemming)** : 단어를 기본 형태로 만드는 작업
- **품사 태깅(Part-of-Speech Tagging)** : 주어진 문장에서 품사를 식별하기 위해 붙여 주는 태그(식별 정보)
    - ex. Det, Noun, Verb, PPrep
    - `nltk.pos_tag()`
- 자연어 처리 과정 : 자연어 입력 텍스트 → 전처리 과정 → 단어 임베딩
    
    ![image.png](attachment:24ae431d-87f5-4202-a436-eb2390cf4320:image.png)
    

**9.1.2. 자연어 처리를 위한 라이브러리**

- NLTK(Natural Language Toolkit) : 말뭉치, 토큰 생성, 형태소 분석, 품사 태깅 등
- KoNLPy : 한국어 처리를 위한 파이썬 라이브러리 (형태소 분석기 - 꼬꼬마(Kkma), 코모란(Komoran), 한나눔(Hannanum), 트위터(Twitter), 메카브(Mecab))
    - `형태소 분석기.morphs()`
    - `형태소 분석기.pos()`
- Gensim : 워드투벡터(Word2Vec) 라이브러리 (임베딩, 토픽 모델링, LDA(Latent Dirichlet Allocation)
- 사이킷런(scikit-learn) : 문서 전처리 라이브러리 제공
    - `CountVectorizer` : 텍스트에서 단어의 등장 횟수를 기준으로 특성 추출
    - `TfidVectorizer` : TF-IDF 값을 사용해 텍스트에서 특성 추출
    - `HashingVectorizer` : CountVectorizer과 동일, 해시 함수 사용으로 실행 시간 감소

### 📌 9.2. 전처리

- 문장 → 결측치 확인, 토큰화 → 단어 색인 → 불용어 제거 → 축소된 단어 색인 → 어간 추출

**9.2.1. 결측치 확인**

- `isnull()` : 결측치 확인 (`df.isnull().sum()`)
- `dropna()` : 결측치 포함한 행 삭제 (`how='all'` 지정하면 모든 행이 NaN일 때만 삭제)
- `fillna(값)`  : 결측치 채우기 (최빈값, 평균값 등)

**9.2.2. 토큰화**

- 토큰화(Tokenization) : 텍스트 → 단어/문자 단위 (문장 토큰화, 단어 토큰화)
- 문장 토큰화 : 마침표, 느낌표, 물음표 등의 기호에 따라 문장 분리 → `sent_tokenize()`
- 단어 토큰화 : 띄어쓰기 기준으로 단어 분리 → `word_tokenize()`
    - 혹은 `nltk.tokenize.WordPunctTokenizer().tokenize()`  (아포스트로피 분리)

**9.2.3. 불용어 제거**

- 불용어(Stop Word) : 문장 내에서 빈번하게 발생하여 의미를 부여하기 어려운 단어들 (ex. a, the)
    - `stopwords.words('english')`

**9.2.4. 어간 추출**

- 어간 추출(Stemming), 표제어 추출(Lemmatization) : 단어 원형을 찾아주는 것
    - 어간 추출 = 단어 자체만 고려 (품사가 달라도 사용 가능)
        - 포터(porter) → `PorterStemmer().stem()`
        - 랭커스터(lancaster) → `LancasterStemmer().stem()`
    - 표제어 추출 = 같은 품사에 대해 사용 (사전에 있는 단어만 추출)
        - `WordNetLemmatizer().lemmatize(단어, 품사)`

**9.2.5. 정규화**

- 정규화(Normalization) : 데이터셋이 가진 특성(칼럼)의 모든 데이터가 동일한 정도의 범위(스케일, 중요도)를 갖도록 함
    - `MinMaxScaler().fit_transform()` : 0~1 사이에 위치하도록 값의 범위 조정 (이상치에 민감)
        - $\frac{x-x_{min}}{x_{max}-x_{min}}$
    - `StandardScaler().fit_transform()` : 평균 0, 분산 1로 범위 조정
        - $\frac{x - \mu}{\sigma}$
    - `RobustScaler().fit_transform()` : 중간값(Median)과 사분위수 범위(IQR) 사용 (더 넓게 분포)
        - Q1 = 25%, Q2 = 50%, Q3 = 75%, IQR = Q3-Q1
    - `MaxAbsScaler().fit_transform()` : 절댓값이 0~1 사이가 되도록 조정 (이상치에 민감)