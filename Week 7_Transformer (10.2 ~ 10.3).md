[Google Colab](https://colab.research.google.com/drive/1olH1GlfaiHx9ocn8Wncl_90Xa3uylB7E?usp=sharing)

### 📌 10.2. 트랜스포머 어텐션

- 인코더-디코더 네트워크 : 입력에 대한 벡터 변환 → 인코더(Encoder) → 디코더(Decoder)
- **어텐션(Attention)** : 인코더의 은닉 상태 모두를 디코더에서 활용 (소프트맥스 함수를 사용하여 가중합을 구하고 그 값을 전달)
    - 초기 정보를 잃어버리는 기울기 소멸 문제 해결
    - 은닉 상태 → 소프트맥스 함수(중점적으로 집중해서 보아야 할 벡터)로 점수를 매긴 후 각 은닉 상태 벡터들과 곱함 → 은닉 상태를 모두 더하여 하나의 값을 만들어 전달
    
    ![image.png](attachment:f1172c04-5eb8-4749-b8e5-c24c01a4e8f1:image.png)
    
- **트랜스포머(Transformer)** : 어텐션을 극대화하는 방법
    - 인코더와 디코더를 여러 개(6개) 중첩시킨 구조 (블록 = 각 인코더와 디코더)
    
    ![image.png](attachment:10c6becd-da87-4643-acdd-a34f157b2abf:image.png)
    
    - 인코더 블록 구조 = 셀프 어텐션(Self-Attention) + 전방향 신경망(Feed Forward Neural Network)
        - 임베딩 : 단어 → 벡터
        - 셀프 어텐션 : 문장에서 각 단어끼리 얼마나 관계가 있는지 계산 (단어 간의 관계 파악)
    - 디코더 블록 구조 = 셀프 어텐션(Self-Attention) + 인코더-디코더 어텐션(Encoder-Decoder Attention) + 전방향 신경망(Feed Forward Neural Network)
        - 인코더-디코더 어텐션 : 인코더가 처리한 정보를 받아 Attention 메커니즘 수행
    - Attention 메커니즘
        
        ![image.png](attachment:f7db3815-2c12-4125-a414-de30f07d99ca:image.png)
        
        - 어텐션 스코어 $e_{ij} = a(s_{i-1}, h_j)$ 계산 (디코더 현 시점 은닉 상태 & 인코더의 모든 은닉 상태의 유사도 판단)
        - 소프트웨어 함수 적용(특정 시점 가중치) $a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}$
        - 컨텍스트 벡터(Context Vector, 가중합) $c_i = \sum_{j=1}^{T_x}a_{ij}h_j$
        - 디코더 은닉 상태 $s_i = f(s_{i-1}, y_{i-1}, c_i)$ 계산 (디코더 이전 시점 은닉 상태 & 디코더 이전 시점 출력 & 컨텍스트 벡터)

**10.2.1. seq2seq**

- seq2seq (sequence to sequence) : 입력 시퀀스에 대한 출력 시퀀스를 만드는 모델 (ex. 번역)
    - 입력/출력 시퀀스 길이가 다를 수 있음
        
        ![image.png](attachment:2450f246-1db2-4379-a9bf-70641fe70f33:image.png)
        
        ![image.png](attachment:3ede9567-0f23-48da-8a1e-12822d2524d5:image.png)
        
    - teacher force : 번역(예측)하려는 목표 단어를 디코더의 다음 입력으로 넣어줌
        
        ![image.png](attachment:6dc95ee7-f429-4be0-9554-d77fc99724d0:image.png)
        

**🤔 Attention의 등장 배경…**

- seq2seq → RNN(LSTM, GRU)의 마지막 은닉 상태만 디코더로 전달됨
    - 하나의 고정된 크기의 벡터에 모든 정보를 담음 (정보의 손실)
    - RNN의 기울기 소멸 문제
- Attention → 특정 시점마다 다른 컨텍스트 벡터를 사용

**10.2.2. 버트(BERT)**

- BERT(Bidirectional Encoder Representations from Transformers) : 양방향 자연어 처리 모델 (ex. 문장 예측, Next Sentence Prediction) ⇒ 트랜스포머와 사전 학습으로 성능 향상
    - 기존의 단방향 자연어 처리 모델들의 단점을 보완
    - 검색 문장의 단어를 입력된 순서대로 하나씩 처리하는 것이 아니라, Transformers를 이용하여 구현
    - 방대한 양의 텍스트 데이터로 사전 훈련된 언어 모델 (전이 학습 → 인코더-디코더 중 인코더만 사용하는 모델)
    - BERT-base (L=12, H=768, A=12), BERT-large (L=24, H=1024, A=16) (L은 전이 블록 수, H는 은닉층 크기, A는 전이 블록에서 사용되는 어텐션 블록 수)
- 트랜스포머(Transformer)라는 인코더를 쌓아 올린 구조
    
    ![image.png](attachment:490053ef-a1ec-4dbe-b9d9-00a095b4f83e:image.png)
    
    - 문장 → 입력 형식 변환 (문장 시작은 `[CLS]`, 문장 끝은 `[SEP]`)
    - 한 문장의 단어 대한 토큰화 (고양이 → `고##`, `#양#`, `##이`)
    - 각 토큰에 대해 고유 아이디 부여 (존재하지 않는자리는 `0`)
- 버트 모델 : `bert-base-multilingual-cased`
    - `bert-base-uncased`는 가장 기본적인 모델, 모든 문장을 소문자로 대체
    - `BertTokenizer.from_pretrained` : 사전 훈련된 버트의 토크나이저 사용

![image.png](attachment:dddc9a85-30ad-43f1-bfcd-63b04eb26a0f:image.png)

![image.png](attachment:70a7d42c-0fa3-4899-b7ac-aafc1250b6f7:image.png)

![image.png](attachment:37a68cd5-0c7b-4ca9-b746-b7d18a07446b:image.png)

### 📌 10.3. 한국어 임베딩

- 코드 참고