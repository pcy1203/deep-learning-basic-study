[Google Colab](https://colab.research.google.com/drive/1ARPBH3agNvQ_xYzm_xyEmP7cWIzC4gFE?usp=sharing)

### **📌 4.1. ~ 4.2. 인공 신경망의 한계와 딥러닝 출현 / 딥러닝 구조**

- **딥러닝의 정의**
    - **퍼셉트론** : 딥러닝의 기원이 되는 알고리즘, 다수의 신호를 입력으로 받아 하나의 신호를 출력(`1` 또는 `0`) → but 비선형적 분류의 어려움 (XOR)
    - **다층 퍼셉트론(Multi-Layer Perceptron)** : 입력층과 출력층 사이에 하나 이상의 중간층(은닉층)을 두는 방식 → 비선형적 분류 가능
    - **딥러닝 = 심층 신경망(Deep Neural Network)** : 은닉층이 여러 개 있는 신경망
        
        ![image.png](attachment:f85c0dc8-fb79-4fbc-8cde-63f949a5d4d7:image.png)
        
    - 딥러닝의 이점 : 특성 추출(Feature Extraction - 패턴이나 규칙 찾기 → SVM, Naive-Bayes, Logistic Regression 특성 추출 과정 통합), 빅데이터 효율적 활용
- **딥러닝의 구조**
    - 💻 Code
        
        ```python
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_output):
                super(Net, self).__init__()
                self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 은닉층
                self.relu = torch.nn.ReLU(inplace=True)             # 활성화 함수
                self.out = torch.nn.Linear(n_hidden, n_output)      # 출력층
                self.softmax = torch.nn.Softmax(dim=n_output)        # 활성화 함수
            def forward(self, x):
                x = self.hidden(x)
                x = self.relu(x)
                x = self.out(x)
                x = self.softmax(x)
                return x
        ```
        
    - 1️⃣ **입력층(Input Layer)** : 데이터 입력
    - 2️⃣ **은닉층(Hidden Layer)** : 가중합 계산 → 활성화 함수 적용 → 출력층 전달
        - **가중합 (전달 함수)** : 입력 값에 가중치(연산 결과 조정)를 곱함 ($\sum_i w_ix_i + b$)
        - **활성화 함수** : 비선형 함수로 출력 값 변화
            - *활성화 함수의 종류?*
                - **Sigmoid** (0~1) : $sigmoid\ x = \frac{1}{1+e^{-x}}$
                - **Hyperbolic Tangent** (-1~1) : $tanh\ x = \frac{sinh\ x}{cosh\ x} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
                    
                    → 기울기 소멸 문제 (Vanishing Gradient Problem)
                    
                - **ReLU** (입력이 음수일 때 0 출력, 양수일 때 그대로 출력)
                    
                    → 빠른 학습 속도, 기울기 소멸 문제 X (주로 은닉층에서 사용)
                    
                - **Leaky ReLU** (입력이 음수일 때 매우 작은 수를 반환하는 ReLU)
                - **Softmax** (0~1, 출력 값들의 총합이 1이 되도록 정규화) : $y_k = \frac{e^{a_k}}{\sum_{i=1}^{n}e^{a_i}}$
                    
                    → 출력 노드의 활성화 함수
                    
    - 3️⃣ **출력층(Output Layer)** : 최종 결과값
        - **손실 함수(Loss Function)** : 출력 함수의 결과와 실제 값 간의 오차를 측정
            - *손실 함수의 종류?*
                - **평균 제곱 오차(Mean Squared Error, MSE)** : $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$
                    
                    → 회귀
                    
                    `torch.nn.MSELoss(reduction='sum')(예측, 타깃)`
                    
                - **크로스 엔트로피 오차(Cross Entropy Error, CEE)** : $CrossEntropy=-\sum_{i=1}^{n}y_i\ log\hat{y_i}$
                    
                    → 분류 (원-핫 인코딩 방식)
                    
                    `torch.nn.CrossEntropyLoss()(예측, 타깃)` * 예측은 (N, Class 개수), 타깃은 (N, 1)
                    
    - **순전파**(FeedForward, 예측 값 계산) → 손실 함수 계산 → **역전파**(BackPropagation, 손실 함수 비용이 0에 가까워지도록 가중치를 조정)
        
        ![image.png](attachment:bf287a60-7d32-4a45-a4e3-48b7b702b8d4:image.png)
        
        - *손실 함수는 오차를 구하는 방법, 역전파는 오차를 최소화하는 방향으로 가중치를 조정!*
        - **경사 하강법** : 학습률(learning rate)과 손실 함수의 순간 기울기(미분값)를 이용해 가중치 업데이트
- **딥러닝 성능 높이기**
    - **문제1. 과적합(Over-Fitting)** : 훈련 데이터를 과하게 학습 (실제 데이터 오차 증가) → *단순히 은닉층 개수가 많다고 좋은 게 아니다!*
        - **드롭아웃(Dropout)** : 신경망 모델의 학습 과정 중 일부 노드들을 임의로 학습에서 제외시킴 → `torch.nn.Dropout(사용하지 않을 비율)`
    - **문제2. 기울기 소멸 문제 발생** : 출력층 → 은닉층 전달 오차가 크게 줄어들어 학습이 되지 않는 현상
        - ReLU 활성화 함수 사용 (Sigmoid, Tanh 대신)
    - **문제3. 성능이 나빠지는 문제**
        
        ![image.png](attachment:82d4887e-9153-4c03-8c46-b983d2719628:image.png)
        
    - **배치 경사 하강법 (Batch Gradient Descent, BGD)** : 전체 데이터셋에 대해 손실 함수 계산 → 기울기를 한 번만 계산하여 모델 파라미터 업데이트 (전체 데이터셋에 대해 가중치 편미분)
    $W = W-a▽J(W,b)$
    - **확률적 경사 하강법(Stochastic Gradient Descent, SGD)** : 임의로 선택한 데이터에 대해 기울기를 계산하는 방법 (빠른 계산)
    - **미니 배치 경사 하강법 (Mini-batch Gradient Descent)** : 전체 데이터셋을 미니 배치 여러 개로 나누고, 미니 배치 하나마다 기울기를 구한 후 평균 기울기를 이용하여 모델 업데이트 (안정적)
        
        `DataLoader(데이터셋, batch_size=미니 배치 크기, shuffle=랜덤으로 섞으면 True)`
        
        - 💻 Code
            
            ```python
            class CustomDataset(Dataset):
                def __init__(self):
                    self.x_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                    self.y_data = [[12], [18], [11]]
                    def __len__(self):
                        return len(self.x_data)
                    def __getitem__(self, idx):
                        x = torch.FloatTensor(self.x_data[idx])
                        y = torch.FloatTensor(self.y_data[idx])
                        return x, y
            dataset = CustomDataset()
            dataloader = DataLoader(
                dataset, # 데이터셋
                batch_size=2, # 미니 배치 크기
                shuffle=True, # 데이터를 불러올 때마다 랜덤으로 섞기
            )
            ```
            
- **가중치 업데이트하기 - 옵티마이저**
    - **옵티마이저(Optimizer)** : 학습 속도와 운동량을 조정
        
        ![image.png](attachment:01e94b04-96b9-4a5a-8b72-092c20c7fdd9:image.png)
        
        - **AdaGrad (Adaptive Gradient)** : 가중치의 업데이트 횟수에 따라 학습률을 조정 (많이 변화하는 변수들의 학습률을 크게 함)
        - **AdaDelta (Adaptive Delta)** : AdaGrad에서 학습이 멈추는 문제 해결
        - **RMSProp** : AdaGrad에서 학습률이 작아지는 문제 해결
        - **Momentum** : 가중치를 수정하기 전에 이전 수정 방향을 참고하여 같은 방향으로 일정한 비율만 수정 (관성 효과) → SGD와 함께 사용
        - **Nesterov Accelerated Gradient (NAG)** : 모멘텀 값이 적용된 지점에서 기울기 값 계산
        - **Adam (Adaptive Moment Estimation)** : Momentum + RMSProp 장점 결합
            - 💻 Code
                
                ```python
                optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
                optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0) # 기본값 1.0
                optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01) # 기본값 1e-2
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # momentum 증가시키며 사용
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True) # nesterov 기본값 False
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 기본값 1e-3
                ```
                

### **📌 4.3. ~ 4.4. 딥러닝 알고리즘 / 우리는 무엇을 배워야 할까?**

- **딥러닝 알고리즘**
    - **심층 신경망(DNN)** : 입력층과 출력층 사이에 다수의 은닉층을 포함하는 인공 신경망
        
        → 많은 연산량, 기울기 소멸 문제 (드롭아웃, 렐루 함수, 배치 정규화 적용으로 해결)
        
    - **합성곱 신경망(Convolutional Neural Network)** : 합성곱층(Convolutional Layer)과 풀링층(Pooling Layer) 포함, 이미지 처리 성능이 좋은 인공 신경망
        
        → LeNet-5, AlexNet, VGG, GoogLeNet, ResNet
        
    - **순환 신경망(Recurrent Neural Network)** : 시계열 데이터(시간성 정보) 같이 시간 흐름에 따라 변화하는 데이터를 학습하기 위한 인공 신경망 (동적이고 길이가 가변적인 데이터)
        
        → 기울기 소멸 문제 (LSTM으로 메모리 개념 도입)
        
    - **제한된 볼츠만 머신(Restricted Boltzmann Machine)** : 가시층(Visible Layer)과 은닉층(Hidden Layer)로 구성된 모델 (가시층은 은닉층과만 연결)
        
        → 차원 감소, 분류, 선형 회귀 분석, 협업 필터링, 특성 값 학습, 주제 모델링 / 사전 학습 용도
        
    - **심층 신뢰 신경망(Deep Belief Network)** : 제한된 볼츠만 머신을 여러 층으로 쌓은 형태로 연결된 신경망
        
        → 비지도 학습 가능 (제한된 볼츠만 머신 사전 훈련 → 순차적 학습으로 계층적 구조 생성)
        

### **📌 5.1. ~ 5.2. 합성곱 신경망 / 합성곱 신경망 맛보기**

- **합성곱 신경망의 정의**
    - **합성곱 신경망(CNN)** : 음성 인식, 이미지/영상 인식에 주로 사용되는 신경망 (다차원 배열 처리 특화) → 데이터의 공간적 구조까지 활용!
- **합성곱 신경망의 구조**
    - 1️⃣ **입력층(Input Layer)** : 입력 데이터는 (높이, 너비, 채널) 값을 갖는 3차원 데이터
        - 그레이스케일이면 채널 `1`, 컬러(RGB)면 채널 `3`
    - 2️⃣ **합성곱층(Convolutional Layer)** : 입력 데이터에서 특성 추출
        - **커널(Kernel) = 필터** : 스트라이드(Stride, 간격)에 따라 이동하며 이미지의 모든 영역을 훑고 특성을 추출함 (주로 3x3, 5x5 커널 사용) → **특성 맵(Feature Map)**
        - **패딩** : 입력 데이터 주위를 0으로 채움
        - 입력 값의 채널이 1이 아닌 경우 : 각 채널(RGB 각각)에 서로 다른 가중치로 합성곱을 적용 후 결과를 더함
        - 2개 이상의 필터를 사용하는 경우 : 필터 각각이 특성 맵의 채널이 됨
        - 출력 데이터의 크기 : (W, H, D) → $(\frac{W-F+2P}{S}+1, \frac{H-F+2P}{S}+1, K)$
            - 필터 개수 K, 필터 크기 F, 스트라이드 S, 패딩 P
    - 3️⃣ **풀링층(Pooling Layer)** : 특성 맵의 차원을 다운 샘플링하여 연산량 감소
        - 최대 풀링(Max Pooling), 평균 풀링(Average Pooling), 최소 풀링
        - **`MaxPool2d`** : 출력 데이터 크기 축소, 특정 데이터 강조
        - 합성곱층 + 풀링층 → 입력 이미지의 주요 특성 벡터(Feature Vector) 추출
    - 4️⃣ **완전연결층(Fully Connected Layer)** : Flatten - 1차원 벡터로 변환
    - 5️⃣ **출력층 (Output Layer)** : 활성화 함수(Softmax)의 최종 결과 출력 (각 레이블에 속할 확률)
        - 데이터를 배열 형태로 변환하여 작업, 하이퍼파라미터 값에 따라 출력 크기 달라짐
- **합성곱 차원** ← 필터의 이동 방향 수, 출력 형태에 따라
    
    ![image.png](attachment:c9eeb7be-701f-4bb5-b8d2-a11c78ec0cf2:image.png)
    
    - **1D 합성곱** : 1개 방향 움직임 (시간 축으로 좌우 이동) `W x (k, k) → W`
    - **2D 합성곱** : 2개 방향 움직임 `(W, H) x (k, k) → (W, H)`
    - **3D 합성곱** : 3개 방향 움직임 `(W, H, L) x (k, k, d) → (W, H, L) (d < L)`
        - 3D 입력을 갖는 2D 합성곱 `(W, H, L) x (k, k, L) → (W, H)` (LeNet-5, VGG)
        - 1x1xL 필터 : 연산량을 감소 (깊이 없앰)
    - **BatchNorm2d** : 각 배치 단위별로 데이터가 다양한 분포를 가지더라도 평균과 분산을 이용하여 정규화 (평균 0, 표준편차 1로 데이터 분포 조정)
    - 💻 Code
        
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        
        import torch
        import torch.nn as nn  # 딥러닝 모델 구성
        from torch.autograd import Variable
        import torch.nn.functional as F
        
        import torchvision
        import torchvision.transforms as transforms # 데이터 전처리
        from torch.utils.data import Dataset, DataLoader
        
        from google.colab import drive
        drive.mount("/content/drive")
        
        import os
        os.chdir("drive/MyDrive/프로메테우스 스터디")
        
        # GPU 설정
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 데이터 로드
        train_dataset = torchvision.datasets.FashionMNIST("chap05/data", download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = torchvision.datasets.FashionMNIST("chap05/data", download=True,
                        train=False, transform=transforms.Compose([transforms.ToTensor()]))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
        
        labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat',
                      5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
        
        # 데이터 확인하기
        fig = plt.figure(figsize=(8,8));
        columns = 4;
        rows = 5;
        for i in range(1, columns*rows +1):
            img_xy = np.random.randint(len(train_dataset))
            img = train_dataset[img_xy][0][0,:,:]
            fig.add_subplot(rows, columns, i)
            plt.title(labels_map[train_dataset[img_xy][1]])
            plt.axis('off')
            plt.imshow(img, cmap='gray')
        plt.show()
        
        # 모델1
        class FashionDNN(nn.Module):
            def __init__(self):  # 속성값 초기화 (객체 생성과 함께 호출)
                super(FashionDNN, self).__init__()  # 부모 클래스 상속
                self.fc1 = nn.Linear(in_features=784, out_features=256)  # 선형 회귀 모델
                self.drop = nn.Dropout(0.25)
                self.fc2 = nn.Linear(in_features=256, out_features=128)
                self.fc3 = nn.Linear(in_features=128, out_features=10)
        
            def forward(self, input_data):  # 순전파 학습 진행
                out = input_data.view(-1, 784)
                out = F.relu(self.fc1(out))
                out = self.drop(out)
                out = F.relu(self.fc2(out))
                out = self.fc3(out)
                return out
                
        # 모델2
        class FashionCNN(nn.Module):
            def __init__(self):
                super(FashionCNN, self).__init__()
                self.layer1 = nn.Sequential(  # nn.Sequential : 네트워크 모델 정의
                    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 채널 = 깊이
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
                self.drop = nn.Dropout2d(0.25)
                self.fc2 = nn.Linear(in_features=600, out_features=120)
                self.fc3 = nn.Linear(in_features=120, out_features=10)  # 마지막 계층의 out_features = 클래스 개수
        
            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = self.drop(out)
                out = self.fc2(out)
                out = self.fc3(out)
                return out         
        
        # 손실 함수, 학습률, 옵티마이저 정의
        learning_rate = 0.001
        model = FashionCNN()
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()  # 분류 문제 손실 함수
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(model)
        
        # 모델 학습
        num_epochs = 5
        count = 0
        loss_list = []
        iteration_list = []
        accuracy_list = []
        
        predictions_list = []
        labels_list = []
        
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
        
                # torch.autograd.Variable 이용해 역전파를 위한 미분 값 자동 계산
                train = Variable(images.view(100, 1, 28, 28))
                labels = Variable(labels)
        
                outputs = model(train)  # 학습 데이터를 모델에 적용
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count += 1
        
                if not (count % 50):
                    total = 0
                    correct = 0
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)  # GPU 사용
                        labels_list.append(labels)
                        test = Variable(images.view(100, 1, 28, 28))
                        outputs = model(test)
                        predictions = torch.max(outputs, 1)[1].to(device)
                        predictions_list.append(predictions)
                        correct += (predictions == labels).sum()
                        total += len(labels)
        
                    accuracy = correct * 100 / total # 정확도
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)
        
                if not (count % 500):
                    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
        ```
        
- 💻 **코드 이해하기**
    - GPU 사용
        
        ```python
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net()
        if torch.cuda.device_count() > 1:  # 다수의 GPU 사용
            model = nn.DataParallel(net)  # 배치 크기가 알아서 각 GPU로 분배
        model.to(device)
        ```
        
    - 데이터셋
        - `fashion_mnist` 데이터셋 : 28x28 픽셀 이미지 7만 개
        - `train_images` 넘파이 배열
        - `train_labels` 0~9 정수 값 → 이미지의 클래스 레이블
        - 입력값 : (N, 1, 28, 28) 형태 → 이미지 텐서와 정수형 레이블 튜플
        - `torchvision.datasets.데이터이름("내려받을 위치", download=True, transform=transforms.Compose([transforms.ToTensor()]))`
        - `torch.utils.data.DataLoader(데이터셋, batch_size=배치 크기)`  : 배치 크기 단위로 데이터 묶어서 불러오기
    - 랜덤값 받기
        - `np.random.randint(시작, 끝+1)`
        - `np.random.randint(끝)` : 0~끝
        - `np.random.rand(행렬 크기)` : 0~1 사이 표준정규분포
        - `np.random.randn(행렬 크기)` : 평균 0, 표준편차 1 가우시안 정규분포 난수
        - `np.arange(시작, 끝+1, 건너뛰기)`
    - 모델 : `torch.nn.Module` 상속
        - `nn.Linear(in_features=입력 크기, out_features=출력 크기)`
        - `nn.Dropout(드롭아웃 비율)` 비율만큼 텐서의 값이 0이 되고, 0이 되지 않는 값은 1/(1-p)배
    - 활성화 함수 (두 가지 방법)
        - `F.relu()`  → `forward()` 함수
        - `nn.ReLU()` → `__init__()` 함수
    - `torch.nn` vs. `torch.nn.functional`
        - 💻 Code
            
            ```python
            
            # torch.nn.xx
            inputs = torch.randn(64, 3, 244, 244)
            conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
            outputs = conv(inputs)
            layer = nn.Conv2d(1, 1, 3)
            
            #-------------
            
            # nn.functional.xx
            import torch.nn.functional as F
            inputs = torch.randn(64, 3, 244, 244)
            weight = torch.randn(64, 3, 3, 3)
            bias = torch.randn(64)
            # 입력과 가중치 자체 넣기
            outputs = F.conv2d(inputs, weight, bias, padding=1)
            ```
            
        
        | **torch.nn** | **torch.nn.functional** |
        | --- | --- |
        | 클래스 사용 | 함수 사용 |
        | 하이퍼파라미터 전달 후 함수 호출을
        통해 데이터 전달 | 함수 호출할 때 하이퍼파라미터,
        데이터 전달 |
        | nn.Sequential 내에 위치 | nn.Sequential 내에 위치 불가 |
        | 파라미터 새로 정의할 필요 없음 | 가중치 전달할 때마다 가중치
        값을 새로 정의 |

### **📌 5.3. ~ 5.5. 전이 학습 / 설명 가능한 CNN / 그래프 합성곱 네트워크**

- **전이 학습의 정의**
    - **전이 학습 (Transfer Learning)** : 아주 큰 데이터셋으로 훈련된 모델(사전 훈련된 모델)의 가중치를 가져와 해결하는 과제에 맞게 보정하여 사용하는 것
        - 합성곱층(합성곱층 + 풀링층) + 완전연결층(데이터 분류기)
    - **특성 추출(Feature Extractor)** : 사전 훈련된 모델 → 마지막 완전연결층 부분만 새로 학습 (나머지 계층 가중치 고정)
        
        ![image.png](attachment:6fa2cde5-372e-42cc-a06d-324b44b8e0ac:image.png)
        
    - **미세 조정 기법(Fine-Tuning)** : 사전 훈련된 모델, 합성곱층, 완전연결층의 가중치 업데이트하여 훈련
    - 모델
        
        ```python
        import torchvision.models as models  # 무작위 가중치 모델
        resnet18 = models.resnet18()
        alexnet = models.alexnet()
        vgg16 = models.vgg16()
        squeezenet = models.squeezenet1_0()
        densenet = models.densenet161()
        inception = models.inception_v3()
        googlenet = models.googlenet()
        shufflenet = models.shufflenet_v2_x1_0()
        mobilenet_v2 = models.mobilenet_v2()
        mobilenet_v3_large = models.mobilenet_v3_large()
        mobilenet_v3_small = models.mobilenet_v3_small()
        resnext50_32x4d = models.resnext50_32x4d()
        wide_resnet50_2 = models.wide_resnet50_2()
        mnasnet = models.mnasnet1_0()
        
        import torchvision.models as models  # 사전 학습된 모델
        resnet18 = models.resnet18(pretrained=True)
        alexnet = models.alexnet(pretrained=True)
        squeezenet = models.squeezenet1_0(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        densenet = models.densenet161(pretrained=True) 
        inception = models.inception_v3(pretrained=True)
        googlenet = models.googlenet(pretrained=True)
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
        mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
        resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
        wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
        mnasnet = models.mnasnet1_0(pretrained=True)
        ```
        
- **미세 조정 기법**
    - 사전 학습된 모델을 목적에 맞게 재학습 혹은 학습된 가중치 일부 재학습
        
        
        | 재학습할 부분 | **데이터셋 大** | **데이터셋 小** |
        | --- | --- | --- |
        | **유사성 大** | 합성곱층 뒷부분 + 완전연결층 | 완전연결층 (과적합 조심) |
        | **유사성 小** | 모델 전체 재학습(…) | 합성곱층 일부분 + 완전연결층 |
    - 💻 Code
        
        ResNet18 : 50개의 계층으로 구성된 CNN, ImageNet 데이터베이스 영상 이용하여 훈련
        
        ```python
        !pip install opencv-python
        
        import os
        import time
        import copy
        import glob
        import cv2  # 라이브러리
        import shutil
        
        import torch
        import torchvision  # 컴퓨터 비전(computer vision) 용도의 패키지
        import torchvision.transforms as transforms  # 데이터 전처리 패키지
        import torchvision.models as models  # 파이토치 네트워크 패키지
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        
        import matplotlib.pyplot as plt
        
        from google.colab import drive
        drive.mount("/content/drive")
        
        import os
        os.chdir("drive/MyDrive/프로메테우스 스터디")
        
        # 데이터 받아오기
        data_path = 'chap05/data/catanddog/train'
        
        transform = transforms.Compose([
                        transforms.Resize([256, 256]),  # 이미지 크기 조정
                        transforms.RandomResizedCrop(224),  # 이미지를 랜덤한 크기 및 비율로 자르기 (데이터 확장 용도)
                        transforms.RandomHorizontalFlip(),  # 랜덤하게 수평으로 뒤집기
                        transforms.ToTensor()
        ])
        train_dataset = torchvision.datasets.ImageFolder(
                        data_path,  # 불러올 대상(경로)
                        transform=transform  # 불러올 방법
        )
        train_loader = torch.utils.data.DataLoader(
                       train_dataset,
                       batch_size=32,  # 한 번에 불러올 데이터 양
                       #  num_workers=8,  # 하위 프로세스 개수 (데이터 불러올 때)
                       shuffle=True  # 데이터 무작위로 섞기
        )
        print(len(train_dataset))
        
        # 데이터 출력해보기
        samples, labels = next(iter(train_loader))
        classes = {0:'cat', 1:'dog'}  # 레이블
        fig = plt.figure(figsize=(16,24))
        for i in range(24):  # 24개 이미지
            a = fig.add_subplot(4,6,i+1)
            a.set_title(classes[labels[i].item()])  # 레이블 정보(클래스) 함께 출력
            a.axis('off')
            a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))  # 행렬 차원 바꾸기
        plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
        
        # 사전 학습된 모델
        model = models.resnet18(pretrained=True)  # pretrained=True : 사전 학습된 가중치 사용
        
        for name, param in model.named_parameters():  # 모델 파라미터 값
            if param.requires_grad:
                print(name, param.data)
        
        # 완전연결층
        model.fc = nn.Linear(512, 2)  # 2개 클래스
        
        for param in model.parameters():  # 합성곱층 가중치 고정
            param.requires_grad = False  # 역전파 중 파라미터 변화 계산 X
        
        for param in model.fc.parameters():  # 완전연결층 학습
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(model.fc.parameters())
        cost = torch.nn.CrossEntropyLoss()  # 손실 함수
        print(model)
        
        # 모델 훈련
        def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=13, is_train=True):
            since = time.time()  # 컴퓨터의 현재 시각
            acc_history = []
            loss_history = []
            best_acc = 0.0
        
            for epoch in range(num_epochs):  # 에포크만큼 반복
                print('Epoch {}/{}'.format(epoch, num_epochs-1))
                print('-' * 10)
        
                running_loss = 0.0
                running_corrects = 0
        
                for inputs, labels in dataloaders:  # 데이터로더에 전달된 데이터만큼 반복
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    model.to(device)
                    optimizer.zero_grad()  # 기울기를 0으로 설정
                    outputs = model(inputs)  # 순전파 학습
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    loss.backward()  # 역전파 학습
                    optimizer.step()
        
                    running_loss += loss.item() * inputs.size(0)  # 출력 결과와 레이블의 오차 누적
                    running_corrects += torch.sum(preds == labels.data)  # 출력 결과와 레이블이 동일한지 확인한 결과 누적
        
                epoch_loss = running_loss / len(dataloaders.dataset)  # 평균 오차
                epoch_acc = running_corrects.double() / len(dataloaders.dataset)  # 평균 정확도
        
                print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
                if epoch_acc > best_acc:
                   best_acc = epoch_acc
        
                acc_history.append(epoch_acc.item())
                loss_history.append(epoch_loss)
                torch.save(model.state_dict(), os.path.join('chap05/data/catanddog/',  '{0:0=2d}.pth'.format(epoch)))
                print()
        
            time_elapsed = time.time() - since # 실행 시간(학습 시간)
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best Acc: {:4f}'.format(best_acc))
            return acc_history, loss_history  # 정확도, 오차
        
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)  # 파라미터 학습 결과 저장
                print("\t", name)
        
        optimizer = optim.Adam(params_to_update)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        train_acc_hist, train_loss_hist = train_model(model, train_loader, criterion, optimizer, device)
        
        # -----------------------------------------------------------------------------
        
        test_path = 'chap05/data/catanddog/test'
        
        transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
        ])
        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            num_workers=1,
            shuffle=True
        )
        
        print(len(test_dataset))
        
        def eval_model(model, dataloaders, device):
            since = time.time()
            acc_history = []
            best_acc = 0.0
        
            saved_models = glob.glob('chap05/data/catanddog/' + '*.pth')  # 원하는 파일 추출
            saved_models.sort()
            # print('saved_model', saved_models)
        
            for model_path in saved_models:
                print('Loading model', model_path)
        
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model.to(device)
                running_corrects = 0
        
                for inputs, labels in dataloaders:  # 테스트 반복
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    with torch.no_grad():  # autograd 사용 X
                         outputs = model(inputs)
        
                    _, preds = torch.max(outputs.data, 1)  # 배열의 최댓값이 들어 있는 인덱스
                    preds[preds >= 0.5] = 1  # >= 0.5 : 올바르게 예측
                    preds[preds < 0.5] = 0  # < 0.5 : 틀리게 예측
                    running_corrects += preds.eq(labels).int().sum()
        
                epoch_acc = running_corrects.double() / len(dataloaders.dataset)  # 정확도 계산
                print('Acc: {:.4f}'.format(epoch_acc))
        
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    acc_history.append(epoch_acc.item())
                    print()
        
                time_elapsed = time.time() - since
                print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('Best Acc: {:4f}'.format(best_acc))
        
                return acc_history  # 계산된 정확도
        
        val_acc_hist = eval_model(model, test_loader, device)
        
        plt.plot(train_acc_hist)
        plt.plot(val_acc_hist)
        plt.show()
        plt.plot(train_loss_hist)
        plt.show()
        
        def im_convert(tensor):
            image = tensor.clone().detach().numpy()
            image = image.transpose(1, 2, 0)
            image = image * (np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5)))
            image = image.clip(0, 1)
            return image
        
        classes = {0:'cat', 1:'dog'}
        
        dataiter = iter(test_loader)  # 테스트 데이터셋
        images, labels = next(dataiter)
        output = model(images)
        _, preds = torch.max(output, 1)
        
        fig = plt.figure(figsize=(25,4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            plt.imshow(im_convert(images[idx]))
            a.set_title(classes[labels[i].item()])
        ax.set_title("{}({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].
                     item()])), color=("green" if preds[idx]==labels[idx] else "red"))
        plt.show()
        plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
        ```
        
    - 💻 Code (RandomResizedCrop 데이터 확장)
        
        ```python
        !pip install mxnet
        !pip install --user mxnet
        
        import matplotlib.pyplot as plt
        import mxnet as mx
        from mxnet.gluon.data.vision import transforms
        
        example_image = mx.image.imread("chap05/data/cat.jpg")
        plt.imshow(example_image.asnumpy())
        
        def show_images(imgs, num_rows, num_cols, scale=2):
            aspect_ratio = imgs[0].shape[0]/imgs[0].shape[1]  # 확장할 이미지의 크기 조정
            figsize = (num_cols * scale, num_rows * scale * aspect_ratio)
            _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
            for i in range(num_rows):
                for j in range(num_cols):
                    axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
                    axes[i][j].axes.get_xaxis().set_visible(False)
                    axes[i][j].axes.get_yaxis().set_visible(False)
            plt.subplots_adjust(hspace=0.1, wspace=0)
            return axes
        
        def apply(img, aug, num_rows=2, num_cols=4, scale=3):
            Y = [aug(img) for _ in range(num_rows * num_cols)]  # 데이터 확장 적용
            show_images(Y, num_rows, num_cols, scale)
            
        shape_aug = transforms.RandomResizedCrop(size=(200, 200),
                                                 scale=(0.1, 1),
                                                 ratio=(0.5, 2))
        apply(example_image, shape_aug)
        ```
        
- **특성 맵 시각화**
    - **설명 가능한 CNN(Explainable CNN)** : 딥러닝 처리 결과를 사람이 이해할 수 있는 방식으로 제시하는 기술 → 필터 시각화, 특성 맵 시각화
    - **특성 맵(Feature Map)** : 필터를 입력에 적용한 결과 *(입력 특성을 감지하는 방법!*)
    - 💻 Code
        
        ```python
        !pip install pillow
        
        import matplotlib.pyplot as plt
        from PIL import Image
        import cv2
        import torch
        import torch.nn.functional as F
        import torch.nn as nn
        from torchvision.transforms import ToTensor
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.models as models
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class XAI(torch.nn.Module):
            def __init__(self, num_classes=2):
                super(XAI, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),  # inplace=True는 기존의 데이터를 연산의 결괏값으로 대체하는 것을 의미
                    nn.Dropout(0.3),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512, 512, bias=False),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
        
            def forward(self, x):
                x = self.features(x)
                x = x.view(-1, 512)
                x = self.classifier(x)
                return F.log_softmax(x)
                
        model = XAI()  # model이라는 이름의 객체를 생성
        model.to(device)
        model.eval()
        
        class LayerActivations:
            features = []
            def __init__(self, model, layer_num):
                self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        
            def hook_fn(self, module, input, output):
                self.features = output.detach().numpy()
        
            def remove(self):  # hook 삭제
                self.hook.remove()
                
        img = cv2.imread("chap05/data/cat.jpg")
        plt.imshow(img)
        img = cv2.resize(img, (100,100), interpolation=cv2.INTER_LINEAR)
        img = ToTensor()(img).unsqueeze(0)
        print(img.shape)
        
        result = LayerActivations(model.features, 0)  # 0번째 Conv2d 특성 맵 확인
        
        model(img)
        activations = result.features
        
        fig, axes = plt.subplots(4, 4)
        fig = plt.figure(figsize=(12,8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for row in range(4):
            for column in range(4):
                axis = axes[row][column]
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                axis.imshow(activations[0][row*10+column])
        plt.show()
        
        result = LayerActivations(model.features, 20)  # 20번째 Conv2d 특성 맵 확인
        
        model(img)
        activations = result.features
        
        fig, axes = plt.subplots(4, 4)
        fig = plt.figure(figsize=(12,8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for row in range(4):
            for column in range(4):
                axis = axes[row][column]
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                axis.imshow(activations[0][row*10+column])
        plt.show()
        
        result = LayerActivations(model.features, 40)  # 40번째 Conv2d 특성 맵 확인
        
        model(img)
        activations = result.features
        
        fig, axes = plt.subplots(4, 4)
        fig = plt.figure(figsize=(12,8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for row in range(4):
            for column in range(4):
                axis = axes[row][column]
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                axis.imshow(activations[0][row*10+column])
        plt.show()
        ```
        
- **그래프 합성곱 네트워크**
    - **그래프 합성곱 네트워크(Graph Convolutional Network)** : 에지로 연결된 노드의 집합 (노드 + 에지)
    - **그래프 신경망(Graph Neural Network, GNN)** : 그래프 구조 신경망
        
        ![image.png](attachment:158717a0-c9ad-46f5-afa9-24b000b69225:image.png)
        
    - **그래프 표현 방법** : 인접 행렬(Adjacency Matrix), 특성 행렬(Feature Matrix, 이용할 특성 선택)
    - **리드아웃(Readout)** : 특성 행렬을 하나의 벡터로 변환하는 함수 (특성 벡터 평균을 구하여 그래프 전체를 하나의 벡터로 표현)

### **🤔 더 알아보기**

- 음성인식에서도 CNN을 사용한다?

[구글 브레인 팀에게 배우는 딥러닝 with TensorFlow.js: 4.4.1 스펙트로그램: 사운드를 이미지로 표현하기 - 1](https://thebook.io/080237/0215/)

- 미니 배치 경사 하강법이 빠른 이유는? → 행렬 연산을 활용한 병렬 처리 가능
- 1D, 2D, 3D 합성곱이 따로따로 필요한가? → 필터의 이동 방향에 따른 구분

[11-02 자연어 처리를 위한 1D CNN(1D Convolutional Neural Networks)](https://wikidocs.net/80437)

- DataLoader는 어떻게 사용하지?

[5. Dataset 과 DataLoader](https://wikidocs.net/156998)