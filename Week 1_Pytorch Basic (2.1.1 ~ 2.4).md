### 📌 2.1. 파이토치 개요

- **파이토치(PyTorch)** : GPU에서 텐서 조작 및 동적 신경망 구축이 가능한 프레임워크 (유연성과 속도를 제공하는 딥러닝 라이브러리)
    - PyTorch의 데이터 형태는 **텐서(Tensor)** = 단일 데이터 형식의 다차원 행렬
        - `torch.tensor()` 사용
        - 1차원 배열 형태(스토리지) 저장, 스트라이드 = 각 차원에서 다음 요소를 얻기 위해 건너뛰어야 하는 요소 개수
    - `.cuda()`를 사용하여 GPU로 연산을 빠르게 수행할 수 있도록 함
    - 훈련을 반복할 때마다 모델의 네트워크 조작이 가능한 동적 신경망
- **파이토치 아키텍처** : 파이토치 API (사용자 사용) - 파이토치 엔진 (다차원 텐서 및 자동 미분 처리) - 연산 처리
    - 파이토치 API : `torch` (GPU 지원 텐서 패키지), `torch.autograd` (자동 미분 패키지), `torch.nn` (신경망 구축 및 훈련 패키지), `torch.multiprocessing` (파이썬 멀티프로세싱 패키지), `torch.utils` (DataLoader 및 기타 유틸리티 패키지)
    - 파이토치 엔진 : Autograd C++ (미분 자동 계산), Aten C++ (C++ 텐서 라이브러리 제공), JIT C++ (계산 최적화 JIT 컴파일러)
    - 연산 처리 : 다차원 텐서 연산 처리 (C 또는 CUDA 패키지)

### 📌 2.2. 파이토치 기본 문법

데이터셋 로드 → 모델 정의 (파라미터 정의) → 모델 훈련 → 모델 평가

- **텐서 사용법**
    - `torch.tensor(배열)` : 텐서 생성
        - `torch.tensor(배열, device="cuda:0", dtype=torch.float64)`
        - `텐서.numpy()` 로 ndarray 변환 가능
        - `torch.FloatTensor`, `torch.DoubleTensor`, `torch.LongTensor`
    - `텐서[인덱스]` 혹은 `텐서[인덱스:인덱스]` : 인덱스 조작
    - 텐서 간의 사칙 연산 가능 (단, 타입이 다르면 불가)
    - `텐서.view(값, ...)` : M x … x N 행렬로 차원 조작
        - `-1` 지정 시 다른 차원으로부터 값을 유추
    - 이외 : `stack` `cat` `t` `transpose`
- **데이터셋 로드**
    - 단순하게 파일 불러오기 : `pd.read_csv('파일명.csv')`
        - `x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()`
    - 커스텀 데이터셋(데이터를 조금씩 나누어 불러오기) :
        
        ```python
        import pandas as pd
        import torch
        from torch.utils.data import Dataset
        from torch.utils.data import DataLoader
        
        class CustomDataset(Dataset):
            def __init__(self, csv_file):
                self.label = pd.read_csv(csv_file)
        
            def __len__(self):
                return len(self.label)
        
            def __getitem__(self, idx):
                sample = torch.tensor(self.label.iloc[idx,0:3]).int()
                label = torch.tensor(self.label.iloc[idx,3]).int()
                return sample, label
        
        tensor_dataset = CustomDataset('파일명.csv')
        dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)
        ```
        
    - 토치비전(torchvision) : 데이터셋 패키지 (requests 라이브러리 별도 설치)
        
        ```python
        import torchvision.transforms as transforms
        
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))
        ]) # 평균이 0.5, 표준편차가 1.0이 되도록 데이터의 분포(normalize)를 조정
        
        from torchvision.datasets import MNIST
        import requests
        download_root = '경로'
        
        train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
        valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
        ```
        
- **모델 정의**
    - 단순 신경망 : `nn.Linear(in_features=1, out_features=1, bias=True)`
    - `nn.Module()` 상속
        
        ```python
        class MLP(Module):
            def __init__(self, inputs):
                super(MLP, self).__init__()
                self.layer = Linear(inputs, 1)
                self.activation = Sigmoid()
        
            def forward(self, X):
                X = self.layer(X)
                X = self.activation(X)
                return X
        ```
        
    - Sequential 신경망 정의
        - `model.modules()` 네트워크 모든 노드 반환, `model.children()` 같은 수준 하위 노드 반환
        
        ```python
        import torch.nn as nn
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
        
                self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
        
                self.layer3 = nn.Sequential(
                    nn.Linear(in_features=30*5*5, out_features=10, bias=True),
                    nn.ReLU(inplace=True)
                )
        
                def forward(self, x):
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = x.view(x.shape[0], -1)
                    x = self.layer3(x)
                    return x
        model = MLP()
        ```
        
    - 함수로 신경망 정의
        
        ```python
        def MLP(in_features=1, hidden_features=20, out_features=1):
            hidden = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
            activation = nn.ReLU()
            output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
            net = nn.Sequential(hidden, activation, output)
            return net
        ```
        
- **모델 파라미터 정의**
    - 손실 함수(loss function) : 출력(wx + b)과 실제 값(정답)(y) 사이의 오차
        - 이진 분류 BCELoss, 다중 분류 CrossEntropyLoss, 회귀 모델 MSELoss
    - 옵티마이저(optimizer) : 모델 업데이트 방법 결정 (step() 메서드를 통해 전달받은 파라미터 업데이트)
        - `torch.optim.Optimizer(params, defaults)` 기본 클래스
        - `zero_grad()` 메서드 : 파라미터 기울기를 0으로 만듦
        - Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSProp, Rprop, SGD
    - 학습률 스케줄러(learning rate scheduler) : 지정한 에포크를 지날 때마다 학습률 감소 (초기 빠른 학습, 전역 최소점 근처에서 최적점을 찾을 수 있도록 함)
        - LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
    - 지표(metrics) : 훈련과 테스트 단계 모니터링
    - 예시
        
        ```python
        from torch.optim import optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        for epoch in range(1, 100+1): 
            for x, y in dataloader:
                optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
        scheduler.step()
        ```
        
- **모델 훈련**
    - 모델, 손실 함수, 옵티마이저 정의
    - `optimizer.zero_grad()` 기울기 초기화 (누적이 필요하지 않은 경우, 배치가 반복될 때마다)
    - `output = model(input)`  출력 계산, `loss = loss_fn(output, target)` 오차 계산
    - `loss.backward()` 기울기 값 계산(역전파 학습)
    - `optimizer.step()` 기울기 업데이트
- **모델 평가**
    - 함수 이용 (torchmetrics 설치) : `torchmetrics.functional.accuracy(pred, target)`
    - 모듈 이용
        
        ```python
        import torch
        import torchmetrics
        metric = torchmetrics.Accuracy()
        
        n_batches = 10
        for i in range(n_batches):
            preds = torch.randn(10, 5).softmax(dim=-1)
            target = torch.randint(5, (10,))
        
            acc = metric(preds, target)
            print(f"Accuracy on batch {i}: {acc}")
        
        acc = metric.compute()
        print(f"Accuracy on all data: {acc}")
        ```
        
- **훈련 과정 모니터링**
    - 텐서보드 설정, 기록, 모델 구조 살펴보기 (tensorboard 설치)
        - `tensorboard --logdir=<저장 위치> --port=6006`
        - `model.train()` 모델 훈련(훈련 데이터셋 사용), 드롭아웃 활성화, `model.eval()` 모델 평가(검증과 테스트 데이터셋 사용), 모든 노드 사용, 역전파 불필요
    
    ```python
    import torch
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("저장 위치")
    
    for epoch in range(num_epochs):
        model.train()  # 학습 모드로 전환(dropout=True)
        batch_loss = 0.0
    
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device).float(), y.to(device).float()
            outputs = model(x)
            loss = criterion(outputs, y)
            writer.add_scalar("Loss", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    writer.close()
    
    # ----------------------------------------------
    
    model.eval() # 검증 모드로 전환(dropout=False)
    with torch.no_grad():
        valid_loss = 0
    
        for x, y in valid_dataloader:
            outputs = model(x)
            loss = F.cross_entropy(outputs, y.long().squeeze())
            valid_loss += float(loss)
            y_hat += [outputs]
    
    valid_loss = valid_loss / len(valid_loader)
    ```
    

### 📌 2.3. 실습 환경 설정

- **실습 환경 구축**
    - 아나콘다 설치 https://www.anaconda.com/download
    - `conda create -n <가상환경 이름> python=3.9.0` : 가상환경 생성
    - `conda env list` : 생성된 가상환경 확인
    - `activate <가상환경 이름>` : 가상환경 활성화
    - `conda env remove -n <가상환경 이름>` : 가상환경 삭제
    - `python -m ipykernel install --user --name <가상환경 이름> --display-name "<가상환경 이름>"` : 가상환경에 커널 연결
    - `conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 -c pytorch` : Pytorch 설치
    - `jupyter notebook` : 주피터 노트북 실행

### 📌 2.4. 파이토치 코드 맛보기

- **파이토치 코드 맛보기**
    - matplotlib (그래프), seaborn (시각화), scikit-learn (머신러닝) 설치
    - 데이터 파악하기, 데이터 전처리(preprocessing)
        - 범주형 데이터 → dataset(category) → 넘파이 배열 → 텐서
        - `astype()` 메서드로 데이터를 범주형으로 전환
        - `cat.codes()` 범주형 데이터를 숫자(넘파이 배열)로 변환
        - `np.stack` 새로운 축으로 합치기, `np.concatenate` 축 기준 연결
        - `get_dummies()` 넘파이 배열로 변환 (가변수로 만듦)
        - `ravel()` `reshape()` `flatten()` 텐서의 차원 바꾸기
    - 워드 임베딩 : 유사한 단어끼리 유사하게 인코딩되도록 표현하는 방법 (임베딩 크기 정의 필요, 칼럼 고유 값 수 / 2 많이 사용)
    - 모델의 네트워크 계층
        - Linear : 선형 계층, 선형 변환 진행 (y = Wx + b)
        - ReLU : 활성화 함수
        - BatchNorm1d : 배치 정규화
        - Dropout : 과적합 방지
- **딥러닝 분류 모델의 성능 평가 지표**
    - TP (실제 == 예측 == True), TN (실제 == 예측 == False), FP (실제 False, 예측 True, Type I 오류), FN (실제 True, 예측 False, Type II 오류)
    - 정확도(accuracy) = TP + TN / TP + TN + FP + FN
    - 재현율(recall) = TP / TP + FN (정답 True일 때 예측 True)
    - 정밀도(precision) = TP / TP + FP (예측 True일 때 실제 True)
    - F1-Score : 2 x Precision x Recall / (Precision + Recall) → 조화평균

### **🤔 더 알아보기**

[옵티마이저(Optimizer)와 학습률(Learning Rate)](https://naver.me/GpCEzizb)

[[pytorch] Learning Rate Scheduler (학습률 동적 변경)](https://naver.me/IgJDDd94)

[경사 하강법 Gradient Descent 에 대한 수학적 이해와 활용](https://data-scientist-jeong.tistory.com/46)