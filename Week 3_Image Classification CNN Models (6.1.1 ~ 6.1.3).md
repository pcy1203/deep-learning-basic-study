코드!

[Google Colab](https://colab.research.google.com/drive/1BVmjUbRuBqh3hLlyMNxUj7cPMeqlpixk?usp=sharing)

### 📌 6.1. 이미지 분류를 위한 신경망

**6.1.1. LeNet-5 (1995)**

- 합성곱(Convolution), 다운 샘플링(Sub-Sampling) 반복 → 마지막 완전연결층 (분류)
- 구조 (Conv 2개, MaxPool 2개)
    - Input : 32 x 32 (x 3)
    - Conv : 6 [5 x 5 Filters (Stride = 1, Activation = ReLU)] → 6 x 28 x 28
    - MaxPool : 6 [2 x 2 Filters (Stride = 2)] → 6 x 14 x 14
    - Conv : 16 [5 x 5 Filters (Stride = 1, Activation = ReLU)] → 16 x 10 x 10
    - MaxPool : 16 [2 x 2 Filters (Stride = 2)] → 16 x 5 x 5
    - Fully Connected : 120 (Activation = ReLU)
    - Fully Conntected : 84 (Activation = ReLU)
    - Fully Connected : 10 혹은 클래스 개수 (Activation = Softmax)

![image.png](attachment:e912f33a-3949-41b4-9346-72f0ad946046:image.png)

**6.1.2. AlexNet (2012)**

- 병렬 구조 (GPU 두 개 기반)
- 구조 (Conv 5개, MaxPool 3개)
    - Input : 227 x 227 (x 3)
        - 이미지 크기가 크지 않으면 풀링층 때문에 크기가 계속 줄어듦
    - Conv : 96 [11 x 11 Filters (Stride = 4, Activation = ReLU)] → 96 x 55 x 55
        - GPU-1 컬러 무관 정보, GPU-2 컬러 관련 정보 추출
    - MaxPool : 96 [3 x 3 Filters (Stride = 2)] → 96 x 27 x 27
    - Conv : 256 [5 x 5 Filters (Stride = 1, Activation = ReLU)] → 256 x 27 x 27
    - MaxPool : 256 [3 x 3 Filters (Stride = 2)] → 256 x 13 x 13
    - Conv : 384 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 384 x 13 x 13
    - Conv : 384 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 384 x 13 x 13
    - Conv : 256 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 256 x 13 x 13
    - MaxPool : 256 [3 x 3 Filters (Stride = 2)] → 256 x 6 x 6
    - Fully Connected : 4096 (Activation = ReLU)
    - Fully Conntected : 4096 (Activation = ReLU)
    - Fully Connected : 1000 (Activation = Softmax)

![image.png](attachment:f427cd35-bf79-49ca-a372-c65c7f1b39c7:image.png)

**6.1.3. VGGNet (2015)**

- 네트워크를 깊게 만드는 것 → 성능 영향?
- 네트워크 계층의 총 개수에 따른 유형 (VGG16, VGG19)
- VGG16 구조
    - Input : 224 x 224 (x 3)
    - Conv : 64 [**3 x 3 Filters** (Stride = 1, Activation = ReLU)] → 64 x 224 x 224 ⇒ 2번
    - MaxPool : 64 [**2 x 2 Filters** (Stride = 2)] → 64 x 112 x 112
    - Conv : 128 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 128 x 112 x 112 ⇒ 2번
    - MaxPool : 128 [2 x 2 Filters (Stride = 2)] → 128 x 56 x 56
    - Conv : 256 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 256 x 56 x 56 ⇒ 4번
    - MaxPool : 256[2 x 2 Filters (Stride = 2)] → 256 x 28 x 28
    - Conv : 512 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 512 x 28 x 28 ⇒ 4번
    - MaxPool : 512 [2 x 2 Filters (Stride = 2)] → 512 x 14 x 14
    - Conv : 512 [3 x 3 Filters (Stride = 1, Activation = ReLU)] → 512 x 14 x 14 ⇒ 4번
    - MaxPool : 512 [2 x 2 Filters (Stride = 2)] → 512 x 7 x 7
    - Fully Connected : 4096 (Activation = ReLU)
    - Fully Conntected : 4096 (Activation = ReLU)
    - Fully Connected : 1000 (Activation = Softmax)

![image.png](attachment:70fe011b-fe89-45c9-8813-be08318e4c85:image.png)

- 얕은 복사(shallow copy) vs. 깊은 복사 (deep copy)
    - 얕은 복사 `copy.copy()` → 같은 메모리 공간 공유
    - 깊은 복사 `copy.deepcopy()` → 별도의 메모리 공간

### **🤔 더 알아보기**

![image.png](attachment:45d1da0f-5b2d-4fd7-88b1-7133dcb54246:image.png)

[[CNN 개념정리] CNN의 발전, 모델 요약정리 1 (AlexNet ~ GoogLeNet)](https://warm-uk.tistory.com/44)

[[CNN 개념정리] CNN의 발전, 모델 요약정리 2 (ResNet, DenseNet)](https://warm-uk.tistory.com/46)

[[CNN] (2) CNN의 역사와 발전 과정, 주요 모델들](https://naver.me/FFGOpT8b)

![image.png](attachment:fd60704d-53b3-4afa-8fe1-0f3d3d16f801:image.png)

[경사 하강법 Gradient Descent 에 대한 수학적 이해와 활용](https://naver.me/Gal8opO2)