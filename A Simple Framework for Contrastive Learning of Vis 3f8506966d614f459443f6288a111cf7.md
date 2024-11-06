# A Simple Framework for Contrastive Learning of Visual Representations

# Introduction

- effective visual representation
    - generative
        - input space에서 pixel을 생성하거나 modeling
            
            → pixel단위의 생성은 계산비용이 많이 듦
            
            → 표현학습에 반드시 필요한 것은 아님
            
    - discriminative
        - 지도학습과 유사한 목적 함수를 사용
        - unlabeld dataset에서 input, label을 추출하는 pretext task를 수행해야함
            
            → 많은 heuristic에 의존
            
            → trained representation에 일반성을 제한할 수 있음
            
            → latent space에서 contrastive learning 기반으로 하는 discriminative 접근법이 SOTA달성
            
- SimCLR : visual representation을 위한 간단한 contrastive learning 프레임워크
    - 이전의 연구들을 능가함
    - 특수한 아키텍처나 메모리 뱅크가 필요하지 않음 → 더 간단함

- 효과적인 contrastive learning을 위한 주요 요소
    - 데이터 증강 → 비지도 대조 학습이 더 큰 이점을 얻음
    - representation과 contrastive loss 사이에 학습가능한 non-linear 변환을 도입
    - contrastive cross entropy loss은 정규화된 임베딩과 적절한 온도 매개변수로 이점을 얻을 수 있음
    - contrastive learning은 supervise learning보다 더 큰 배치 크기와 더 긴 훈련 시간에서 강점을 가짐, 더 깊은 네트워크에서 좋은 성능을 보임

# 2. Method

## 2-1. The Contrastive Learning Framework

<img width="400" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 06 09" src="https://github.com/user-attachments/assets/53fd09d9-b05f-4dbf-af10-fc83cff78162">


- SimCLR는 representation을 같은 data example을 서로 다른 augment를 통해 view(?)간의 일치를 최대화하는 contrastive loss를 학습함
    1. data example을 무작위로 변환하여 동일한 example을 두 개의 상관된 view인 x_i, x_j를 생성함 이 쌍을 positive pair라고 함.
        - 세 개의 augment를 순차적으로 적용
            
            (1). random crop 후 원래 크기로 resize
            
            (2). random color distortion
            
            (3). random Gaussian blur
            
    2. augmented data example을 representation vector로 변환 ResNet 사용
        
<img width="448" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 22 51" src="https://github.com/user-attachments/assets/f53d1a43-8875-4993-8ef8-bc72cf7d2108">
        
-
    - h_i : avg pooling layer 후 출력
        
    3. representation을 contrastive loss가 적용되는 공간으로 mapping함.
        - one hidden layer MLP 사용
        - sigma : ReLU(non-linearity)
            
<img width="324" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 29 09" src="https://github.com/user-attachments/assets/d4feec14-7110-46cf-9b42-6f76bd7aacb1">

-

    4. contrastive loss function : postive pair를 포함한 data set {x_i}가 주어졌을 때, contrastive prediction task는 주어진 x_i에 대해 {x_k}_k≠i 중에서 x_j를 식별하는 것을 목표로함
- 임의로 N개의 example을 mini batch로 sampling하고, mini batch에서 파생된 augmented example 쌍에 대해 task를 정의 → 2N개의 data point 생성
- negative pair를 따로 sampling하기보다 positive pair가 주어졌을때, 다른 2(N-1)개의 augmented example을 negative example로 간주
- 
    
<img width="295" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 42 56" src="https://github.com/user-attachments/assets/c0f24559-246c-40d4-aa2f-00c81afe76b2">
    
<img width="490" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 45 01" src="https://github.com/user-attachments/assets/1d530bc9-57c0-43d3-9f14-ae7e63104e22">
    
<img width="232" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 45 20" src="https://github.com/user-attachments/assets/fdef7ca6-a002-4bd6-a4a9-1e0638241ae0">
    
- 이 loss는 mini batch 내의 모든 positive pair((i,j), (j,i))에 대해 계산됨 → NT-Xent라고 부르자

<img width="623" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 48 29" src="https://github.com/user-attachments/assets/b2ed9718-a1dd-42c2-9fca-80460254cdae">
## 2-2. Training with Large Batch Size

- 학습에 메모리 뱅크를 사용하지 않음 대신 훈련 batch 크기 N을 다양하게(256-8192) 조정함
    - 대규모 batch는 linear lr scaling을 사용하면 training이 불안정해질 수 있음
    - 그래서 모든 batch 크기에 대해 LARS 옵티마이저 사용
- global batch normalization

## 2-3. Evaluation Protocol

### Dataset and Metrics

- unlabeled 상태에서 encoder network f를 학습하는 비지도 사전 학습 대부분은 ImageNet 데이터셋 + CIFAR-10 → transfer learning
- representation을 평가하기 위해 linear evaluation protocol 사용
- frozen base network위에 linear classifier를 학습시키고, 테스트 정확도를 representation의 품질을 측정하는 지표로 사용

### Default setting

- base encoder network : ResNet-50
- 2 layer MLP(왜?) : 128차원 잠재공간으로 representation을 투영
- loss : NT-Xent 사용
- lr = 4.8 (0.3 * BatchSize/256)
- weight decay = 10^-6
- batch size = 4096, 100epoch
- cosine decay schedule

# 3. Data Augmentation for Contrastive Representation Learning

### Data augmentation defines predictive tasks

- data augmentation은 지도/비지도 학습에 널리 사용되었지만, contrastive predict task에 사용된 적은 많이 없음(대부분 network 아키텍처를 변경하는 방법을 사용)
- 
    
<img width="326" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-20_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 09 43" src="https://github.com/user-attachments/assets/1289a943-5d0b-438b-ac52-3d749d145dea">

-    
    - (a).
        - A : local view, B : Global view
        - B→A 간의 일치를 학습
        - 같은 이미지 내에서의 다양한 부분들이 서로 어떻게 연결되는지 이해할 수 있음
    - (b).
        - 서로 인접하지만 겹치지 않은 두부분을 자름
        - D→C 같이 인접 뷰들 간의 일치를 학습 : 공간적 연속성을 이해할 수 있음
- (b)와 같은 설계는 predict task를 다른 구성 요소들로부터 독립적으로 만들 수 있음
    - data augmentation 범위를 확장하고, 더 넓은 범위의 task를 정의할 수 있음

## 3-1. Composition of data augmentation operations is crucial for learning good representation

<img width="523" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 48 33" src="https://github.com/user-attachments/assets/56101c90-51d7-4865-85cc-d13229580e92">

- 각종 data augmentation의 방법들
- 데이터 증강의 영향을 연구함(crop, resize, flip, cutout, color dist, blur등)
- resize가 처음에 필수로 필요하기에 resize가 아닌 다른 증강기법 효과를 연구하기가 힘듦
    - 원본과 resize data를 구분함
    - 비대칭 data augment가 성능에 부정적인 영향을 미치나, 개별 augment의 영향을 실질적으로 바꾸지는 않음

<img width="523" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_10 48 14" src="https://github.com/user-attachments/assets/706f0299-2b84-4b5a-ad50-7f207043c90b">

- 대각선 셀 : 각 augment가 단독으로 적용되었을 때임 → 성능이 아주 좋지는 않음
- (color, crop) 조합이 가장 좋은 성능을 보임
- random crop만 쓰는 것은 대부분의 이미지 패치가 유사한 색상 분포를 가지는 문제를 가지고 있기 때문에 별로 성능이 좋지 않음
    
<img width="475" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 02 29" src="https://github.com/user-attachments/assets/a00c80f1-aef9-47c6-936f-dfee2a7ce758">
    
- color dist가 data의 color 분포를 크게 바꿈 → predict 어렵게 함 → 일반화된 feature 학습

## 3-2. Contrastive learning needs stronger data augmentation than supervised learning

<img width="518" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 12 51" src="https://github.com/user-attachments/assets/23136e60-039d-449f-be92-ab5e33b5f1d6">

- SimCLR은 color dist가 강해지면 더욱 성능이 올라가고, supervised는 점점 떨어짐
- 성능은 supervised가 조금 더 좋긴함

# 4. Architectures for Encoder and Head

## 4-1. Unsupervised contrastive learning benefits(more) from bigger  models

<img width="495" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 23 38" src="https://github.com/user-attachments/assets/ca73bc2a-32e2-4287-beed-53ea96d4a68f">

- 파란색 : 100epoch unsupervised model / 빨간색 : 1000epoch unsupervised model / 녹색 : 90epoch supervised model
- model의 크기가 커질수록 supervised와의 격차가 줄어듦(그러나 supervised가 아직 좋긴함)

## 4-2. A nonlinear projection head improves the representation quality of the layer before it

<img width="393" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 43 49" src="https://github.com/user-attachments/assets/425464d4-58fb-46e1-a782-5fc715c350a5">

- non-linear progection head( g(h) )의 중요성을 연구함
- non-linear head가 linear head보다 성능이 더 좋은 것을 확인할 수 있음
- non-linear projection 이전의 표현을 사용하는 것이 중요한 이유는 contrastive loss가 유발하는 정보 손실 때문이라고 추측함(..?)
- z=g(h)는 data augmentation에 대해 불변하도록 학습되기 때문에 g는 색상이나 객체의 방향과 같은 downstream task에 유용할 수 있는 정보를 제거할 수 있음
    
<img width="478" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 55 06" src="https://github.com/user-attachments/assets/c0607db8-bff5-42f3-8729-e3d898ff00fc">
    
- non-linear projection 이후 많은 data의 representation을 손실한 것을 확인할 수 있음
    
    → 모델이 변환에 불변한 표현을 학습하는 것을 유도함!
    

# 5. Loss Functions and Batch Size

## 5-1. Normalized cross entropy loss with adjustable temperature works better than alternative

<img width="530" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 02 21" src="https://github.com/user-attachments/assets/3458b6ce-29a2-4adc-a5d3-45c5539b5ef0">

1. NT-Xent loss : temp와 cosine sim을 사용해 negative example의 난이도에 따라 weight를 다르게 부여함. → 학습이 보다 어려운 negative example을 통해 더 잘 이루어질 수 있도록함
2. NT-Logistic loss : negative loss의 난이도를 고려하지 않고, logistic func의 출력에 따라 gradient를 계산함
3. Margin Triplet loss : negative example과 positive example 간의 차이가 일정 마진 m을 초과하지 않을때만 gradient를 계산함 → negative example이 어느 정도 가까워졌을때만 학습에 반영되도록 함
- 2,3은 negative example의 난이도를 고려하지 않기 때문에 semi-hard negative example mining과 같은 추가적인 기법이 필요함

<img width="447" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 21 40" src="https://github.com/user-attachments/assets/a0f9b0f4-e03f-4a31-9f41-a916933495f5">

- NT-Xent을 사용한 model의 성능이 가장 좋은 것을 확인할 수 있음

<img width="429" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 22 49" src="https://github.com/user-attachments/assets/c279d2fc-e128-4dbc-9ac7-5b702516519a">

- l2 norm을 사용하고, tau=0.1일 때 성능이 가장 높은 것을 확인할 수 있음

## 5-2. Contrastive learning benefits(more) from larger batch sizes and longer training

<img width="461" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 29 20" src="https://github.com/user-attachments/assets/da96b467-11ff-4ae6-8960-3e3e77daec19">

- epoch가 클수록, batch size가 클수록 성능이 더 좋아지는 것을 확인할 수 있음
- 다만 batch size가 너무 크다고 좋은것은 아님(8192)
- 그 이유는 더 큰 batch가 더 많은 negative example을 제공하여 수렴을 촉진하기 때문

# 6. Comparison with State-of-the-art

### Linear evaluation

<img width="403" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 40 15" src="https://github.com/user-attachments/assets/d176de5e-5d67-4a3c-aae5-4b261d0cfce8">

<img width="608" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-08-21_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 40 37" src="https://github.com/user-attachments/assets/1b8e23bf-15c7-46dc-af8b-95df5ad03fb9">

각각의 실험들(linear evaluation, semi-supervised learning, transfer learning)에서 다른 self-supervised learning model들에 비해 SimCLR이 성능이 좋은 것을 확인할 수 있음

# 8. Conclusion

- data augmentation, non-linear porjection head, loss function의 차이로 contrastive representation의 성능을 올릴 수 있다.
