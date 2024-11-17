# 이윤영 - High-Resolution Image Synthesis with Latent Diffusion Models

[https://velog.io/@lighthouse97/UNet의-이해](https://velog.io/@lighthouse97/UNet%EC%9D%98-%EC%9D%B4%ED%95%B4)

[https://velog.io/@jochedda/딥러닝-Autoencoder-개념-및-종류](https://velog.io/@jochedda/%EB%94%A5%EB%9F%AC%EB%8B%9D-Autoencoder-%EA%B0%9C%EB%85%90-%EB%B0%8F-%EC%A2%85%EB%A5%98)

# 1. Introduction

- 이미지 합성 - 최근 CV 분야에서 눈부신 발전을 이룬 분야 중 하나 but, 가장 높은 계산을 요구함(because of 고해상도 이미지, 수십억개의 parameter 포함)
- GAN : 비교적 제한된 데이터에 대해서는 우수한 성능을 보였으나, 다양한 데이터 분포를 다루는 데 있어 한계가 있음
- Diffusion(DMs) : auto encoder로 구축되어 이미지 합성에서 우수한 성능을 보이고 있음 / GAN에 비해 모드 붕괴 현상이 없고, 수십억개의 parameter 없이도 natural한 이미지의 복잡한 분포를 모델링할 수 있음

### Democratizing Hing-Resolution Image Synthesis

- DMs는 데이터의 미세한 디테일까지 모델링해야하므로 과도한 계산 자원을 소모할 수 있음
    
    → 이를 해결하기 위해 reweighted variational objective을 사용하여 초기 잡음 제거 단계를 줄이고자 함
    
    but, 그럼에도 많은 연산이 필요함
    
- 이는 두가지 문제를 초래하는데…
    1. 모델 training을 위해 막대한 computing resource 필요 → 연구 커뮤니티의 소수만 접근 가능 → 큰 탄소 발자국 남김
    2. 이미 훈련된 모델을 평가하는  것 역시 시간, 메모리 면에서 비용이 많이 듦 → 계산 복잡도를 줄이는 방법이 필요

### Departure to Latent Space

<img width="446" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 13 03" src="https://github.com/user-attachments/assets/d5d7f824-36da-4f56-a61d-c9c5f9b2718e">


- 픽셀 공간에서 pre-trained 된 diffusion 모델의 분석으로 시작함
- likelihood 기반의 모델학습은 두가지로 나뉨
    1. Perceptual compression : high frequency한 세부 정보는 제거하지만 약간의 의미적 변형을 배우게 되는 단계
    2. Sementic compression : 데이터의 의미론적, 개념적 구성을 학습하는 단계
- auto encoder를 먼저 학습
    - 원래의 data space와 perceptual하게 동일하면서도 더 낮은 차원의 space를 제공
    - 이때, 과도한 space 압축 필요없음
- 학습된 공간에서 diffusion 학습
    - 계산 복잡도를 줄이면서 한번의 네트워크 패스를 통해 latent space에서 잠재 공간의 효율적 이미지 생성이 가능
- 이렇게 생성된 모델을 **Latent Diffusion Models, LDMs**라 칭함
- 이 방식의 주요 장점은 일반적인 auto encoding 단계를 한 번만 학습하면 되기때문에 여러 diffusion 모델 학습이나 다양한 이미지 생성 작업 탐색에 재사용할 수 있음
    - 이를 위해 트랜스포머를 U-Net 백본과 연결하여 다양한 토큰 기반 조건 메커니즘 구현

### Contribution

1. 기존 트랜스포머 기반 방식과 달리, 고차원 데이터를 더 효율적으로 확장할 수 있어 high-resolusion image synethesis에 효율적 적용이 가능
2. 여러 task와 dataset에서 경쟁력 있는 성능을 발휘하며 계산비용을 대폭 절감함
3. encoder/decoder architecture, score-based prior를 동시에 학습할 필요 x
    
    → 잠재공간을 최소한으로 정규화
    
4. super resolution, inpainting, sementic synthesis와 같은 작업에서 이미지 해상도와 상관없이 일관되게 생성 가능
5. cross-attention 기반의 general purpose conditioning mechanism 설계하여 multi-modal training이 가능하며, class conditional, text-image, layout-image 모델을 훈련함

# 2. Related Works

# 3. Method

<img width="445" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-15_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 17 22" src="https://github.com/user-attachments/assets/bb57ca59-7fbf-4343-b5bf-e262d3dbb907">

- 고해상도 이미지 합성을 위한 diffusion model의 training은 픽셀 공간에서 작동하므로 perceptual하게 중요하지 않은 디테일들을 무시하도록 손실항목을 subsampling하여 계산 요구량을 줄일 수 있음
    
    → but, 그럼에도 픽셀 공간에서의 연산은 많은 resource가 필요함
    
- 이를 위해, 생성 학습단계와 압축 학습단계를 명시적으로 분리하는 방법을 제안함
- 이를 위해 auto encoding model을 사용하여 이미지 공간과 지각적으로 동등하지만 계산 복잡도를 크게 줄이 공간을 학습함.
- 이러한 접근법은 다음과 같은 장점들을 제공함
    1. 고차원 픽셀 공간을 벗어나기 때문에 계산 효율성의 더 높아짐 → 저차원 공간에서 sampling이 이뤄지기 때문
    2. U-Net 아키텍처에서 파생된 diffusion model의 inductive bias를 활용함. 이는 공간구조를 가진 데이터에 특히 효과적임
    3. general-purpose compression model을 제공함
        
        → latent sapce는 여러 생성 모델을 학습하거나, 단일이미지 CLIP 기반 합성과 같은 downstreme task에도 활용할 수 있음
        

## 3.1. Perceptual Image Compression

- perceptual loss + patch-based adversarial objective
    - 픽셀기반 loss에만 의존할때 발생하는 블러현상을 방지함

<구체적 방법론>

1. RGB 공간의 이미지 $x \in \mathbb{R}^{H\times W\times3}$가 주어졌을 때, 인코더 $\mathcal{E}$는 $x$를 latent representation $z=\mathcal{E}(x)$으로 인코딩함.
2. 디코더 $\mathcal{D}$는 $z$로부터 이미지를 재구성하여, $\tilde{x} = \mathcal{D}(z)= \mathcal{D}(\mathcal{E}(x))$를 생성함. 이때, $z \in \mathbb{R}^{h \times w \times c}$이고, 인코더는 이미지를 downsampling(factor $f = H/h = W/w$, $f = 2^m, \ m\in \mathbb{N}$)함.
- 그래서 고차원 variance latent space를 피하기 위해 두가지 normalize 방식을 실험함(너무 넓게 퍼져있으면 성능이 떨어질 수도 있으므로)
    1. **KL-reg** : VAE와 유사하게 학습된 latent space를 표준 정규 분포로 유도하는 약간의 KL-패널티 적용
    2. **VQ-reg** : vector quantization layer 추가

→ latent space $z$의 2차원 구조 유지하며 상대적으로 낮은 압축률로도 높은 품질의 재구성이 가능함

## 3.2. Latent Diffusion Models, LDMs)

### **Diffusion model**

<img width="350" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 56 12" src="https://github.com/user-attachments/assets/8333002c-7be6-4e01-98fa-bea4ee00f660">


- $t$는 $\{ 1, ... \ ,T\}$에서 균일하게 샘플링

### Generative modeling of Latent Representation

- 훈련된 Perceptual Compression Model (encoder, decoder)을 사용해서 효율적이고 저차원의 latent space에 접근할 수 있게 됨 → high frequency의 인지할 수 없는 세부 정보를 추상화함.
- 저차원의 공간이 고차원의 픽셀 공간에 비해 likelihood-based 생성모델에 더 적합함 그 이유는…
    - 데이터를 구성하는 중요한 의미적 정보에 집중할 수 있음
    - 더 낮은 차원에서 훈련할 수 있어 계산적으로 훨씬 효율적인 공간을 제공
- autoregressive, attention-based transformer models등을 사용한 이전 연구들은 과도하게 압축된 discrete한 latent space를 사용했음
- 이에 비해 LDMs는 image-specific inductive biases를 활용할 수 있음 그 장점으로는…
    - 모델의 기본구조를 2D convolution layer를 주로 구축할 수 있음
    - reweighted bound를 사용해 perceptual하게 가장 중요한 정보에 목적을 집중할 수 있음

reweighted bound는 아래와 같음

<img width="333" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-16_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7 24 01" src="https://github.com/user-attachments/assets/7fe0b1a8-f559-44bf-96aa-58973b6cabe5">


- 기본적인 $L_{DM}$과 다른점은 픽셀 공간에서의 $x$가 아닌, 인코더를 거친 $\mathcal{E}(x)$에서의 기댓값으로 정의된 것을 확인할 수 있고, 그래서 노이즈 $\epsilon_\theta$에 들어가는 값도 $x_t$가 아닌 latent space의 $z_t$를 사용하는 것을 확인할 수 있음
- model의 neural backbone은 time conditional U-Net으로 구현됨
- forward process가 고정되어있기 때문에 training 중에는 latent representation $z_t$는 $\mathcal{E}$로부터 효율적으로 얻을 수 있음
- $p(z)$로부터 샘플링된 결과는 $\mathcal{D}$를 단 한 번 통과하여 이미지 공간으로 디코딩 할 수 있음

## 3.3. Conditioning Mechanisms

- 다른 유형의 generative model과 마찬가지로 diffusion은 $p(z|y)$형식의 조건부 분포를 모델링할 수 있음
- 조건부 denoising encoder $\epsilon_\theta(z_t, t, y)$로 구현
- 입력 $y$를 통해 텍스트, semantic map, image-to-image 변환과 같은 합성 프로세스 가능
- U-Net backbone에 cross-attention 매커니즘을 추가함
    - cross-attention은 다양한 input modality들로부터 attention을 학습할 수 있음
- 다양한 input modality를 전처리하기 위해 domain-specific encoder)$\tau_\theta$를 도입함
- 이 encoder는 conditional input $y$를 중간 representation $\tau_\theta(y)\in \mathbb{R}^{M\times d_\tau}$로 변환함
- 이 중간 representation은 U-Net의 intermidiate layer로 매핑함

<cross-attention>

<img width="315" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2 31 18" src="https://github.com/user-attachments/assets/ba3bd904-6afe-4e88-bccd-1bd5bdc44601">


<img width="424" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2 31 48" src="https://github.com/user-attachments/assets/ad37f33e-d512-46ba-a14c-6ceb143d3153">


- $Q$ : U-Net에서 추출된 $z_t$의 중간 representation
- $K$ : conditional input $y$의 encoder 출력
- $V$ : 동일하게 conditional input $y$의 encoder 출력
- $\varphi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon}$ : U-Net의 $z_t$에서 추출된 중간 representation
- $W_{Q}^{(i)} \in \mathbb{R}^{d \times d_\tau}$, $W_{K}^{(i)} \in \mathbb{R}^{d \times d_\tau}$, $W_{V}^{(i)} \in \mathbb{R}^{d \times d_\tau}$ (learnable projection matrics)

<img width="446" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 24 25" src="https://github.com/user-attachments/assets/34940ad8-b5ea-4441-a187-b081eb96df29">


<conditional LDM>

- 다음 loss를 학습함

<img width="392" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 25 57" src="https://github.com/user-attachments/assets/49ec9f61-766e-4eb5-97e5-80e83f1090e8">


- $\tau_\theta$와 $\epsilon_\theta$는 동시에 최적화됨

# 4. Experiments

<Pixel 기반 diffusion vs LDMs>

<img width="950" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 51 22" src="https://github.com/user-attachments/assets/92e3f622-2523-4fb7-8cb9-eebe84ac5f59">


- VQ-regularized latent space에서 학습된 LDMs은 continuous latent space보다 샘플 품질에서 더 나은 성능을 보임

## 4.1. Perceptual Compression Tradeoff

<img width="859" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 39 32" src="https://github.com/user-attachments/assets/1742f81d-18ca-4f1f-96f5-db939ec49e89">


<img width="442" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 55 20" src="https://github.com/user-attachments/assets/d54d5f31-dafd-4082-87c6-dc68dc1ca5d0">


- 이 실험에서는 LDMs가 downsampling rate $f=\{1, 2,4,8,16,32\}$에서 어떻게 작동하는지 분석함 (LDM-1은 픽셀기반 모델을 의미 / A100 단일 GPU로 실험)
- Table 8. 은 LDM에 사용된 first stage model의 hyper parameter 및 재구성 성능을 보여줌
- Figure 6. 는 ImageNet으로 학습한 class conditional model의 훈련 step에 따른 성능을 비교

<결과>

1. downsampling rate이 작을 수록 (LDM-1, LDM-2)
    - 학습 진행속도가 느림 → perceptual compression이 diffusion model에 남기 때문
2. downsampling rate이 너무 클때(LDM-32)
    - 학습 초기에 샘플 품질이 정체되는 경향이 있음
3. 균형이 잡힌 비율 (LDM-4, LDM-8)
    - 효율성과 perceptual faithful한 결과가 잘 나옴
    - 픽셀 기반 모델(LDM-1)과 비교했을때 확실히 성능이 잘나온 것을 확인할 수 있음

<img width="436" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 02 41" src="https://github.com/user-attachments/assets/3764eec6-3073-4558-a4fc-8158e5a42fd5">


- 다양한 노이즈 제거 step 수에 따른 샘플릭속도를 FID와 비교

<결과>

1. LDM-4, LDM-8은 perceptual compression, conceptual compression 간의 비율이 적절하지 않은 모델을 능가함
2. 특히 LDM-1과 비교했을때 FID 점수는 훨씬 낮으며(좋음) 샘플처리량은 높음
3. 복잡한 데이터셋(ImageNet) 같은 경우 압축 비율을 줄여 성능 저하를 방지함
4. 결론적으로 LDM-{4,8}이 성능이 제일 좋음

## 4.2. Image Generation With Latent Diffusion

<unconditional model>

<img width="451" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 41 04" src="https://github.com/user-attachments/assets/6868c95d-ea5c-4b92-a748-064d34a80fa7">


<img width="899" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 52 46" src="https://github.com/user-attachments/assets/3da19170-ec4b-497c-b74c-d09f217602ea">


- 256*256 해상도의 이미지로 실험
- 샘플 품질, 데이터 매니폴드 커버리지(precision and recall) 두가지 지표를 사용해 평가함
- 여타 다른 모델에 비해 우수한 성능을 보임
- resource나 parameter가 다른 모델들에 비해 현저히 적어도 우수한 성능을 보임

## 4.3. Conditional Latent Diffusion

### 4.3.1. Transformer Encoders for LDMs

- 처음으로 diffusion model을 조건부 modality에 적용함

<text-to-image>

<img width="913" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 27 00" src="https://github.com/user-attachments/assets/0582ab02-fad2-4c1f-ac1e-10483d411586">


- LAION-400M 에서 언어 프롬프트를 조건으로 사용하는 1.45B parameter의 KL-정규화된 LDM을 학습함
- BERT tokenizer 사용, $\tau_\theta$를 트랜스포머로 구현 → latent code로 infer → multi-head cross attention으로 U-Net으로 매핑함

<img width="442" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 27 50" src="https://github.com/user-attachments/assets/10750161-0197-4d5c-af27-9e420a0c04f6">


- parameter수를 크게 줄이면서도 다른 모델 대비 성능이 뛰어난 것을 확인할 수 있음

<sementic layouts>

<img width="434" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 30 36" src="https://github.com/user-attachments/assets/e36022ac-755e-401a-89b7-a413048d3370">


<img width="853" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 32 15" src="https://github.com/user-attachments/assets/79445c57-0a89-4686-8289-bd7d530d9417">


- sementic layout 실험에서도 좋은 성능을 확인할 수 있음

### 4.3.2. Convolutional Sampling Beyond 256^2

<img width="515" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 31 35" src="https://github.com/user-attachments/assets/e98ebccd-2983-40b9-b2db-501db74df307">


<img width="458" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 40 20" src="https://github.com/user-attachments/assets/1c4981bf-7e57-4fc4-a6fe-b015465c0f8b">


- 256^2의 해상도로 학습했지만 더 큰 해상도에서도 일반화가 가능한 것을 보임(512^2, 1024^2)
- signal-to-noise ratio 가 결과에 중요한 영향을 끼침

<img width="560" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 41 05" src="https://github.com/user-attachments/assets/92e9355c-ef34-4362-aaa0-ba36c3663252">


- text-to-image에도 적용이 가능

## 4.4. Super-Resolution with Latent Diffusion)

- 해상도가 낮은 이미지를 condition으로 연결해서 super resolution에 사용할 수도 있음
- 저해상도 condition $y$를 U-Net과 연결, $\tau_\theta$는 identity로 설정

<img width="424" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 51 50" src="https://github.com/user-attachments/assets/916dd0ed-d3e9-49ff-97d0-3544d0b9bd9d">


<img width="861" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 33 10" src="https://github.com/user-attachments/assets/8105c99f-7f77-4255-9923-a323b9a991be">


<img width="437" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 52 09" src="https://github.com/user-attachments/assets/b5b773be-cc4d-4352-8b08-7fdc325cfc1b">


- 완전 SR3보다 좋다할 수는 없지만 그래도 동등한 성능을 보여주고 있음

## 4.5. Inpainting with Latent Diffusion

<img width="434" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 24 43" src="https://github.com/user-attachments/assets/e87bb806-d828-4ab1-8028-f529383a1c5a">


<img width="434" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 12 51" src="https://github.com/user-attachments/assets/e3cee6a2-567e-45e6-8bbf-883baa1e7838">


<img width="430" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 25 05" src="https://github.com/user-attachments/assets/67a028fb-9b43-4f2a-be94-a045658ad2f9">


- inpainting : 이미지의 마스킹된 영역을 새로운 컨텐츠로 채우는 작업
- KL 및 VQ reg 적용 LDM-1 vs LDM-4 vs first step에 attention 없는 LDM-4(디코딩 시 GPU 메모리 줄임)
- 픽셀 기반 모델보다 FID 및 속도향상을 확인할 수 있었음
- 고해상도 모델로도 실험해봤을때 품질 저하가 나타남 → attention layer가 문제일 것이라고 추측
- half epoch 학습으로 finetuning해서 성능 회복
