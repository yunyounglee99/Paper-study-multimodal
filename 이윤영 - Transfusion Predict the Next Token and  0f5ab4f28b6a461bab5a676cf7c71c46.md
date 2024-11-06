# Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

## Abstract

- 텍스트 : 다음 단어(토큰)을 예측 → discrete한 data
- 이미지 : diffusion을 통해 확률적 예측 → continuous한 data
- 기존의 방식 : 이미지를 양자화 → discrete image token으로 변환

<transfusion>

- discrete data + countinuous data를 **하나의 transformer 모델에서 훈련** 가능 → 기존의 방식보다 더 효율적임
- 각 이미지를 16개의 patch로 압축가능
- 7B parameters, 2T multi-modal tokens

## 1. Introduction

- discrete, continuous data를 단일 모델에서 훈련함으로써 기존의 방식보다 정보손실이 덜하고 매끄럽게 생성할 수 있음
- data의 50%는 text, 나머지 50%는 image로 하여 각각 text는 다음 텍스트 예측, image는 diffusion을 학습시킴.
- both modalities and loss functions at each training step
- 두 가지의 embedding layer를 사용
    - standard embedding layers : text tokens → vectors
    - patchification layers : 각각의 image를 sequence of patch vector로 표현

<img width="629" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-09-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 33 43" src="https://github.com/user-attachments/assets/5c425bc3-61ea-466f-b95f-54125073dcf5">


- text에는 casual attention(단일방향, 인과적 attention) / image patch에는 bidirectional attention(양방향 attention)를 사용 → (이미지 패치는 텍스트와 달리 이전 이후 구분이 사실은 없으니 sequence로 나눠도 양방향이 모두 중요하니 이러한 방법을 사용한 것 같음)
- inference에서는 language model(텍스트 생성)과 diffusion model(이미지 생성)를 결합한 디코딩 알고리즘 사용
- 실험
    - text-image 생성 : transfusion이 기존 chameleon 보다 1/3이하의 computing resource만으로도 더 좋은 성능을 보임
    - FID 점수가 chameleon보다 2배 더 낮음(낮을 수록 더 좋음 / 점수 낮을 수록 원본 이미지와 유사함)
- Ablation
    - image의 bidirectional attention이 중요함(일반 attention으로 변경하면 성능 저하됨)
    - U-Net 추가하면 더 큰 이미지를 압축하면서 성능 손실이 거의 없다는 것을 확인함

## 2. Background

### 2-1. Language Modeling

- 일반적인 text generation
- 다음 sequence 토큰 예측
- Language Modeling의 loss 함수
    
<img width="320" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-05_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 12 03" src="https://github.com/user-attachments/assets/30f86029-62a1-40cf-b55e-07263c27b8b3">

    

### 2-2. Diffusion

- 일반적인 diffusion
- cosine scheduler를 사용했다고 함
- foward process
    - 노이즈 추가 단계
        
![ec1e81a6-8f3d-4847-b4f5-056b1c14522e](https://github.com/user-attachments/assets/ef36328c-97c2-4c0c-9bd7-23f7aa8f4cfe)

-         
    - e는 gaussian noise
        
 <img width="320" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-05_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 16 05" src="https://github.com/user-attachments/assets/a78acf1d-b0db-421a-a1e0-fb5bedd33a42">
        
- reverse process
    - 노이즈 제거 단계
    - Diffusion의 손실 함수
        
<img width="320" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-05_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 19 11" src="https://github.com/user-attachments/assets/f3886e84-b1fc-496e-a008-dc520bac106d">
-        
    - 학습 대상인 gaussian noise와 (t번째 step에서의 이미지, t, 텍스트 정보)가 주어진 theta에서의 gaussian noise와의 차이의 기댓값
        
- inference
    - 순수한 gaussian noise에서 noise를 계속 걷어내는 방식

### 2-3. Latent Image Representation

<img width="329" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-05_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 40 17" src="https://github.com/user-attachments/assets/437d1d8e-9dc0-499e-84ca-dc60f3cc4b0a">

- VAE를 사용하여 이미지를 더 낮은 차원의 latent space에 인코딩함
- 특히 VQ-VAE를 사용해서 continuous latent embadding을 discrete latent embadding으로 변환

## 3. Transfusion

<img width="629" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-09-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 33 43" src="https://github.com/user-attachments/assets/92cfb926-91d3-4112-8ec1-db223325691f">


### Data Representation

- 텍스트 : fixed vocab에서 정수로 표현된 discrete token으로 변환
- 이미지 : VAE를 사용해 latent patch로 변환 → 각각의 patch들은 continuous vector
    - patch들은 left-to-right, top-to-bottom 순의 sequenced patch로 구성됨
    - BOI(Beginning of image), EOI(End of image) token으로 이미지 patch를 감쌈

### Model Architecture

- **single transformer**(를 계속 강조함 이미지에서도..ㅋㅋ)
- transformer의 Input으로는 R^d 차원의 sequence vector를 받고 output도 유사한 vector를 출력함
- 텍스트와 이미지는 서로 같은 space를 사용하기 위해 공유되지 않는 가벼운 파라미터들로 이루어진 구성요소를 사용함
    - 텍스트 : 입력 정수를 벡터로 변환하고, 출력벡터는 discrete한 분포를 만드는 embedding matrics 사용
    - 이미지 : k*k patch vector를 single transformer vector로 압축하는 local window를 두가지로 실험함(아마 U-Net과 linear를 얘기하는듯)

### Transfusion Attention

<img width="333" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-05_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 52 24" src="https://github.com/user-attachments/assets/4f1c9dd8-9520-4de6-9b81-5e8f236c76e7">


- LM : casual masking 사용 →효과적인 loss와 gradient 계산 / next token의 정보 유출 가능성 x
- image : 앞서 얘기한대로 sequence 방향이 중요하지 않음
- transfusion은 전체 sequence에 대해 casual masking 적용하고, image에 대해서만 bidirectional 다시 한 번 더 적용
- intra-image attention : model 성능을 올림(section 4.3)
- **training**
    
<img width="216" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 36 08" src="https://github.com/user-attachments/assets/7d9b187a-ccab-4ab7-bf06-7d9de6584d1b">

-    
    - L_LM : compute per token
    - L_DDPM : compute per image(image patch로 퍼져있는)
    - patchification 이전에 image level에서 diffusion loss 계산함
    - lamda로 두 loss값 balancing
- **inference**
    - BOI token이 들어오면 LM mode에서 diffusion mode로 전환
    - 입력 sequence에 n개 순수한 노이즈 x_T patch 만들고 T단계 걸쳐 노이즈 제거
    - 이전단계에 덮어서 노이즈 제거를 하는 방식으로 이전 단계는 attention 적용할 수 없도록
    - EOI token이 들어오면 diffusion mode에서 LM mode로 전환

## 4. Experiments

### Setup

- **Evaluation**
    
<img width="377" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 03 15" src="https://github.com/user-attachments/assets/f4d24f79-0cbc-4838-87c8-d2cc74a327d1">

-    
    - text-text : Wikipidia, C4 corpus에서의 perplexity / Llama2 pretraining evaluation suite로 accuracy
    - text-image : MS-COCO benchmark 사용, zero-shot FID 사용해서 이미지 현실성 측정, CLIP score로 이미지/프롬프트 정렬성 측정
- **Baseline**
    - 기존의 Chameleon 방식을 따라 비교 가능
- **Data**
    - 3억 8천만개의 라이센스가 부여된 Shutterstock 이미지 캡션 사용
    - 이미지 캡션 순서는 무작위로 정렬 / 80퍼는 캡션을 먼저 배치
- **Latent Image Representation**
    - 8600만개의 parameter를 가진 VAE를 사용
    - CNN encoder, decoder 사용하며 차원은 8차원
    - 256*256 pixel image를 32*32*8 tensor로 축소
    - 각 latent 8-dimensional latent pixel은 8*8 pixel patch에 해당함
    - 100만 단계로 훈련
- **Model Configuration**
    
<img width="292" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 02 57" src="https://github.com/user-attachments/assets/7a9ecd0c-de6c-4389-a5e9-ba85f2c1af1a">

-    
    - Llama 표준 설정에 맞게 0.16B, 0.37B, 0.76B, 1.4B, 7B parameter사용
    - linear patch는 전체 parameter에서 적은 부분을 차지함(약 0.5%) / U-Net patch는 소형 모델에서는 큰 부분을 차지하지만 대규모 모델에서는 미미한 편
- **Optimization**
    - AdamW(beta_1 = 0.9, beta_2 = 0.95, epsilon=1e-8)
    - learning rate = 3e-4
    - warm up = 4000step
    - decaying = 1.5e-5 (cosine scheduler)
    - 4096 tokens in batches of 2M tokens for 250k steps
- **Inference**
    - text mode : greedy decoding
    - image mode : 250 diffusion steps(1000timesteps)

### Controlled Coparison with Chameleon

<img width="668" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 00 27" src="https://github.com/user-attachments/assets/11859453-265b-4c82-8a24-c0d7fbbc3d9e">


<img width="671" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 04 32" src="https://github.com/user-attachments/assets/1b4ad11b-4828-4521-8bdd-6d2301abbffa">


<img width="672" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 04 54" src="https://github.com/user-attachments/assets/5a438ed2-0291-44de-875e-53b252d32aa3">


- 전반적으로 기존 Chameleon 방식보다 좋은 성능을 내보임
- 이미지 생성, 계산 효율성, 텍스트 처리에서도 좋은 성능을 보임
- 양자화된 이미지 토큰을 사용하면 텍스트 성능이 떨어짐..!
    - 가설1 : latent space에서 텍스트와 이미지가 서로 경쟁
    - 가설2 : diffusion 덕분에 더 적은 parameter로 이미지 생성 → text 모델링에 유리

### Ablation

- **attention masking**
    
<img width="668" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 13 18" src="https://github.com/user-attachments/assets/bd7a224d-bee5-4042-ad2b-6ccc1aa3eb71">
 -   
    - U-Net의 경우 transformer와 독립적으로 이미지 내에 biderctional attention을 포함하고 있어서 두드러진 성능 개선이 보이지는 않음
- **Patch size**
    
<img width="667" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 23 56" src="https://github.com/user-attachments/assets/753c4f5b-4964-4978-ae9e-1c29b7e8fb4c">
    
    - linear : 더 큰 사이즈의 patch는 훈련 때 더 많은 이미지를 넣을 수 있어 inference때 계산량을 줄일 수 있지만, 성능 손실이 있을 수 있음
    - U-Net : 오히려 더 큰 patch가 성능이 더 좋음
    - patch 커질 수록 text의 성능이 감소
- **Patch Encoding/Decoding Architecture**
    
<img width="672" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 28 56" src="https://github.com/user-attachments/assets/71287a20-df59-429c-8466-8c2a6e450eb4">
-    
    - U-Net은 inductive bias에서 유리함 → U-Net은 그대로 둔 채 transformer 크기만 바꿔보자
    - transformer가 커질수록 U-Net의 상대적 이점이 줄어들지만 아주 사라지지는 않음
- **Image Noising**
    
<img width="663" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 31 07" src="https://github.com/user-attachments/assets/432e3a85-43ec-405e-9842-b591bf9ff6f9">
 -   
    - 노이즈 제한이 이미지 캡셔닝 성능은 올리지만 다른 성능에서는 효과 없음

### Comparison with Image Generation Literature

<img width="670" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 35 02" src="https://github.com/user-attachments/assets/c8275e18-ab93-4742-9205-3a26a8608272">

- 다른 이미지 생성모델과 비교해도 성능이 좋음
- 다만 SD3는 natural한 데이터가 아닌 backtranslation으로 캡션을 생성한(아마도 다른 인공지능을 사용한 캡션 생성을 의미하는듯?) 데이터로 학습되었기 때문에 natural한 데이터로 학습한 transfusion보다 성능이 높다고 변명함

### Image Editing

<img width="662" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 41 00" src="https://github.com/user-attachments/assets/66b1d329-84b6-44f7-9846-76e3aa4948fd">

<img width="660" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 41 16" src="https://github.com/user-attachments/assets/488517f2-1fed-45e7-8364-6f1892e68c0c">

이미지 수정(image-image)성능도 fine tuning하면 잘하는 것을 보임

<img width="670" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 42 43" src="https://github.com/user-attachments/assets/1730c831-e842-4543-b235-187662c0c920">

<img width="664" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 42 58" src="https://github.com/user-attachments/assets/153fefec-0c1c-4b7c-adeb-4606a4542705">

<img width="490" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 43 31" src="https://github.com/user-attachments/assets/5382afb0-d765-47d6-aab5-666c07d6b2a9">

<img width="494" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-06_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 43 47" src="https://github.com/user-attachments/assets/01abdad3-c447-442a-bd7d-e8e544065807">
