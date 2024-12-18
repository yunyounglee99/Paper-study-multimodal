# 이윤영 - OpenVLA : An Open-Source Vision-Language-Action Model

<img width="734" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 52 30" src="https://github.com/user-attachments/assets/e7b38aa3-1c4b-48cd-970f-0cf80e9cc72e">


# 1. Introduction

- 기존의 로봇 조작 학습 정책의 주요 문제점 중 하나는 generalization이 어렵다는 것임.
- 장면 방해 요소나 새로운 객체에 대해 견고하지 못하며, 보지 못한 작업 지시를 수행하기 어려워함
- 반면에 CLIP, SigLIP, Llama2 와 같이 Vision-Language model은 인터넷에서 모은 대규모 학습 데이터셋 덕분에 일반화 능력을 갖추고 있음
- 그러나 로봇 분야에서는 이러한 규모의 사전 학습을 재현하는 것이 여전히 과제로 남아있음
    
    → 가장 큰 로봇 조작 데이터셋이 10만~100만 개의 예제밖에 없음…
    
- **그래서 Vision-Language model을 로봇 정책 학습이 핵심 구성 요소로 활용함으로써, 일반화 성능을 더 끌어올려보자**
- 그래서 원래는 VLA(Vision-Language-Action) model이 존재하긴 했음
    
    → pre-train된 model을 로봇 컨트롤 액션을 생성하는데 직접적으로 활용함 그러나, 두가지 문제점 있음
    
    1. 기존 모델들은 비공개로 유지되어 모델 구조, 학습 절차, 데이터 구성에 대한 정보가 제한적임
    2. 기존 연구들은 일반 하드웨어(consumer-grade GPUs)에서 VLA를 새로운 로봇 또는 환경및 작업에 적용시키기 위한 최적의 방법을 제공하지 못함
- 그래서 이 연구는 오픈소스이면서 효과적인 fine tuning과 적응을 지원하는 일반적인 VLA 모델이 필요하다고 주장함
- 그래서 Open VLA라는 새로운 오픈소스 모델을 소개함
    - 70억개 파라미터
    - pre-trained Visual based language model backbone
    - Open-X Embodiment 데이터셋의 97만개 로봇 조작 궤적 데이터로 fine tuning
    - 55억개 파라미터의 RT-2-X보다 16.5% 높은 작업 성공률
    - WidowX와 Google 로봇 환경에서 29개의 작업에 대해 평가됨
- 다양한 fine tuning 전략도 탐구함
    - LoRA, 모델 양자화를 활용해서 소비자급 GPU에서도 효율적으로 fine tuning 가능함을 보임

# 2. Related Work

### Visually-Conditioned Language Models, VLMs

- 인터넷 규모의 데이터로 학습되어 입력 이미지와 언어 프롬프트를 기반으로 자연어를 생성
- VQA, Object Localization 등의 다양한 응용분야에 사용됨
- 초기 연구는 vision, language feature간 cross attention 구조를 탐구했으나 최근에는 Patch-as-Token 기법으로 단순화됨
    - pre-trained visual transformer의 패치 feature를 토큰으로 간주하여 language model input space에 투영함

### Generalist Robot Policies

- 기존 연구(Octo)들은 pre-train된 언어 임베딩이나 비전 인코더 같은 컴포넌트를 새로 초기화된 추가 컴포넌트와 결합하여 학습을 통해 재구성하는 방식을 사용함
- OpenVLA는 보다 통합적인 접근 방식을 채택하여 VLMs를 fine-tuning하여 로봇 동작을 언어 모델의 토큰으로 생성하게 함

### Vision-Language-Action Models, VLAs

- VLA의 주요 이점
    1. 인터넷 규모의 vision-language 데이터셋에서 pre-train된 vision, language 컴포넌트의 정렬 수행
    2. 로봇 제어에 특화되지 않은 일반적인 아키텍처를 사용하여 현대 VLM 학습 인프라를 활용, 최소한의 코드 수정으로 수십억 파라미터 정책 학습 가능
    3. VLM의 빠른 개선으로부터 로봇 공학이 직접적인 이점을 얻을 수 있음
- OpenVLA가 RT-2-X에 비해 가지는 차별성
    1. 강력한 오픈 VLM 백본과 더 풍부한 로봇 사전 훈련 데이터셋을 결합하여, RT-2-X보다 성능이 우수하면서도 모델 크기는 1/10으로 작음
    2. OpenVLA는 새로운 대상 환경에 대한 fine tuning을 철저히 조사함(새로운 시도)
    3. 최신 파라미터 효율적 fine tuning 및 양자화 접근법을 VLA에 최초로 적용함
    4. 최초의 오픈소스 일반화 VLA

# 3. The OpenVLA Model

## 3.1. **Preliminaries: Vision-Language Models**

- 최신 VLM 아키텍처는 다음 세가지 주요 구성요소로 이루어짐
    1. **vision encoder** : 입력 이미지를 “이미지 패치 임베딩”으로 변환함
    2. **projector** : 비전 인코더의 출력 임베딩을 언어 모델의 입력 공간으로 매핑함
    3. **LLM 백본** : 언어 처리를 담당함
- 인터넷 소스에서 수집된 vision-language 데이터를 기반으로 다음 텍스트 토큰을 예측하는 목표로 학습
- prismatic-7B VLM 기반으로 구축
- 6억 파라미터의 비전 인코더
- 2층 MLP 프로젝터
- 70억 파라미터 Llama 2 backbone
- prismatic = pre-trained SigLIP, DINOv2로 이루어진 2개의 비전 인코더 결합
    - 입력 이미지 패치를 처리하고 각각의 feature vector를 채널 단위로 결합함
    - 특히 DINOv2 특징은 로봇 제어에서 중요한 공간적 추론을 개선하는데 유용함

## 3.2. **OpenVLA Training Procedure**

<img width="724" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 52 13" src="https://github.com/user-attachments/assets/b39985fe-14c0-429e-a816-824ccfe9b5c8">


- OpenVLA를 학습하기 위해 pre-trained Prismatic-7B VLM backbone을 로봇 동작 예측에 맞게 미세조정함
- 동작 예측 문제를 ‘vision-language’ task로 정의하여 입력 관찰 이미지와 자연어 작업 지시를 예측된 로봇 동작의 문자열로 매핑함
- 연속적인 로봇 동작을 언어 모델 토크나이저의 discrete 토큰으로 변환하여 출력공간에 나타냄
- 로봇 동작의 각 차원(움직이는 독립적인 관절과 축 등을 의미)을 개별적으로 256개의 구간(bin)으로 이산화함
- 각 동작 차원의 구간 너비는 훈련 데이터의 1분위수와 99분위수 간격을 균등하게 나누어 설정함 (1%와 99%만을 사용하여 극단적인 값을 배제함) → 데이터 극단값의 영향을 무시하고 이산화 간격의 효과적인 세분화를 유지함
- 이로써 N차원 로봇 동작에 대해 [0, … ,255] 범위의 N개의 discrete한 정수를 얻음
- 그런데 문제는 OpenVLA의 language backbone인 Llama 토크나이저는 fine tuning 중 새로 도입된 토큰에 대해 100개의 특수 토큰만 사용할 수 있어서 가용 토큰 수가 부족함
- 그래서 Llama 토크나이저 어휘에서 가장 적게 사용되는 256개 토큰 덮어써서 로봇 동작 토큰으로 바꿈

## 3.3. Training Data

- 구축 목표 : 다양한 로봇 형태, 장면, 작업을 포괄하는 것 → 최종 모델이 여러 로봇을 즉시 제어할 수 있으며, 새로운 로봇 설정에 대해 효율적으로 fine tuning 할 수 있음
- OpenVLA 학습 데이터셋의 기반은 Open X-Embodiment : 70개 이상의 개별 로봇 데이터, 200만개 이상의 로봇 궤적 데이터
- 데이터 정제
    1. 모든 학습 데이터셋의 일관된 입력 및 출력 공간 보장
        - OpenX 데이터셋에서 최소 한 개의 3인칭 카메라를 사용하고 단일 로봇 팔 조작 데이터를 포함하는 데이터셋만을 학습 데이터로 제한
    2. 로봇 형태, 작업, 장면의 균형 있는 조합
        - Octo 모델에서 사용된 데이터 믹스 가중치를 기반으로 필터링된 데이터셋을 평가하고, 다양성이 낮은 데이터셋의 가중치를 낮추거나 제거함
        - 반면, 작업과 장면의 다양성이 높은 데이터셋의 가중치는 높임

## 3.4. **OpenVLA Design Decisions**

(초기 실험에서는 OpenX 대신 BridgeData V2를 사용해 반복 속도를 높이고 계산 비용도 줄임)

### **VLM Backbone**

- Prismatic, IDEFICS-1, LLaVA 를 테스트함
- **IDEFICS-1, LLaVA**
    - 단일 객체가 포함된 작업에서 유사한 성능을 보였으나, LLaVA는 다중 객체 환경에서 언어 지시를 기반으로 올바른 객체를 조작해야하는 작업에서 더 강력한 언어 기반 성능을 보임
    - 데이터셋 5개 언어 기반 작업에서 LLaVA가 성공률이 35% 더 높았음
- **Prismatic VLM**
    - 추가 성능 향상을 달성
    - 단순 작업, 다중 객체 언어 기반 작업 모두 LLaVA를 약 10% 초과하는 성공률을 보임
        - Prismatic의 SigLIP-DinoV2 backbone의 융합이 제공하는 공간적 추론 능력 향상 덕분으로 분석
    - Prismatic은 뛰어난 성능 뿐 아니라 모듈형이고 사용하기 쉬운 코드베이스를 제공 → 최종 선택

### Image Resolution

- 224x224px vs 384x384px → 성능은 비슷, 후자가 학습 시간이 3배 더 걸림 → 224ver. 최종 선택

### Fine-Tuning Vision Encoder

- 기존 연구에서는 비전 인코더를 고정하는것이 높은 성능을 보인다고 보고됨
- 그러나 VLA학습에서는 인코더를 fine-tuning하는 것이 좋은 성능을 보임
    - pre-trained backbone이 로봇 제어에 필요한 세부적인 공간 정보를 충분히 포착하지 못할 수 있기 때문일 것이라고 추측

### Training Epochs

- 일반적인 LLM이나 VLM은 데이터셋을 1~2회 정도만 반복함
- 그러나 VLA 경우 데이터셋 학습 반복하는 것이 중요함 → 실제 로봇 성능이 훈련 데이터 동작 토큰 정확도가 95%를 초과할 때까지 지속적으로 증가함
- 최종적인 Epochs = 27

### Learning Rate

- 2e-5
- warmup은 큰 이점을 제공하지 않음

## 3.5. **Infrastructure for Training and Inference**

- **학습 환경**
    - 64개의 A100 GPU 클러스터에서 14일 동안 학습됨
    - 총 21,500 GPU-time 소요됨
    - batch size = 2048
- **Inference 성능**
    - 15GB의 GPU 메모리 사용
- **메모리 최적화**
    - 양자화를 통해 메모리 사용량 줄일 수 있으며, 실제 로봇 작업에서 성능을 유지하면서 메모리 step을 크게 줄일 수 있었음
- **원격 inference 지원**
    - 실시간으로 로봇 동작 예측 을 스트리밍할 수 있는 원격 VLA inference 서버를 제공함
    - 로컬 컴퓨팅 장치 없이도 로봇 제어할 수 있음

# 4. **The OpenVLA Codebase**

# 5. Experiments

- 다음 세가지의 질문에 답하고자 함
    1. OpenVLA는 여러 로봇과 다양한 일반화 시나리오에서 기존의 일반화 로봇 정책들과 비교했을 때 어떤 성능을 보이는가?
    2. OpenVLA는 새로운 로봇 설정과 작업에 효과적으로 fine-tuning될 수 있는가? 그리고 데이터 효율적인 최첨단 모방 학습 방법과 비교하면 어떠한가?
    3. OpenVLA 모델의 학습 및 inference를 위한 계산 요구를 줄이고 접근성을 높이기 위해 파라미터 효율적 finetuning과 양자화를 사용할 수 있는가? 그리고 성능과 계산 자원의 균형은 어떠한가?

## 5.1. **Direct Evaluations on Multiple Robot Platforms**

### **Robot Setups and Tasks**

- 두가지 로봇 형태에서 실험을 진행함
    1. **WidowX robot** : BridgeData V2 평가에서 사용된 로봇 (맨 위 그림 왼쪽)
    2. **Google robot** : RT-1, RT-2 평가에서 사용된 모바일 조작 로봇 (맨 위 그림 가운데)
- 여러가지 평가 task set을 정의함
    - 시각적 일반화 : 보지 못한 배경, 방해 객체, 객체의 색상/외형
    - 운동적 일반화 : 보지 못한 객체 위치/방향
    - 물리적 일반화 : 보지 못한 객체 크기/형태
    - 의미적 일반화 : 인터넷에서 학습한 보지 못한 대상 객체, 지시, 개념
- 또한 여러 객체가 포함된 장면에서 사용자가 프롬프트로 지정한 올바른 객체를 조작할 수 있는지 언어 조건화 능력도 평가함

<img width="739" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 11 52" src="https://github.com/user-attachments/assets/70c3fe79-1b90-49d3-8190-a0e4a1902806">


<img width="369" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 12 22" src="https://github.com/user-attachments/assets/716d344b-b0da-4466-81fb-3d129368a1b1">


### Comparisons

- 기존의 3가지 일반화된 조작 정책과 비교함
    1. RT-1-X(35M parameter) : OpenX 데이터셋 하위 집합에서 학습된 트랜스포머 정책
    2. Octo(93M parameter) : 오픈소스 조작 정책 중 최첨단 모델
    3. RT-2-X(55B parameter) : 인터넷 기반 사전 학습된 vision-language backbone 사용한 비공개 VLA 모델
- 결과는 다음과 같음
    - RT-1-X, Octo : 테스트 작업에서 어려움을 겪었으며, 특히 방해 객체가 있을 때 올바른 객체를 조작하지 못하거나, 로봇팔을 무의미하게 휘두르는 경우가 많았음
    - 기존 연구들보다 더 높은 수준의 일반화를 요구하는 작업을 평가했기 때문에, 인터넷 사전 학습이 없는 모델의 낮은 성능은 예견된 결과임
    - RT-2-X : 앞 두 성능을 기록하며, 대규모 사전학습된 VLM의 이점을 보여줌
    - OpenVLA : 다양한 작업에서 더 나은 로봇 행동을 보였고, 특히 방해 객체를 피하며 올바른 객체를 조작함

## 5.2. **Data-Efficient Adaptation to New Robot Setups**

- 새로운 작업 및 로봇 설정에 VLA 모델을 효과적으로 fine tuning하는 방법에 대한 실험
- 새로운 실제 로봇 설정에 빠르게 적응할 수 있는지에 대한 능력을 조사함

### **Robot Setups and Tasks**

<img width="723" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 22 37" src="https://github.com/user-attachments/assets/d9e62207-1340-46af-b236-9a3a0c3ea9b9">


- full fine-tuning : 모델의 모든 파라미터 조정, 목표 작업의 10~150개 시연 데이터로 학습을 수행함
- 테스트 설정
    1. Franka-Tabletop : 고정형 테이블에 장착된 로봇팔
    2. Franka-DROID : 이동형 스탠딩 데스크에 장착된 로봇팔

### Comparisons

- 다음의 모델들과 비교함
    1. Diffusion Policy : 데이터를 효율적으로 사용하는 모방 학습 방법, 처음부터 학습된 모델
        
        (동작 궤적을 diffusion을 통해 생성하는 방법, 특정 지시에 따라 적절한 동작 궤적 생성 가능, 부드럽고 정교한 궤적, 데이터 효율성 등이 장점)
        
    2. Diffusion Policy(matched) : OpenVLA 입출력 사양에 맞춘 Diffusion Policy 버전
    3. Octo
    4. OpenVLA(scratch) : OpenX 사전학습 모델이 아닌, Prismatic VLM을 대상으로 직접 finetuning한 모델
    5. OpenVLA(ours)

<img width="725" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 27 04" src="https://github.com/user-attachments/assets/f8ea0231-e402-45e3-b021-4f5430f1a2ad">


- Diffusion Policy : 단일 명령 기반의 좁은 작업(ex. ‘당근을 그릇에 넣기’, ‘옥수수를 냄비에 붓기’)에서는 우수한 성능을 보임
- OpenVLA, Octo : 다중 객체가 포함된 작업이나 언어지시를 활용한 작업에서 더 나은 성능을 보임
    
    → OpenX pre-train이 중요한 언어 기반 작업에서 모델의 적응 능력을 향상시켰기 때문(OpenVLA(scratch)의 낮은 성능이 이를 뒷받침)
    

<종합 평가>

- OpenVLA는 테스트된 모든 작업에서 평균적으로 가장 높은 성능을 보임
- 기존 연구들은 좁은 단일 명령 작업이나 다양한 다중 명령 작업 중 한가지에만 강한 성능을 보였으나 OpenVLA는 모든 테스트 작업에서 최소 50% 이상의 성공률을 기록한 모델임
- OpenVLA가 특히 다양한 언어 지시를 포함하는 모방 학습 작업에서 강력한 기본 옵션이 될 수 있음을 시사함
- 반면 Diffusion Policy는 좁고 고도로 섬세한 작업에서 여전히 더 부드럽고 정밀한 궤적을 보여줌. Diffusion Policy에서 구현된 action chunking, temporal smoothing을 OpenVLA에 도입한다면 유사한 수준의 섬세함을 달성할 가능성이 있음

## 5.3. **Parameter-Efficient Fine-Tuning**

<img width="444" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 48 29" src="https://github.com/user-attachments/assets/44b9f0bf-1313-40f0-b1d9-c74d932fccd6">


- 계산량과 파라미터를 더욱 효율적으로 사용하는 fine tuning 방식을 탐구하고 평가함
- 다음의 fine tuning 방법들을 비교함
    1. full fine-tuning : 모든 파라미터를 업데이트함
    2. last layer only : OpenVLA의 트랜스포머 백본과 토큰 임베딩 매트릭스 마지막 레이어만 업데이트
    3. frozen vision : 비전 인코더는 고정하고, 나머지 파라미터만 fine tuning
    4. sandwich fine-tuning : 비전 인코더, 토큰 임베딩 매트릭스, 마지막 레이어만 fine tuning
    5. Low-Rank Adaptation : 모든 파라미터 업데이트하지 않고, rank가 낮은 매트릭스 사용해서 일부 파라미터만 학습하는 방식 채택

<종합 평가>

- last layer only : 성능 매우 낮음 → 전체 네트워크 적응력이 부족하다는 것을 보임
- froze vision : 성능 낮음 → 비전 특징이 타겟 장면에 충분히 적응하지 못했기 때문
- sandwich fine-tuning : 비전 인코더를 포함하여 필요한 주요 레이어를 조정했기 때문에 더 나은 성능을 보임, 전체 backbone을 fine tuning하지 않아 GPU 메모리 소비를 줄임
- LoRA : 성능과 메모리 소비 간의 최상의 균형을 이룸. 전체 fine tuning과 유사한 성능을 달성하면서 전체 파라미터의 1.4%만 조정함, 계산량은 8배 감소함

## 5.4. Memory-Efficient Inference via Quantization

<img width="720" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-08_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7 00 46" src="https://github.com/user-attachments/assets/a968bb07-0f7f-47cf-ac49-5d262231d019">



- 현대적인 양자화 기법을 사용해 inference에 필요한 메모리를 추가로 줄이고 VLA 정책의 접근성을 넓힐 수 있는지 테스트함
- 8bit, 4bit 정밀도 사용, 8개 작업 대상으로 실험 진행, 메모리 사용량/작업 수행 성능/다양한 소비자 및 서버용 GPU에서의 제어 주파수(속도)를 평가 지표로 함

<종합 평가>

- 4 bit 양자화가 openVLA 성능을 유지하면서도 메모리 사용량과 추론 속도를 최적화하는데 가장 효과적인 접근 방식으로 나타남

# 6. **Discussion and Limitations**

<discussion>

- OpenVLA는 즉시 사용(out-of-the-box)이 가능하며, 우수한 성능을 발휘함
- 파라미터 효율적인 fine tuning을 통해 새로운 로봇 설정에 쉽게 적응할 수 있음을 입증함

<Limitations>

1. 단일 이미지 관찰만 지원
    - 실제 세계의 로봇환경은 매우 다양하며, 폭넓은 센서 입력이 필요하지만, OpenVLA는 단일 이미지 입력만 지원함
    - 이미지와 텍스트 데이터가 혼합된 데이터로 pre-train된 VLM을 활용하면 이러한 유연한 입력을 처리하는 VLA fine tuning을 더 쉽게 할 수 있음
2. inference 처리량의 한계
    - 처리량 개선은 정교한 양손 조작(bi0maunal manipulation) 작업을 테스트할 수 있는 기반이 됨
    - action chunking 또는 최적화 기법을 탐구하는 것이 잠재적인 대안으로 제시될 수 있음
3. 성능 개선의 여지
    - 아직 매우 높은 신뢰성(90% 이상)이 나오지는 않음
