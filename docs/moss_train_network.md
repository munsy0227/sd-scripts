# sd-scripts 포크 한국어 사용 설명서

이 문서는 `munsy0227/sd-scripts`의 **MOSS Image 학습 기능**을 설명한다.
기존 sd-scripts에 공식 MOSS 트레이너와 LyCORIS LoKR·DoRA, TXT/JSON/layout 캡션,
다중 캡션 캐시, H2D-only block swap을 연결한 포크다.

기준은 2026-10-25의 `codex/sd-scripts-moss-dev-integration` 작업 트리다.
포크의 `main`과 작업 브랜치는 MOSS 구현이 다르며, 원격 브랜치에 아직 반영되지 않은
로컬 변경이 있을 수 있다. 다른 버전을 사용한다면 해당 버전의 `--help`도 확인한다.
SDXL·FLUX·Anima 등은 [모델별 문서](#다른-모델과-상세-문서)를 참고한다.

## 목차

- [빠른 시작](#빠른-시작)
- [설치](#설치)
- [모델 준비](#모델-준비)
- [TXT와 JSON 데이터셋](#txt와-json-데이터셋)
- [학습](#학습)
- [메모리와 캐시](#메모리와-캐시)
- [샘플 이미지와 학습 재개](#샘플-이미지와-학습-재개)
- [ComfyUI](#comfyui)
- [문제 해결](#문제-해결)
- [검증 범위](#검증-범위)
- [다른 모델과 상세 문서](#다른-모델과-상세-문서)

## 빠른 시작

처음 사용할 때는 다음 순서로 진행한다.

1. sd-scripts와 **MOSS 프리셋이 있는 LyCORIS 포크**를 같은 Python 환경에 설치한다.
2. BF16 MOSS DiT, Qwen3.5 텍스트 인코더, FLUX.2 VAE, 토크나이저를 준비한다.
3. 이미지와 같은 이름의 캡션 파일을 준비한다. 기존 TXT를 JSON으로 바꿀 필요는 없다.
4. [학습](#학습)의 `train.toml`에서 모델·데이터·출력 경로를 맞춘다.
5. sd-scripts 폴더에서 실행한다.

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  moss_train_network.py --config_file /workspace/data/train.toml
```

경로는 자신의 환경에 맞춰 바꾼다. 이미 사용하는 `train.toml`이 있다면 이름과 경로를
유지하고 필요한 항목만 수정한다. 별도 dataset 설정 파일은 필수가 아니다.

## 설치

### 저장소와 가상환경

처음 내려받는 경우의 예다. 기존 작업 폴더에 수정 내용이 있다면 다시 clone하거나
브랜치를 바꾸기 전에 `git status`로 먼저 확인한다.

```sh
git clone --branch codex/sd-scripts-moss-dev-integration https://github.com/munsy0227/sd-scripts.git
git clone --branch codex/moss-sd-scripts-integration https://github.com/munsy0227/LyCORIS.git
cd sd-scripts
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

위 명령은 Linux/WSL의 예다. Windows에서는 `python -m venv .venv`로 만들고
PowerShell에서 `.\.venv\Scripts\Activate.ps1`로 활성화한다.

먼저 [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)에서 OS와 GPU에
맞는 PyTorch를 설치한 뒤 나머지 의존성을 설치한다. PyTorch와 torchvision은 서로
호환되는 조합을 사용한다.

```sh
python -m pip install -r requirements.txt
python -m pip install ../LyCORIS
accelerate config
```

`../LyCORIS`는 위처럼 두 저장소를 나란히 받은 경우다. 이미 설치한 로컬 포크가 있다면
그 경로를 사용한다. 일반 PyPI 버전만 설치해서는 이 포크의 동작과 같다고 보장할 수 없다.
`accelerate config`에서는 실제 GPU 수와 실행 환경에 맞춰 설정한다.

현재 `requirements.txt`는 transformers 5.17.0, diffusers 0.40.0,
accelerate 1.15.0을 지정한다. 예전 가상환경에 파일만 복사했다면 의존성도 갱신해야 한다.

### 설치 확인

```sh
python -c "import torch, transformers, diffusers; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__, diffusers.__version__)"
python -c "from transformers import Qwen3_5TextModel; from diffusers import AutoencoderKLFlux2; import lycoris.kohya; print('imports OK')"
python moss_train_network.py --help
```

NVIDIA GPU 학습을 준비했다면 첫 명령의 CUDA 사용 가능 여부가 `True`인지 확인한다.
두 번째 명령에서 클래스 import가 실패하면 학습 설정을 바꾸기 전에 환경 설치부터 확인한다.

## 모델 준비

### 모델 루트 폴더 사용

다음처럼 구성된 로컬 모델 루트를 지정할 수 있다.

```text
/models/MOSS-Image-Base/
├── transformer/     # DiT 가중치와 shard index
├── text_encoder/    # Qwen3.5 가중치
├── vae/             # FLUX.2 VAE 가중치
├── tokenizer/       # 토크나이저 파일
└── scheduler/       # 선택 사항: scheduler_config.json
```

`pretrained_model_name_or_path`에 `/models/MOSS-Image-Base`를 넣는다.
토크나이저는 `tokenizer/`를 우선하고, 폴더가 없으면 `text_encoder/`에서 찾는다.
`scheduler/`가 없으면 학습 shift 1, 샘플 shift 5, 1000 timesteps 기본값을 사용한다.

모델 가중치는 로컬 경로에서 읽는다. 모델 이름만 지정한다고 가중치를 자동으로 받는 방식이 아니다.
모델 배포 정보는 [MOSS 공식 저장소](https://github.com/OpenMOSS/MOSS-Image)를 참고한다.

### 개별 파일 사용

공식 분할 체크포인트의 첫 shard 또는 실수형 ComfyUI 단일 파일도 입력할 수 있다.
분할 파일은 나머지 shard도 같은 폴더에 있어야 한다.

```toml
pretrained_model_name_or_path = "/models/moss_image_base_bf16.safetensors"
text_encoder = "/models/qwen3.5_9b_bf16.safetensors"
vae = "/models/flux2-vae.safetensors"
tokenizer_cache_dir = "/models/moss-tokenizer"
```

위 파일명은 경로 예시다. 실제 받은 파일명으로 바꾼다. 명시한 `text_encoder`와 `vae`는
모델 루트에서 자동 탐색한 경로보다 우선한다.

이 포크의 MOSS 학습에는 BF16 등 **실수형 가중치**를 사용한다.
ConvRot INT8은 통합하지 않았고, 현재 LyCORIS full-matrix LoKR+DoRA와도 호환되지 않는다.
`fp8_base` 등으로 대신 양자화하는 경로도 지원하지 않는다.

오프라인에서는 토크나이저 파일을 모두 준비하고 `tokenizer_cache_dir`를 명시한다.
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 설정하면 허브 접근을 막을 수 있다.
토크나이저 경로를 생략한 온라인 환경에서는 캐시를 조회하거나 내려받을 수 있다.

## TXT와 JSON 데이터셋

### 별도 dataset 설정 없이 사용

`train_data_dir` 아래에 `<반복 횟수>_<분류명>` 폴더를 둔다.

```text
/workspace/data/images/
└── 10_my_style/
    ├── 001.png
    ├── 001.txt
    ├── 002.jpg
    └── 002.txt
```

이때 `train_data_dir`는 `/workspace/data/images`다. `10`은 이미지마다 적용되는 반복 횟수다.
이미지와 캡션의 확장자를 제외한 이름은 같아야 한다. TXT 캡션의 예는 다음과 같다.

```text
style_trigger, character_name, 1girl, solo, green hair, An anime girl waves against a pink background.
```

기존 `트리거, 캐릭터, 태그, 자연어 설명` 순서를 그대로 사용할 수 있다.
`shuffle_caption=false`, `enable_wildcard=false`를 유지하면 해당 처리를 켜지 않는다.

### 캡션 형식 선택

| 설정 | 동작 |
| --- | --- |
| `caption_extension=".txt"` | 같은 이름의 TXT sidecar를 읽는다. |
| `caption_extension=".json"` | 같은 이름의 JSON sidecar를 읽는다. |
| `caption_format="auto"` | 일반 텍스트와 지원하는 두 JSON 구조를 구분한다. MOSS 기본값이다. |
| `caption_format="text"` | 일반 텍스트로 처리한다. |
| `caption_format="moss_caption"` | `caption.long/medium/short` 중 선택한 문장을 읽는다. |
| `caption_format="moss_layout"` | `summary/items/bbox` 구조를 레이아웃 프롬프트로 읽는다. |
| `moss_caption_length="long"` | JSON 문장 길이를 선택한다. `medium`, `short`도 가능하다. |

일반 TXT는 기본적으로 첫 줄을 읽는다. 한 문장의 여러 줄을 모두 사용하려면 아래의
`enable_multiline_caption`을 켠다. JSON에서 선택한 문장의 의도된 줄바꿈은 보존한다.

여러 폴더에 서로 다른 확장자·캡션 형식·반복 횟수를 지정하려면 기존 dataset 설정 방식을
사용할 수 있다. 이미 이 방식을 쓰고 있다면 `dataset_config` 경로를 유지한다.
[기존 MOSS dataset 예제](../examples/moss/dataset.toml)와
[공통 설정 문서](config_README-en.md)를 참고한다. 이 설명서를 쓰기 위해 새 설정 파일을 만들 필요는 없다.

### 문장 JSON

```json
{
  "caption": {
    "long": "style_trigger, A girl with green hair waves.\nA pink background is behind her.",
    "medium": "A girl with green hair waves.",
    "short": "A waving girl."
  },
  "watermark": [],
  "borders": ["none"]
}
```

`moss_caption_length`로 선택한 필드가 비어 있으면 오류로 중단한다.
`watermark`와 `borders`는 학습 프롬프트에 넣지 않는다.
이 구조는 길이별 문장을 선택하는 형식이며, 세 문장을 무작위로 바꾸는 옵션은 아니다.

### 레이아웃 JSON

```json
{
  "summary": {"caption": "A girl at the center of the image."},
  "items": [
    {"label": "girl", "caption": "green hair", "bbox": [-256, -128, 256, 128]}
  ],
  "bbox_format": "yxyx",
  "bbox_coord_space": "abs_centralized"
}
```

`bbox`는 `[y_min, x_min, y_max, x_max]`이며 **원본 이미지 중심이 원점인 픽셀 좌표**다.
0~1로 정규화한 좌표가 아니다. 위 박스는 이미지 중심에서 위아래 256픽셀, 좌우 128픽셀 범위다.

이미지의 resize/crop/flip에 맞춰 좌표도 변환한다. 잘린 박스는 경계에 맞추고,
완전히 사라진 항목은 제거한다. `caption_prefix`와 `caption_suffix`는 `summary.caption`에 들어간다.
레이아웃에는 일반 bucket/crop을 사용하고 face crop을 사용하지 않는다.

### 여러 캡션 후보

다음 설정을 켠다.

```toml
enable_multiline_caption = true
multiline_caption_separator = "---===---"
```

TXT 파일에서 구분선으로 후보를 나눈다. 구분선 없이 여러 줄만 쓰면 하나의 여러 줄 캡션이다.

```text
style_trigger, A girl waves.
Her hair is green.
---===---
style_trigger, A waving girl with green hair.
```

각 학습 step에서 후보 하나를 선택한다. MOSS 텍스트 캐시는 모든 후보를 저장하므로
이 선택 기능과 함께 사용할 수 있다. 후보마다 prefix/suffix가 적용되며, 캐시 용량과
인코딩 작업량은 후보 수에 따라 늘어난다. 문장 JSON·레이아웃 JSON을 이 구분선으로 나누지는 않는다.

## 학습

### 두 실행 파일의 차이

| 항목 | `moss_train_network.py` | `moss_image_train_network.py` |
| --- | --- | --- |
| 목적 | 기존 LyCORIS 사용법을 보존하는 진입점 | 공식 방식의 공통 트레이너 진입점 |
| 기본 network module | `lycoris.kohya` | `networks.lora_generic` |
| 기본 timestep 분포 | `legacy` | `sigmoid` |
| Qwen adapter 학습 | 지원하지 않음 | 명시적으로 켠 경우 가능 |
| 기본 샘플 steps | 30 | 50 |

이 포크의 LoKR+DoRA를 사용할 때는 `moss_train_network.py`로 시작하면 된다.
두 진입점은 같은 모델·공통 트레이너를 사용하지만, 기본값이 달라 실행 파일만 바꿔도
학습이 같아지는 것은 아니다. Qwen adapter는 [공식 진입점 설명](moss_image_train_network.md)을 참고한다.

### train.toml 설정

아래는 별도 dataset 파일 없이 TXT 이미지 폴더를 읽는 구성이다.
이미 사용하는 `train.toml`에 맞춰 반영한다. 경로와 수치는 실행 방법을 설명하기 위한 값이며,
화질·VRAM에 맞춘 최적값을 뜻하지 않는다.

```toml
pretrained_model_name_or_path = "/models/MOSS-Image-Base"
train_data_dir = "/workspace/data/images"
output_dir = "/workspace/output"
output_name = "moss-style"

network_module = "lycoris.kohya"
network_train_unet_only = true
network_dim = 16
network_alpha = 16
network_args = [
  "algo=lokr",
  "preset=moss",
  "full_matrix=True",
  "factor=4",
  "weight_decompose=True",
]

resolution = "1024,1024"
enable_bucket = true
bucket_reso_steps = 16
train_batch_size = 1
gradient_accumulation_steps = 1
max_data_loader_n_workers = 0

caption_extension = ".txt"
caption_format = "auto"
shuffle_caption = false
enable_wildcard = false

mixed_precision = "bf16"
save_precision = "bf16"
attn_mode = "torch"
loss_type = "l2"
timestep_sampling = "sigmoid"
moss_max_token_length = 4096
moss_text_encoder_device = "cpu"

gradient_checkpointing = true
blocks_to_swap = 20
cache_latents_to_disk = true
cache_text_encoder_outputs_to_disk = true

optimizer_type = "AdamW"
learning_rate = 0.0001
lr_scheduler = "constant"
max_train_steps = 1000
seed = 42

save_model_as = "safetensors"
save_every_n_steps = 100
save_state = true
```

実行 명령은 [빠른 시작](#빠른-시작)과 같다. 별도 dataset 설정 없이 쓰려면 기존 설정에
남아 있는 `dataset_config`도 확인한다. dataset 설정을 지정하면 그 안의 데이터 구성을 사용한다.

위 예시는 `sigmoid`를 **명시적으로** 선택했다. 예전 명령의 학습 분포를 보존하려면
기존 설정을 유지하거나 `timestep_sampling="legacy"`를 지정한다. 분포 변경은 기존 학습의
이후 경로를 바꾸므로 재개 중에 임의로 바꾸지 않는다.

`loss_type`은 MOSS flow matching에 맞게 `l2`를 사용한다.
`min_snr_gamma`, `debiased_estimation_loss` 등 DDPM/SNR용 옵션은 사용하지 않는다.

### LoRA·LoKR·DoRA와 학습 범위

| 목적 | `network_args`에 넣을 값 |
| --- | --- |
| 기본 LoRA | `"algo=lora", "preset=moss"` |
| full-matrix LoKR | `"algo=lokr", "preset=moss", "full_matrix=True", "factor=4"` |
| full-matrix LoKR+DoRA | 위 LoKR 설정에 `"weight_decompose=True"` 추가 |

LoRA로 바꿀 때는 `network_args` 목록을 LoRA용 값으로 교체한다.
`network_dim`과 `network_alpha`는 일반 LoRA에서 의미가 있다. 이 포크의 full-matrix LoKR는
스케일 1.0을 사용하므로 alpha/rank로 추가 스케일하지 않는다.

| 범위 | 기본 MOSS 모델에서의 대상 수 | 의미 |
| --- | ---: | --- |
| LyCORIS `preset=moss` | 252 | 42개 DiT 블록의 주요 Linear |
| LyCORIS `preset=moss-full` | 269 | 위 범위와 DiT 내부 임베딩·변조·출력 계층 |
| generic `train_text_embedder=true` | 263 | 공식 generic adapter의 별도 범위 |

`moss-full`에 포함된 text embedder는 **DiT 내부 계층**이다. 별도 Qwen3.5를 학습하는 뜻이 아니다.
Qwen과 VAE는 위 LyCORIS 구성에서 고정한다. `train_llm_adapter=true` 같은 다른 network module의
옵션을 LyCORIS 설정에 그대로 옮기지 않는다.

## 메모리와 캐시

### 메모리를 줄이는 순서

1. `train_batch_size=1`로 두고 더 큰 실효 batch는 `gradient_accumulation_steps`로 만든다.
2. latent 캐시와 텍스트 출력 캐시를 사용한다. Qwen 텍스트 계산이 끝나면 DiT 로드 전에 해제한다.
3. `gradient_checkpointing=true`를 사용한다.
4. `blocks_to_swap`으로 DiT 블록 일부를 CPU로 옮긴다. MOSS의 최대값은 40이다.
5. 필요하면 아래의 H2D-only 방식을 선택한다.

CPU로 옮긴 모델과 캐시 계산에도 시스템 RAM이 필요하다. swap 수를 늘리면 보통 VRAM은 줄고
전송 비용은 늘어난다. GPU 종류·RAM·해상도·adapter 범위가 달라 고정된 VRAM/속도를 보장하지 않는다.
block swap을 사용할 때는 microbatch 1 제약도 지킨다.

### H2D-only block swap

기존 `blocks_to_swap` 설정에 다음을 추가한다.

```toml
block_swap_h2d_only = true
block_swap_ring_size = 2
use_pinned_memory_for_block_swap = false
```

`gradient_checkpointing=true`도 필요하다. 동결한 기본 가중치를 CPU에 보관하고 GPU로만 복사하므로,
기존 방식의 GPU→CPU 복사를 줄인다. 이 옵션을 켜지 않으면 기존 classic 방식으로 동작한다.

| 옵션 | 의미 |
| --- | --- |
| `block_swap_ring_size=2` | GPU 스트리밍 버퍼 수. 기본값 2로 다음 블록 전송과 계산을 겹칠 수 있다. |
| `block_swap_ring_size=1` | 버퍼 메모리는 줄지만 전송·계산 겹치기가 제한된다. |
| `use_pinned_memory_for_block_swap=true` | CPU master 가중치를 pinned 메모리에 보관한다. 호스트 메모리 사용 특성도 달라진다. |

H2D-only는 CUDA와 동결한 기본 모델의 adapter 학습을 위한 기능이다. 전체 모델 미세조정용이 아니다.
MOSS LoKR+DoRA의 두 프리셋에서도 축소 BF16 gradient 비교를 확인했다.
자세한 구조는 [block swap 문서](block_swap.md)를 참고한다.

### 함께 쓸 수 없는 옵션

| 조합 | 처리 |
| --- | --- |
| block swap + `cpu_offload_checkpointing` | 함께 사용하지 않는다. |
| block swap + `unsloth_offload_checkpointing` | 함께 사용하지 않는다. |
| CPU offload + unsloth offload | 하나만 선택한다. |
| H2D-only 학습 + gradient checkpointing 끔 | 오류로 중단한다. |
| `compile` + `torch_compile` | 함께 사용하지 않는다. |
| `compile_fullgraph` + `split_attn` | 함께 사용하지 않는다. |

처음에는 `attn_mode="torch"`로 확인한다. xformers를 선택하면 해당 패키지와 `split_attn=true`가
필요하다. `sageattn`은 이 트레이너에서 학습용으로 사용할 수 없다.

### 캐시를 켜거나 꺼야 하는 경우

처음 실행하면 캐시를 생성하고 다음 실행부터 유효한 파일을 재사용한다.
`cache_text_encoder_outputs_to_disk=true`는 텍스트 출력 캐시를 켜는 효과도 있다.
`cache_latents_to_disk=true`도 latent 캐시를 함께 활성화한다.

| 데이터 처리 | 텍스트 출력 캐시 |
| --- | --- |
| 고정 TXT·문장 JSON | 사용 가능 |
| 여러 고정 캡션 후보 | 모든 후보를 저장하므로 사용 가능 |
| deterministic layout | 사용 가능 |
| `caption_dropout_rate` | 캐시한 빈 프롬프트 출력으로 대체하므로 사용 가능 |
| 태그 셔플·wildcard·token warmup·태그 dropout·epoch 단위 dropout | 끈다. |
| layout의 random crop·flip | 좌표와 프롬프트가 바뀌므로 끈다. |

동적 캡션을 사용할 때는 **두 항목 모두** 끈다.

```toml
cache_text_encoder_outputs = false
cache_text_encoder_outputs_to_disk = false
```

이미지의 random crop이나 color augmentation 등은 별도의 latent 캐시 제약도 따른다.
설정 검증에서 캐시 불가 오류가 나면 해당 변환을 끄거나 관련 캐시를 끈다.

### 캐시 검증과 파일

텍스트 캐시는 `001_moss_te.npz`처럼 저장한다. latent 파일명은 원본 크기를 포함한
`001_1024x1024_moss.npz` 같은 형식이다. 텍스트 파일에는 후보별 출력과 빈 프롬프트 출력을
저장한다. 현재 캐시 버전은 2다.

문장·후보·prefix/suffix·선택 길이·layout geometry·토크나이저/인코더 설정을 확인한다.
latent 캐시는 VAE와 이미지 정보를 확인한다. 변경을 감지하면 다시 계산한다.
여러 후보가 없어도 모델·설정 fingerprint가 맞는 기존 버전 2 캐시는 재사용할 수 있다.

이전 파일을 교체할 때는 `.legacy-v1.bak`으로 백업한다. 같은 이름이 있으면 `.1`, `.2`를 붙인다.
계산한 파일을 임시 경로에 쓴 뒤 교체하므로 기존 캐시를 먼저 지울 필요는 없다.
백업도 공간을 사용하므로 정상 학습과 새 캐시를 확인한 뒤 필요 없는 백업을 정리한다.

`skip_cache_check=true`이면 텍스트 캐시의 현재 캡션 비교와 expected fingerprint 비교를 건너뛴다.
버전·형식·배열 검사는 유지하며 latent 캐시는 자체 검사를 유지한다.
보통은 기본값 `false`로 두어 변경 감지를 유지한다.

Qwen weight는 BF16, rotary buffer는 FP32로 유지하고 VAE는 FP32로 처리한다.
최대 텍스트 길이는 chat template을 포함한 4096 토큰이다. 길이를 줄이면 사용량은 줄일 수 있지만
초과 문장이 잘리므로 캐시도 다시 확인해야 한다.

## 샘플 이미지와 학습 재개

### 학습 중 샘플 이미지

`train.toml`에 다음을 추가한다.

```toml
sample_prompts = "/workspace/data/sample_prompts.txt"
sample_every_n_steps = 100
```

해당 TXT 파일은 프롬프트 하나를 한 줄에 적는다.

```text
style_trigger, A girl with green hair waves. --w 1024 --h 1024 --s 30 --l 3 --d 42
```

`--w/--h`는 크기, `--s`는 steps, `--l`은 CFG, `--d`는 seed다.
크기는 양수인 16의 배수를 사용한다. `--fs`로 샘플 flow shift, `--n`으로 negative를 지정할 수 있다.
샘플은 `output_dir/sample/`에 저장한다.

샘플러의 기본 CFG는 3, flow shift는 5, negative 분기 skip layer는 22다.
호환 진입점은 기본 30 steps와 seed 42, 공식 진입점은 기본 50 steps를 사용한다.
초기 noise는 CPU generator에서 만들므로 이전 GPU generator와 같은 seed라도 이미지가 달라질 수 있다.

### 저장한 상태에서 재개

`save_state=true`로 저장한 `*-state` 폴더를 지정한다.

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  moss_train_network.py --config_file /workspace/data/train.toml \
  --resume /workspace/output/moss-style-000100-state
```

실제 만들어진 폴더명을 사용한다. `max_train_steps`는 추가 학습 횟수가 아니라 **전체 목표 step**이며
재개 시점보다 커야 한다. optimizer·scheduler·RNG 등을 복원하므로 관련 설정을 일관되게 유지한다.

adapter `.safetensors`만 이어서 읽으려면 `network_weights`를 사용한다.
이는 optimizer 상태를 복구하는 완전 재개와 다르다. 예기치 않은 종료 전에 state가 저장되지 않았다면
마지막으로 저장한 상태 이후의 학습까지 복원되지는 않는다.

## ComfyUI

1. 학습한 adapter `.safetensors`를 ComfyUI의 `models/loras/`에 넣는다.
2. [연결 예제 workflow](../examples/moss/comfyui_workflow.json)를 열고 모델·Qwen·VAE·adapter 파일명을 자신의 파일로 바꾼다.
3. `Load LoRA (Model Only)` 노드로 adapter를 적용하고 strength를 조절한다.

TXT 학습이면 기존 트리거·태그·자연어 문장을 positive에 넣는다.
문장 JSON이면 선택했던 `caption.long` 등의 **문장만** 넣는다.
layout이면 생성 크기의 좌표에 맞춘 `summary/items/bbox` **JSON 전체**를 넣는다.

workflow는 기본 CFG/Euler 연결 예제이며 공식 negative 분기 block skip 전체를 구현한 예제는 아니다.
학습과 추론에 같은 베이스를 사용한다. Base에서 학습한 adapter를 SFT에 적용한 품질은 확인하지 않았다.

ComfyUI 쪽에도 MOSS, adapter 이름, fused QKV 부분 적용, LoKR/DoRA 처리가 맞아야 한다.
이 sd-scripts 저장소를 갱신하는 것만으로 별도 ComfyUI 코드가 갱신되지는 않는다.
이 포크는 기존 portable tensor key를 사용하며 독자적인 추론용 key를 추가하지 않는다.
MOSS edit LoRA의 reference conditioning은 이 문서의 T2I 학습 범위에 포함하지 않는다.

## 문제 해결

| 증상 | 확인할 내용 |
| --- | --- |
| `No module named lycoris` | accelerate와 Python이 같은 가상환경인지 확인하고 로컬 LyCORIS 포크를 설치한다. |
| `Qwen3_5TextModel` 또는 `AutoencoderKLFlux2` import 실패 | 사용하는 환경에 `requirements.txt`의 라이브러리 버전을 설치했는지 확인한다. |
| `Specify the MOSS ... checkpoint` | 모델 루트가 완전한지, 개별 파일 사용 시 `text_encoder`·`vae`를 지정했는지 확인한다. |
| 학습 이미지가 0개 | `train_data_dir`가 반복 횟수 폴더의 상위 폴더인지, 이미지 경로와 확장자가 맞는지 확인한다. |
| JSON 캡션 오류 | UTF-8, 선택한 `caption.long/medium/short`, layout 필수 필드와 좌표 순서를 확인한다. |
| 캐시 불가 오류 | 동적 캡션·layout 증강과 텍스트 캐시 조합을 확인한다. 디스크 캐시 옵션도 함께 꺼야 한다. |
| CUDA 메모리 부족 | batch 1, 캐시, checkpointing, swap 수, 해상도, 텍스트 길이와 사용 중인 다른 프로세스를 확인한다. |
| H2D-only가 동작하지 않음 | CUDA, 양수 `blocks_to_swap`, `gradient_checkpointing=true`, 최신 포크 코드인지 확인한다. |
| `unrecognized arguments` | 다른 모델·원본 최신 dev의 옵션을 섞었는지 확인하고 현재 파일의 `--help`를 확인한다. |
| 캡션을 바꿨는데 출력이 그대로임 | `skip_cache_check=false`인지 확인하고 모델·토크나이저·데이터 경로가 같은지 확인한다. |

## 검증 범위

현재 선별 이식 코드는 Linux의 RTX 4070에서 확인했다.

| 검사 | 확인한 내용 |
| --- | --- |
| CPU 선택 검사 | 109 passed. 캡션 처리, 다중 후보 캐시, fingerprint·백업, 축소 모델·트레이너, 공통 offloader |
| CUDA 선택 검사 | 40 passed. H2D ring/pinned 설정, 출력·gradient, 두 LyCORIS 프리셋의 LoKR+DoRA, 축소 BF16 저장·재개 |
| 실제 작업 경로 재확인 | 캐시와 CUDA LoKR+DoRA block swap 21 passed |

세 선택은 일부 겹치므로 고유 테스트 수로 합산하지 않는다. 전체 5B/9B 가중치의 장시간 학습,
화질 비교, 속도·최대 메모리 수치, Windows/MPS는 이번 검사에서 확인하지 않았다.
현재 기존 가상환경의 Qwen/VAE 실제 클래스 검사는 의존성 버전 제약으로 제외했다.
이전 통합에서 수행한 별도 환경의 검사와 이번 실행 결과는
[기존 통합 기록](../.ai/worklogs/2026-10-18-official-moss-dev-integration.md),
[이번 선별 이식 기록](../.ai/worklogs/2026-10-25-selected-moss-upstream.md)에 구분해 남겼다.

## 다른 모델과 상세 문서

| 목적 | 문서 |
| --- | --- |
| 공식 MOSS generic adapter·Qwen adapter | [공식 진입점 가이드](moss_image_train_network.md) |
| block swap 구조와 옵션 | [Block Swap](block_swap.md) |
| 일반 network 학습 | [Network training](train_network.md) |
| SDXL | [SDXL network training](sdxl_train_network.md) |
| FLUX.1 | [FLUX network training](flux_train_network.md) |
| Anima | [Anima network training](anima_train_network.md) |
| dataset TOML의 세부 옵션 | [Dataset config](config_README-en.md) |
| MOSS 추론 파라미터·레이아웃 | [MOSS 공식 사용자 가이드](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) |
