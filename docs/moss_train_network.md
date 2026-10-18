# MOSS Image 어댑터 학습과 ComfyUI 사용

공식 sd-scripts `dev`의 `6a329018`과 stage 3 trainer 커밋 `cb5c24f`를 기반으로 표준 LoRA 또는 LyCORIS LoKR를 학습한다.
기존 `moss_train_network.py`는 공식 `moss_image_train_network.py`의 호환 진입점이다.
LoKR는 full-matrix와 DoRA 조합을 지원한다. Qwen3.5 텍스트 인코더와 FLUX.2 VAE는 고정한다.
기존 Anima 데이터셋의 `트리거, 캐릭터, 태그, 자연어 설명` TXT를 재작성할 필요가 없다.

## 준비

기존 환경을 변경하지 않으려면 별도 가상환경에 이 저장소의 `requirements.txt`를 설치한 뒤
MOSS 프리셋이 들어 있는 **로컬 LyCORIS 포크**를 설치한다. PyTorch/CUDA는 장치에 맞는 버전을 사용한다.

```sh
python -m pip install -r requirements.txt
python -m pip install /path/to/LyCORIS
```

기존 sd-scripts 환경에 이전 버전이 남아 있으면 requirements 설치가 필요하다.
이번 검증에는 PyTorch 2.13.0+cu130, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0을 사용했다.
기존 모델 루트의 `transformer/`, `text_encoder/`, `vae/`를 사용할 수 있고 `scheduler/`가 없으면
공식 기본값(학습 shift 1, 샘플 shift 5, 1000 timesteps)을 사용한다.
tokenizer는 `tokenizer/`를 우선하고, 별도 폴더가 없으면 `text_encoder/`에서 읽는다.
모델 가중치는 로컬 파일만 읽는다. 별도 tokenizer 경로가 없으면 공식 tokenizer를 Hugging Face 캐시에서
찾거나 내려받는다. 오프라인 실행은 `--tokenizer_cache_dir /path/to/tokenizer`와
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 사용한다.

공식 첫 shard와 **실수형 ComfyUI 단일 파일**도 지원한다. 각각 `--pretrained_model_name_or_path`,
`--text_encoder`, `--vae`에 지정한다. 명시한 TE/VAE 경로가 모델 루트 자동 탐색보다 우선한다.
DiT의 fused QKV는 로드할 때 Q/K/V로 분리하고, Qwen은 vision tower를 제외한 text model만 읽는다.
FP8·INT8 등 양자화 가중치는 이번 학습 범위에 포함하지 않는다.

## TXT와 JSON 데이터셋

`examples/moss/dataset.toml`의 경로를 수정한다. 필요한 subset만 남겨 사용한다.
이미지와 캡션은 같은 basename을 사용한다. `.txt`와 `.json`은 subset별 `caption_extension`으로 선택한다.

| 설정 | 의미 |
|---|---|
| `caption_format="auto"` | MOSS 학습 기본값. 일반 TXT와 두 JSON 구조를 구분한다. |
| `caption_format="text"` | 기존 sd-scripts TXT 처리. 다른 학습기의 기본 동작도 유지된다. |
| `caption_format="moss_caption"` | `caption.long/medium/short` 객체에서 선택한 문장을 사용한다. |
| `caption_format="moss_layout"` | `summary/items/bbox` 구조를 레이아웃 프롬프트로 사용한다. |
| `moss_caption_length="long"` | 기본 `long`; `medium`, `short`도 선택 가능하다. |

학습 JSON의 `watermark`, `borders`는 프롬프트에 넣지 않는다. 선택한 필드는 비어 있으면 안 되며
의도한 문장 내부 줄바꿈은 보존한다. JSON 예제는 `examples/moss/caption.json`에 있다.

레이아웃 예제 `examples/moss/layout.json`은 **1024×1024 원본 이미지**를 기준으로 한다.
박스는 `[y_min,x_min,y_max,x_max]`이며 원본 이미지 중심을 원점으로 하는 픽셀 좌표다.
실제 이미지의 resize/crop/flip과 함께 좌표도 바뀐다. 잘린 박스는 경계에 맞추고 완전히 사라진 항목은 제거한다.
레이아웃에 태그 셔플·와일드카드 치환을 적용하지 않으며 prefix/suffix는 `summary.caption` 안에 들어간다.

`shuffle_caption=false`가 기본이므로 기존 TXT의 트리거·태그·자연어 순서를 유지한다.
태그 셔플, wildcard, epoch 단위 dropout에는 텍스트 출력 캐시를 끈다.
`caption_dropout_rate`는 캐시된 빈 프롬프트 임베딩으로 대체하므로 캐시와 함께 사용할 수 있다.
validation에는 이 dropout을 적용하지 않는다. 레이아웃의 random crop/flip도
고정 텍스트 캐시와 함께 사용할 수 없다. random crop/color augmentation은 기존 latent 캐시 제약도 따른다.
레이아웃에는 일반 bucket/random crop을 사용하고 face crop은 사용하지 않는다.

텍스트 캐시는 문장·선택 길이·레이아웃 변환·tokenizer/인코더 설정을 검사한다.
latent 캐시는 VAE 구성과 원본 이미지의 변경을 검사한다. MOSS 캐시의 식별 검사는
`skip_cache_check`와 관계없이 수행하며, 기존 다른 모델의 캐시 동작은 유지한다.

새 캐시 버전은 **2**다. 텍스트 파일은 공식 BF16 비트 형식의 conditional/unconditional 임베딩을 저장하고,
latent 파일은 32채널 raw VAE latent를 저장한다. packing과 BatchNorm 정규화는 trainer에서 수행한다.
기존 캐시가 같은 이름으로 있으면 재계산 후 설치하기 전에 `.legacy-v1.bak`으로 보존한다.
같은 백업 이름이 있으면 `.1`, `.2`를 붙인다. 버전·스키마·식별 정보가 맞는 캐시는 재사용한다.
VAE는 기존 posterior sampling에서 공식 mean 인코딩으로 전환했으므로 latent 캐시는 다시 계산한다.
Qwen 가중치는 DiT의 mixed precision 설정과 관계없이 BF16으로 인코딩하고 rotary buffer는 FP32로 유지한다.
캐시 도구도 같은 정밀도를 사용한다. 이전 인코딩 fingerprint의 텍스트 캐시는 백업 후 다시 생성한다.

## 학습

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 moss_train_network.py \
  --pretrained_model_name_or_path /path/to/MOSS-Image-Base \
  --dataset_config examples/moss/dataset.toml \
  --network_module lycoris.kohya --network_dim 16 --network_alpha 16 \
  --network_args algo=lokr preset=moss full_matrix=True factor=4 weight_decompose=True \
  --mixed_precision bf16 --gradient_checkpointing --blocks_to_swap 20 \
  --cache_latents_to_disk --cache_text_encoder_outputs_to_disk \
  --moss_text_encoder_device cpu --max_data_loader_n_workers 0 \
  --optimizer_type AdamW --learning_rate 1e-4 --max_train_steps 1000 \
  --output_dir output --output_name moss-style --save_model_as safetensors \
  --save_every_n_steps 100 --save_state
```

위 명령은 연결 예제이며 화질이나 VRAM 사용량을 조정한 권장 레시피는 아니다.
`preset=moss`는 DiT 블록의252개 Linear, `preset=moss-full`은 Transformer 내부269개 Linear를 대상으로 한다.
`moss-full`의 추가 text adapter는 Transformer 내부 계층이며 Qwen 자체를 학습하는 설정이 아니다.
표준 LoRA는 `--network_args algo=lora preset=moss`로 변경한다.
학습·ComfyUI 호환성 검증은 표준 LoRA, LoKR, LoKR+DoRA를 대상으로 한다. LoRA+DoRA 동작은 이번 검증 범위에 포함하지 않는다.

Qwen은 기본적으로 CPU에서 텍스트를 사전 계산하며, 캐시 후 DiT를 읽기 전에 해제한다.
VRAM이 충분하면 `--moss_text_encoder_device accelerator`를 선택할 수 있다.
train/validation dataset과 sample prompt를 모두 처리한 뒤 한 번 해제하며, 보관할 prompt 임베딩은 CPU에 둔다.
block swapping의 microbatch는1로 설정하고 실효 batch는 `--gradient_accumulation_steps`로 늘린다.
`--cpu_offload_checkpointing`과 block swapping은 함께 사용할 수 없다.
`--moss_max_token_length` 기본값은4096이다. Qwen의 각 forward에서는 padding을 제거한다.
`--attn_mode torch/flash/sageattn/xformers`와 `--split_attn`을 사용할 수 있다.
추가 attention backend는 해당 패키지가 설치된 환경에서 사용한다. 이번 CUDA 검증은 torch attention을 사용했다.

기존 명령의 `--timestep_sampling` 기본값은 `legacy`다. 기존 이산 uniform grid와 모델 루트의
`scheduler_config.json`을 유지한다. 다음 옵션을 명시하면 공식 sampling 경로를 사용한다.

```sh
--timestep_sampling sigma --weighting_scheme logit_normal \
--logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0
```

위 값은 upstream PR의 분포 설정이며 이번 검증에서 화질을 평가한 레시피는 아니다.
`uniform`, `sigmoid`, `shift`, `flux_shift`도 선택할 수 있다. `--show_timesteps console`로
가중치 로드 없이 분포를 확인한다. 새 공식 진입점의 기본값은 `sigmoid`다.

`--compile`은 블록별 compile을 활성화하며 `--torch_compile`과 함께 사용할 수 없다.
`--compile_fullgraph`는 `--split_attn`과 함께 사용할 수 없다.
`--unsloth_offload_checkpointing`은 gradient checkpointing을 활성화하고,
CPU offload나 block swap과 함께 사용할 수 없다. CPU offload는 저장할 입력을 CPU에 보관하며
계산 출력은 모델 device에 유지한다.

두 진입점은 같은 inference sampler를 사용한다. 공식 진입점의 기본 sampling은 50 steps, CFG 3,
shift 5, unconditional block skip 22다. 기존 명령은 30 steps와 기본 seed 42를 유지한다.
초기 noise를 CPU generator에서 생성하므로 이전 GPU generator와 같은 seed의 이미지가 같지는 않을 수 있다.
sampling이 실패해도 RNG, VAE 위치, block swap 학습 모드를 복원한다.

공식 generic LoRA/LoHa/LoKr 경로와 opt-in TE adapter는
[공식 진입점 가이드](moss_image_train_network.md)를 참고한다.
공식 `train_text_embedder=true` 범위는 263개로, `moss-full` 269개와 다르다.

단일 파일 입력은 위 명령의 모델 경로 부분을 다음과 같이 바꾼다.

```sh
--pretrained_model_name_or_path /models/moss_image_base_bf16.safetensors \
--text_encoder /models/qwen3.5_9b_bf16.safetensors \
--vae /models/flux2-vae.safetensors \
--tokenizer_cache_dir /models/moss-tokenizer
```

별도 캐시 도구와 trainer는 같은 MOSS 전처리·형식·식별 정보를 사용한다.
같은 tokenizer·정밀도·최대 길이·dataset 설정을 지정해야 재사용된다.

```sh
python tools/cache_latents.py --moss_image \
  --pretrained_model_name_or_path /path/to/MOSS-Image-Base \
  --dataset_config examples/moss/dataset.toml --mixed_precision bf16
python tools/cache_text_encoder_outputs.py --moss_image \
  --pretrained_model_name_or_path /path/to/MOSS-Image-Base \
  --dataset_config examples/moss/dataset.toml --mixed_precision bf16 \
  --cache_text_encoder_outputs_to_disk --moss_max_token_length 4096
```

`--resume output/<이름>-<step>-state`는 optimizer/RNG 등을 포함한 native 학습 상태에서 재개한다.
portable `.safetensors`를 `--network_weights`로 읽으면 optimizer 상태까지 복원하는 완전 재개가 되지 않는다.
이전 trainer의 adapter와 native 상태도 읽는다. 다만 공식 VAE mean 인코딩·BF16 텍스트 캐시·batched DiT로
전환하면서 이후 학습 궤적은 달라질 수 있다.
기존 LyCORIS 명령은 Qwen adapter 학습을 지원하지 않는다. 전체 모델 미세조정, FP8/양자화 모델 학습,
DDPM/SNR loss 설정은 두 명령의 지원 범위에 포함하지 않는다.

## ComfyUI

1. 학습한 `.safetensors`를 ComfyUI의 `models/loras/`에 넣는다.
2. `examples/moss/comfyui_workflow.json`을 읽고 모델·Qwen·VAE·adapter 파일명을 선택한다.
3. 기존 `Load LoRA (Model Only)` 노드로 적용하고 strength를 조절한다.

TXT로 학습했다면 positive에 기존 트리거·태그·자연문을 그대로 입력한다.
학습 캡션 JSON은 사용한 `caption.long` 등의 **문장만** positive에 넣는다.
레이아웃은 생성 크기의 좌표에 맞춘 `summary/items/bbox` **JSON 전체**를 positive에 넣는다.
기존 `MOSS-Image Layout Prompt` 노드로 레이아웃을 만들 수도 있다.

workflow는 기본 CFG/Euler 경로를 쓰는 연결 예제다. 공식 가이드의 negative 분기 block skip까지 구현한 예제는 아니다.
학습과 추론에 같은 베이스를 사용한다. Base에서 학습한 adapter를 SFT에 적용했을 때의 화질은 검증하지 않았다.

ComfyUI에는 MOSS용 generic `lycoris_` 이름과 fused QKV 부분 적용, 공통 DoRA 출력축 방향 norm 수정을 반영했다.
기존 `lora_unet_` 이름도 사용한다. 독자적인 portable tensor key는 추가하지 않는다.
ai-toolkit의 실험적 MOSS edit LoRA는 reference conditioning과 별도
[ComfyUI 노드](https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit)를 사용한다.
이 문서의 T2I 학습·workflow에 edit 지원을 추가한 것은 아니다.

## 검증 범위

축소 Transformer의 CPU FP32/CUDA BF16에서 학습·gradient·portable 저장 복원·native 재개를 검증했다.
TXT/학습 JSON/레이아웃 JSON 실제 데이터셋은 축소 모델과 가짜 Qwen/VAE로 학습 루프를 확인했다.
공식 소스와 축소 모델의 forward·전체 parameter gradient, scheduler도 일치했다.
공식 shard/ComfyUI DiT 로더와 실제 축소 Qwen3.5 text model 로더를 검증했다.
새/기존 텍스트 캐시, 여러 dataset과 sample prompt 처리 후 TE를 CPU로 이동하고 해제한 뒤 DiT를 읽는지 확인했다.
실제 ComfyUI 로더의 두 prefix·QKV slice·LoKR/DoRA 적용도 검사했다.
새 trainer에서 복수 validation timestep의 캐시 재사용과 학습 전용 dropout을 확인했다.
두 진입점의 축소 CUDA BF16 학습·저장·재개, CPU/unsloth offload의 출력·전체 parameter gradient를 검사했다.
per-block compile은 CPU `eager` backend의 forward/backward를 검증했다. CUDA Inductor의 성능은 미검증이다.
전체5B/Qwen/VAE 가중치로 수행하는 학습·이미지 생성·화질 평가·장기 실행·Windows/MPS는 미검증이다.

참고: [공식 사용자 가이드](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md),
[공식 파이프라인](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/pipeline_moss_image.py).
