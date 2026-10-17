# MOSS Image 어댑터 학습과 ComfyUI 사용

공식 MOSS Image Base/SFT 모델 디렉터리에서 표준 LoRA 또는 LyCORIS LoKR를 학습한다.
LoKR는 full-matrix와 DoRA 조합을 지원한다. Qwen3.5 텍스트 인코더와 FLUX.2 VAE는 고정한다.
기존 Anima 데이터셋의 `트리거, 캐릭터, 태그, 자연어 설명` TXT를 재작성할 필요가 없다.

## 준비

기존 환경을 변경하지 않으려면 별도 가상환경에 이 저장소의 `requirements.txt`를 설치한 뒤
MOSS 프리셋이 들어 있는 **로컬 LyCORIS 포크**를 설치한다. PyTorch/CUDA는 장치에 맞는 버전을 사용한다.

```sh
python -m pip install -r requirements.txt
python -m pip install /path/to/LyCORIS
```

기존 sd-scripts 환경에 이전 버전이 남아 있으면 requirements 설치가 필요하다. 검증에는
PyTorch 2.14.1, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0을 사용했다.
모델 루트에는 `transformer/`, `text_encoder/`, `vae/`, `scheduler/`와 tokenizer 파일이 있어야 한다.
tokenizer는 `tokenizer/`를 우선하고, 별도 폴더가 없으면 `text_encoder/`에서 읽는다.
모든 모델 로딩은 `local_files_only=True`다. 모델 자동 다운로드는 하지 않는다.
ComfyUI용 분할·양자화 가중치를 직접 학습하는 로더는 포함하지 않는다.

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
태그 셔플, dropout, wildcard가 필요하면 텍스트 출력 캐시를 끈다. 레이아웃의 random crop/flip도
고정 텍스트 캐시와 함께 사용할 수 없다. random crop/color augmentation은 기존 latent 캐시 제약도 따른다.
레이아웃에는 일반 bucket/random crop을 사용하고 face crop은 사용하지 않는다.

텍스트 캐시는 문장·선택 길이·레이아웃 변환·tokenizer/인코더 설정을 검사한다.
latent 캐시는 VAE 구성과 원본 이미지의 변경을 검사한다. MOSS 캐시의 식별 검사는
`skip_cache_check`와 관계없이 수행하며, 기존 다른 모델의 캐시 동작은 유지한다.

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
block swapping의 microbatch는1로 설정하고 실효 batch는 `--gradient_accumulation_steps`로 늘린다.
`--cpu_offload_checkpointing`과 block swapping은 함께 사용할 수 없다.
`--moss_max_token_length` 기본값은4096이다. Qwen의 각 forward에서는 padding을 제거한다.

`--resume output/<이름>-<step>-state`는 optimizer/RNG 등을 포함한 native 학습 상태에서 재개한다.
portable `.safetensors`를 `--network_weights`로 읽으면 optimizer 상태까지 복원하는 완전 재개가 되지 않는다.
전체 모델 미세조정, Qwen adapter 학습, FP8/양자화 모델 학습, DDPM/SNR loss 설정은 지원 범위에 포함하지 않는다.

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

## 검증 범위

축소 Transformer의 CPU FP32/CUDA BF16에서 학습·gradient·portable 저장 복원·native 재개를 검증했다.
TXT/학습 JSON/레이아웃 JSON 실제 데이터셋은 축소 모델과 가짜 Qwen/VAE로 학습 루프를 확인했다.
공식 소스와 축소 모델의 forward·전체 parameter gradient, scheduler도 일치했다.
전체5B/Qwen/VAE 가중치로 수행하는 학습·이미지 생성·화질 평가·장기 실행·Windows/MPS는 미검증이다.

참고: [공식 사용자 가이드](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md),
[공식 파이프라인](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/pipeline_moss_image.py).
