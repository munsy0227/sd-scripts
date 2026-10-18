# 공식 MOSS stage 3 trainer 통합

후속: 최신 공식 dev 병합과 커밋/푸시 승인 작업은 [게시 기록](2026-10-18-moss-dev-publication.md)에 정리했다. 아래 Git 상태는 최초 통합 시점의 기록이다.

## 요청과 기준

- 사용자의 새 trainer 브랜치 검토 후 “그럼 해보자” 승인에 따른 구현.
- 공식 기준: `cb5c24fc724f758fcb1e63969872f2f561ac1396`.
- 기반 dev: `3e84e58f647a4246892d7857186715c8e0b6c408`.
- 작업 브랜치: `codex/sd-scripts-moss-dev-integration`.
- 이전 dev 통합은 미커밋 merge 상태였다. 그 index 전체를 먼저 백업한 뒤 stage 3의 전체 13개 파일 diff를 `git apply --3way --index`로 적용했다. `library/moss_image_train_utils.py` 충돌을 양쪽 기능을 보존하며 해결했다.
- HEAD와 MERGE_HEAD는 기존 값 그대로다. stage 3 내용은 작업본/index에 반영했으며, cb5c24f가 Git 조상으로 기록되는 별도 merge commit은 만들지 않았다. commit/push/main 반영 없음.

## 복구 자료

원래 51개 staged 파일의 binary/full-index 패치와 Git 상태를 다음에 보존했다.

```text
/home/munsy0227/LyCORIS/.internal/backups/sd-scripts-pre-stage3-20261018.patch
/home/munsy0227/LyCORIS/.internal/backups/sd-scripts-pre-stage3-20261018.json
```

패치는 358,312 bytes다. 기존 HEAD `2de3c83f925a06f5f14f4939633bff1be807ec18`, MERGE_HEAD `3e84e58f647a4246892d7857186715c8e0b6c408`와 staged 파일 수를 JSON에 기록했다. 기존 backup branch/bundle과 이전 native fixture도 보존했다.

## 구현

- `moss_image_train_network.py`를 공식 기반 구현으로 채택하고, `moss_train_network.py`는 이를 상속하는 LyCORIS 호환 진입점으로 축소했다.
- 공식 generic LoRA, MOSS ArchConfig, TE adapter merge hook, 공통 sampler, compile/unsloth 옵션, SAI metadata 및 공용 compile 인자 추출을 포함했다.
- 기존 모델 root/명시 component 우선순위/공식 shard/Comfy DiT/로컬 tokenizer/offline, TXT/JSON/layout, cache 식별과 백업, LyCORIS preset 및 optimizer/scheduler/global step 복원을 보존했다.
- 기존 명령의 기본 timestep sampling은 `legacy`다. 이전 이산 uniform grid, root scheduler 설정과 RNG draw 순서를 유지한다. 공식 진입점 기본값은 `sigmoid`이며 새 분포는 두 명령에서 명시적으로 선택할 수 있다.
- Qwen은 두 진입점 모두 기본 frozen/CPU다. 공식 진입점의 TE adapter는 `--moss_train_text_encoder --moss_text_encoder_device accelerator`로 opt-in하며 text-output cache와 함께 사용할 수 없다. LyCORIS MOSS preset은 TE를 계속 고정한다.
- TE와 캐시 도구는 BF16 weight/FP32 rotary buffer를 사용한다. encoding에서 외부 DiT autocast를 차단했다. cache NPZ schema는 v2를 유지하며 encoding fingerprint를 `qwen35-text-only-unpadded-last-hidden-v3-bf16`으로 바꾸어 이전 precision/cache를 백업 후 재생성한다.
- sample prompt와 unconditional embedding은 CPU에 보관한다. train/validation group 처리 후 lazy DiT 경계에서 한 번 TE를 CPU로 옮기고 참조를 해제한다. prompt encoding도 frozen TE에서는 중복하지 않는다.
- 공식 `process_batch`의 cache 변형을 제거했다. 다섯 필드 원본 cache를 유지하고 noise prediction 단계에서 학습인 경우에만 dropout을 적용한다. 복수 validation timestep 재사용에서 conditional embedding을 보존한다.
- 공식 sampling과 minimal inference가 `generate_latents`를 공유한다. sampling 성공/실패 시 RNG, VAE 위치, model/TE training 상태와 block swap을 복원한다. 기존 명령의 sample 기본값 30 steps/seed 42는 유지한다. 초기 noise의 CPU generator 사용으로 과거 GPU-generator sample과는 동일 seed의 결과가 달라질 수 있다.
- 공식 기본 범위 252개는 `moss`와 동일하다. `train_text_embedder=true`는 263개이며 `moss-full` 269개와 다르다. LyCORIS 수학 및 scope를 변경하지 않았다.

## 실행 중 발견한 추가 오류

CUDA BF16에서 `--cpu_offload_checkpointing`의 원래 구현은 checkpoint 출력까지 CPU로 옮겼다. 다음 text refiner는 CPU 입력을 보고 CPU를 계산 device로 선택하지만 weight는 CUDA에 남아 device mismatch가 발생했다. main-block 이후 residual/final projection에도 같은 문제가 생길 수 있었다.

`library/moss_image_models.py`에서 non-reentrant checkpoint의 저장 입력만 `torch.autograd.graph.save_on_cpu`로 offload하고, live output은 model device에 유지하도록 고쳤다. CPU pinned memory는 CUDA 입력에서만 사용한다. CPU offload/unsloth 두 경로의 실제 CUDA BF16 출력, input gradient, 전체 parameter gradient가 일반 경로와 일치했다.

## 검증 환경

기존 `.venv`를 변경하지 않고 `/tmp/moss-stage3-venv`에 요구 버전을 설치했다. 기존 site-packages는 torch 등 재사용을 위한 읽기 경로로만 참조했다. OMP/MKL/OpenBLAS 스레드를 1개로 제한했고, 마무리 CLI와 하위 프로세스 검사에는 NumExpr 제한도 명시했다. Hugging Face offline 모드를 사용했다.

| 항목 | 값 |
| --- | --- |
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu130 |
| transformers / diffusers | 5.17.0 / 0.40.0 |
| accelerate / safetensors | 1.15.0 / 0.8.0 |
| GPU | NVIDIA GeForce RTX 4070, CUDA 접근 가능 |

## 검사 결과

여러 단계에서 다음 선택을 실행했다. 아래 숫자를 중복 없는 전체 테스트 수로 합산하지 않는다.

| 선택 | 결과 |
| --- | --- |
| 공식 trainer + generic LoRA + 기존 MOSS training | 51 passed. TXT/JSON/layout 루프, cache 도구 재사용, native/portable 복원, TE release 포함 |
| 새 통합 회귀 | 초기 10 passed: 반복 validation, legacy RNG, opt-in/cache gate, 252/263/269 scope, CPU compile, sampling 실패 복귀, CUDA offload |
| 두 진입점 CUDA BF16 실제 축소 학습·저장·step 1→2 재개 | 별도 프로세스 2 passed. 가짜 TE/VAE + 실제 DiT/dataset/optimizer/trainer |
| 공식 모델/전략/캡션/cache/LR/offload 묶음 | 96 passed, 51 skipped. 이때 49개 ComfyUI opt-in과 XPU/MPS 2개가 skip |
| 실제 로컬 ComfyUI 경로를 지정한 별도 실행 | 49 passed. prefix/QKV slice/LoKR/DoRA/multiplier |
| LyCORIS MOSS 및 Kohya 검사 | unittest 22 tests OK, optional 3 skipped. sd-scripts 모델 matrix의 24 subtests 포함 |
| 확장한 실제 축소 Qwen 로더 | CPU FP32/CPU BF16/CUDA BF16 3 passed. 외부 FP16 autocast에서도 dtype/hidden state 일치 |
| 마지막 encoding 보완의 영향 범위 | Qwen/generic/TE-release/cache-tool 재사용 선택 17 passed |

51개 선택 실행에서 `MOSS_LEGACY_STATE=/tmp/moss-dev-legacy-fixture/output/moss-step00000002-state`를 설정했다. 이전 trainer `2de3c83`가 만든 상태를 새 호환 trainer에서 step 2→3으로 재개하고, 이전 portable adapter도 로드했다.

추가 확인:

- 두 trainer, minimal inference, 두 cache 도구의 `--help` 성공.
- 두 진입점의 `--show_timesteps console --show_timesteps_resolution 32` 성공. 공식 경로는 sigma/logit-normal 옵션을 함께 지정했다.
- MOSS 변경 파일의 Ruff F 및 format, Python AST, whitespace/Git conflict 검사를 수행했다.
- 공용 Anima 유틸리티의 기존 F541은 별도 기존 진단으로 구분한다.
- 공식 `lora_utils.py`의 CRLF를 LF로 정규화했다. docstring을 제외한 실행 AST는 upstream stage 3과 동일하다.

최종 확인: staged 64개 경로(삭제 포함), 그중 존재하는 Python 소스 50개 AST 정상. MOSS 변경 파일 Ruff F 정상 및 9개 format 정상. 양 저장소 diff whitespace 정상, sd-scripts unresolved/unstaged/untracked 없음. LyCORIS core 변경 없음.

## 제한과 후속

실제 pretrained 5B DiT/9B Qwen/VAE 전체 학습이나 이미지 품질·장기 실행·성능 평가는 하지 않았다. compile 검사는 CPU `eager` backend이므로 CUDA Inductor/Triton 최적화 성능을 검증한 것이 아니다. real 9B TE LoRA, TE adapter의 ComfyUI 적용, Flash/Sage/xformers, Windows/MPS, 양자화/edit/full-model 학습도 미검증 또는 후속 범위다.

검증 후 이번 임시 환경/다운로드 cache와 stage 3 임시 패치를 정리했다. 기존 `.venv`, 이전 native fixture 및 복구 패치는 보존했다. 사용자 작성 `artifacts/`와 별도 autostop worklog는 변경하지 않았다.
