# 공식 dev 전체 통합과 MOSS trainer 재구성

## 범위와 Git 상태

- 사용자가 공식 dev 전체 통합, 기존 TXT/JSON·레이아웃·LoKR/DoRA·ComfyUI 연결 보존을 승인했다. 새 양자화 backend와 edit 학습은 제외했다.
- 이전 HEAD: `2de3c83f925a06f5f14f4939633bff1be807ec18`.
- 백업: `codex/pre-sd-scripts-dev-20261018` → 위 HEAD.
- 작업 브랜치: `codex/sd-scripts-moss-dev-integration`.
- 검토한 공식 dev `3e84e58f647a4246892d7857186715c8e0b6c408` 전체를 `git merge --no-commit --no-ff`로 적용했다. `main`과 원격에는 반영하지 않았다.
- **머지 커밋 대기 상태**다. `MERGE_HEAD`는 위 공식 dev를 가리키며 새 커밋·푸시·이력 재작성은 하지 않았다.
- `dataset.py`, `dreambooth_dataset.py`, `subset.py`의 충돌은 공식 다중 행 캡션과 포크 JSON/레이아웃 처리를 결합해 해결했다. 기존 global step 재개 수정도 보존했다.

## 구현

- 공식 `moss_image_models.MossImage`, text-only Qwen 로더, `flux2_vae`, attention/checkpointing/block swap, 샘플 sigma/Euler를 trainer에 연결했다.
- 중복 구현인 `library/moss_models.py`, `library/strategy_moss.py`, `library/moss_train_utils.py`를 제거했다. 학습 flow batch와 scheduler 설정 읽기는 공식 training utility에 연결했다.
- 기존 `moss_train_network.py`, 모델 루트 입력과 TOML을 유지한다. 공식 첫 shard·실수형 ComfyUI DiT, `--text_encoder`·`--vae` 덮어쓰기와 로컬 tokenizer 경로를 지원한다.
- 공식 모델의 Tensor 반환을 사용한다. raw VAE latent를 trainer에서 packing/BatchNorm 정규화한다. VAE는 posterior sampling에서 공식 mean 인코딩으로 전환했다.
- 공식 conditional/unconditional 텍스트 캐시를 연결하고 caption dropout은 학습에만 적용한다. 확률적 JSON/layout 변환·후보 선택의 캐시 제약은 trainer와 사전 캐시 도구에서 공유한다.
- 캐시 버전 2와 캡션·tokenizer/TE/VAE·최대 길이·정밀도·이미지 정보를 기록한다. 기존 같은 이름의 캐시는 `.legacy-v1.bak`으로 보존하며 충돌 시 `.1`, `.2`를 붙인다. 다른 모델의 캐시 규약은 유지한다.
- 텍스트 사전 계산은 train/validation dataset과 sample prompts까지 마친 뒤 한 번 정리한다. DiT 로더 호출 전에 TE를 CPU로 이동하고 list/loop 변수 참조를 제거한다. sample/unconditional 임베딩은 CPU에 보관한다.
- LyCORIS 핵심 수학·프리셋·portable tensor key는 변경하지 않았다. generic/Kohya 두 API의 공식 모델 통합 검사를 추가했다.

## ai-toolkit 대조

검토 범위: `9a1828f20131f3dc4fac8bab75bcfdf53df96398` → `8cf43dd2cc1fe828884001aceeda044649d9abd0`의 5개 커밋.

- `8cf43dd`: 새 텍스트 캐시 생성 뒤 TE가 GPU에 남는 수정. 같은 수명주기 보장을 유지하고 fresh/reused cache·여러 dataset·sample prompts·GPU 해제 회귀 검사에 반영했다.
- `88360c6`, `7b94dc5`: edit reference attention과 별도 ComfyUI 노드 안내. T2I와 문서에서 구분하고 이식하지 않았다.
- `577004a`: ConvRot 구형 Triton libdevice 호환. 새 양자화 backend가 범위 밖이므로 이식하지 않았다.
- `9d408eb`: Flux.3 reference 처리이므로 MOSS에 이식하지 않았다.

## 검증 환경

- Linux, Python 3.12.15, PyTorch 2.13.0+cu130, transformers 5.17.0, diffusers 0.40.0, accelerate 1.15.0, safetensors 0.8.0.
- CUDA: NVIDIA GeForce RTX 4070, driver 615.78.08. GPU 검사는 sandbox 밖의 승인된 장치 접근으로 실행했다.
- 별도 `/tmp/moss-dev-integration-venv`에 요구 버전과 테스트 도구를 설치했다. 기존 sd-scripts `.venv`는 수정하지 않았다. PyTorch 등 기존 런타임은 읽기 전용 site-packages 경로로 재사용했다.
- 모든 실행의 OMP/MKL/OpenBLAS/NumExpr 스레드를 1개로 제한했고 Hugging Face는 오프라인으로 설정했다.
- ComfyUI 검사는 `/mnt/MSY_HDD/webui/ComfyUI`의 실제 코드와 맞는 임시 의존성(comfy-aimdo 0.5.6, comfy-kitchen 0.2.38, SciPy)을 사용했다. ComfyUI 소스는 수정하지 않았다.
- 검증 종료 후 임시 가상환경·다운로드 캐시·GPU smoke 산출물은 정리했다. 이전 trainer의 검토용 native/portable fixture는 `/tmp/moss-dev-legacy-fixture`에 남겼다. 재실행하려면 별도 검증 환경을 다시 준비한다.

## 통과 결과

아래 결과는 서로 겹치는 선택 실행이며 하나의 고유 테스트 총수로 합산하지 않는다.

| 선택 | 결과 |
| --- | --- |
| 공식 MOSS model/strategy, 포크 caption/training/ComfyUI, 공식 caption cache/multiline/custom LR | CPU 156 passed, 10 CUDA cases skipped |
| 공통 offloading·Lumina cache 회귀 | CPU 16 passed, 5 GPU cases skipped |
| sd-scripts tiny adapter·block swap·TE release 선택 | GPU 접근 환경 17 passed, skip 없음 |
| sampler → 학습 backward 복귀 | CPU/CUDA 2 passed |
| LyCORIS 두 API × 두 preset × LoRA/LoKR/LoKR+DoRA × CPU/CUDA | 2 tests, 24 subtests passed |
| 기존 LyCORIS MOSS 범위와 Kohya optimizer 회귀 | 17 passed, 3 optional model cases skipped, 10 subtests passed |

- 통합 전에 실제 이전 trainer(`2de3c83`)로 축소 모델의 native/portable 상태를 만들었다. 새 trainer에서 native step 2→3과 이전 portable 로딩을 검사했다. `MOSS_LEGACY_STATE`로 이전 native 폴더를 지정하면 TXT 루프 검사에 포함된다.
- 추가로 축소 공식 DiT + 가짜 Qwen/VAE + 실제 dataset/optimizer 루프에서 CUDA BF16 step 1→2 저장·재개를 확인했다.
- 실제 shard/ComfyUI DiT 로더를 축소 가중치로 실행해 복원한 parameter와 forward가 원본과 정확히 일치함을 확인했다.
- 실제 Qwen3.5의 linear/full attention을 가진 축소 text model을 로드했다. vision/LM-head 키를 제외하고 복원한 hidden state가 원본과 정확히 일치했다.
- 캐시 도구→trainer의 동일 파일 재사용, 구 캐시의 byte 단위 백업, 설정·원본 변경 무효화, validation dropout 제외를 검사했다.
- 실제 ComfyUI의 49개 호환 검사는 CPU 선택 156개에 포함한다. 두 prefix, 252/269개 대상, fused QKV slice, LoKR/DoRA와 multiplier를 검사했다.
- 첫 추가 API smoke의 작은 SGD 1-step은 BF16 출력에서 변화가 관측되지 않았다. 테스트를 AdamW 2-step으로 바꾸어 유한한 비영 gradient와 실제 출력을 확인했다. 라이브러리 수학이나 수치 허용오차는 변경하지 않았다.
- MOSS 변경 파일 및 새 LyCORIS 테스트의 Ruff F/format 검사, 변경 Python AST 검사와 `git diff --check`를 수행했다. 공통 파일의 기존 Ruff F 진단은 공식 dev 대비 증가하지 않았다.
- trainer, 공식 minimal inference, 캐시 CLI 두 개의 `--help`가 정상 설치 환경에서 통과했다.

CPU 주요 재현 선택:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/LyCORIS:/path/to/sd-scripts \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  MOSS_COMFYUI_PATH=/path/to/ComfyUI MOSS_LEGACY_STATE=/path/to/old-native-state \
  python -m pytest tests/test_moss_image_models.py tests/test_strategy_moss_image.py \
  tests/test_moss_caption.py tests/test_moss_training.py tests/test_moss_comfy.py \
  tests/test_caption_processing_for_cache.py tests/test_multiline_caption.py \
  tests/test_custom_lr_scheduler.py -q --tb=short --show-capture=no
```

## 한계와 후속 단계

- 5B DiT·Qwen9B·전체 VAE 체크포인트 학습, 실제 생성 화질·수렴·VRAM/속도 측정·장기 실행·Windows/MPS는 미검증이다.
- 공식 VAE mean, BF16 텍스트 저장과 batched DiT 전환으로 기존 학습과 이후 궤적은 달라질 수 있다. 상태 복원 호환과 bitwise 학습 궤적 동일성은 구분한다.
- torch attention을 검증했다. Flash/Sage/xformers 실행과 새 양자화 backend·MOSS edit 학습은 검증 범위 밖이다.
- 기존 `strategy_base.py`/einops escape SyntaxWarning, torch JIT deprecation 등은 실패와 구분했다.
- 사용자 검토 후 커밋·푸시·main 반영은 각각 별도 승인 단계다.
