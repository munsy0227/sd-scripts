# MOSS Image 학습·캡션·ComfyUI 연동

## 승인과 이력

- 사용자가 계획 전체의 구현을 승인했다. 로컬 이력 정리와 코드·문서·예제 작성까지 수행하며 새 커밋·푸시·원격 이력 변경은 하지 않는다.
- 기존 main `7eff3367d8cf757e1cb807a51945d3a07cf74085`를 `codex/sd-scripts-pre-moss-20261018`에 보존했다.
- 전체 refs bundle: `/home/munsy0227/LyCORIS/.internal/backups/sd-scripts-pre-moss-20261018.bundle`.
- `git bundle verify` 통과. SHA256: `b8333ac6b8f1b9a8c0c78bb62c47c78c83f73432364a588c7bc858d73e6159ad`.
- bundle을 `/tmp`의 별도 bare 저장소로 clone하고 `7eff336`의 코드·테스트·이전 기록이 복구되는 것도 확인했다.
- 로컬 main을 upstream/main `690ea7f96c23182352ec63def76d431c6120bd2f`로 되돌렸다. origin/main은 기존 커밋 그대로다.
- 제거한 Anima 변경은 Block/FinalLayer AdaLN에서 `use_fp32=False`일 때 `nullcontext()`로 호출자 BF16 autocast를 보존하던 수정이다. 관련 precision 테스트와 이전 작업 기록도 활성 브랜치에서 제거됐다. 이 BF16 해결책은 현재 main에 없다.
- 백업 브랜치나 bundle에서 이전 코드·테스트·기록을 복구할 수 있다. 새 MOSS 수정이 있는 작업 트리를 강제로 덮어쓰지 말고 별도 checkout에서 복구한다.

## 구현

- `moss_train_network.py`: 기존 NetworkTrainer의 optimizer, dataloader, 저장/재개와 연결. 공식 Base/SFT 디렉터리의 dense floating-point 가중치만 로컬에서 읽는다. Qwen/VAE는 고정한다.
- 강화한 native 재개 검사에서 공통 `train_network.py`가 복원한 global step을0으로 초기화하는 기존 문제를 발견했다. epoch 내 skip offset과 global step을 분리해 복원한 step을 유지하고 남은 progress 길이도 이에 맞췄다. 세 캡션 형식 모두 step1 state에서 재개해 새로운 step2 state가 생기는 것을 확인했다.
- `library/moss_models.py`: 공식 이름·연산을 유지한 Transformer. 비재진입 checkpointing, CPU activation offload, 기존 ModelOffloader 연결. block swapping은 microbatch 1에서 사용한다.
- `library/moss_train_utils.py`: 공식 training grid의 균등 sampling, `(1-sigma)z+sigma*noise`, 목표 `noise-z`, scheduler, FLUX.2 unpack/역정규화.
- `library/strategy_moss.py`: 공식 open-think chat template, 마지막 hidden state, 각 Qwen forward의 padding 제거. VAE latent 2x2 packing/BN 정규화. 모델별 latent/text cache와 fingerprint 확인.
- dataset 옵션 `caption_format=auto|text|moss_caption|moss_layout`, `moss_caption_length=long|medium|short`. MOSS 기본 auto/long, 기존 다른 학습기 기본 text.
- 원본 TXT는 재작성하지 않는다. JSON 전체를 읽어 선택한 caption만 추출하고 줄바꿈을 보존한다. watermark/borders는 제외한다.
- 레이아웃은 JSON 구조를 보존하고 실제 resize/crop/flip에 맞춰 bbox를 변환한다. 기존 SD crop metadata는 실제 random crop 위치와 다르므로 변환 값을 별도로 전달한다. 기존 helper 반환 형식은 유지한다.
- stochastic caption 또는 레이아웃 random crop/flip과 고정 TE cache 조합은 시작 전에 거절한다. 레이아웃에는 face crop을 사용하지 않는다.
- Qwen CPU 사전 계산이 기본이다. 캐시 후 DiT를 읽기 전에 encoder 참조를 해제한다. 기존 학습 환경을 업그레이드하지 않았다.
- 공식 파생 코드의 Apache-2.0 원문은 `LICENSE-MOSS-Image`에 보존했다.
- `docs/moss_train_network.md`, `examples/moss/`에 세 캡션 형식, dataset TOML, native ComfyUI workflow와 사용 명령을 추가했다.

## ComfyUI 변경

- 대상은 `/mnt/MSY_HDD/webui/ComfyUI`, 기존 HEAD `a639f187a`다. 기존 untracked 사용자 데이터는 유지했다.
- 기존 MOSS 구현과 표준 LoRA loader를 사용한다. generic `lycoris_` 이름과 fused QKV의 q/k/v 부분 mapping을 추가했다. `lora_unet_` 이름은 기존에도 252/269개가 모두 대응됐다.
- 공유 DoRA 출력축 분모가 기본 weight norm을 쓰던 문제를 업데이트 방향 norm으로 수정했다. 사전 독립 예제의 최대 오차는 output 축 0.671199, input 축 약 2.4e-7이었다.
- 코드·회귀 테스트만 변경했고 서버 재시작이나 이미지 생성 요청은 하지 않았다.

## 검증

- `/tmp/moss-training-venv` 분리 환경. PyTorch 2.14.1, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0, safetensors 0.8.0, pytest 9.1.1. CPU 스레드 환경 변수는 모두1.
- 신규 MOSS 테스트: 83개. TXT/JSON/좌표/cache, CPU FP32/CUDA BF16의 두 프리셋과 세 학습 조합, native/portable 복원, 실제 dataset을 통한 세 캡션 형식의 1→2 step 재개를 포함한다. 실제 학습 루프의 Qwen/VAE와 pretrained loader는 축소 대역을 사용한다.
- 실제 CUDA block swapping을 켜고 2단계 forward/backward/SGD를 수행해 비활성 경로의 출력/gradient와 비교했다. optimizer 복원은 같은 구성의 native 모델에 수행한다. portable 복원 후 optimizer index를 그대로 연결하면 parameter 순서가 달라질 수 있으므로 이를 완전 재개로 취급하지 않는다.
- Comfy interop은 실제 `load_lora/calculate_weight`를 사용한다. 모든 252/269개 이름, 두 prefix, QKV offset, 배율0/0.5/1, 표준 LoRA/LoKR/LoKR+DoRA 양축을 확인했다.
- 기존 sd-scripts Lumina strategy/trainer 및 Flux train utils: 41개 통과.
- Comfy 공유 DoRA 독립 회귀12개와 기존 LoRA stack node 회귀13개: 합계25개 통과. Linear/Conv weight의 두 norm축과 배율0/0.5/1을 포함한다.
- 최종 선택은 신규 MOSS83개 + 기존 sd-scripts41개 + Comfy25개 = 총149개 통과했다. 신규 파일의 Python 정적 오류/unused import 검사와 전체 diff whitespace 검사는 통과했다. 공통 파일에는 변경 전부터 있던 dataset forward-reference F821 지적4개가 그대로 있으며 추가된 지적은 없다.
- 공식 pipeline source SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`을 메모리에서 참조했다. 축소 모델의 CPU FP32/CUDA BF16 forward와 모든 parameter gradient가 정확히 일치했다. shift/step 조합9개의 scheduler sigmas도 정확히 일치했다.
- workflow JSON의 10개 노드와10개 연결 구조, Python AST, diff whitespace를 검사했다.
- 기존 환경의 flash-attn 확장은 검증 환경의 torch ABI와 맞지 않아 upstream 경고 후 native SDPA로 fallback했다. flash-attn 검증으로 표현하지 않는다.

## 제한과 후속 작업

- 원본 전체 Base/SFT 모델 디렉터리는 준비하지 않았다. 공식 소스 코드만 확인했고 전체 모델 가중치를 다운로드하지 않았다.
- 전체5B/Qwen/VAE 학습·생성·VRAM 사용량·수렴/화질·장기 실행·Windows/MPS는 미검증이다. native UI workflow는 구조를 검사했으며 실제 UI 이미지 생성은 하지 않았다.
- 검증 조합은 표준 LoRA, full-matrix LoKR, full-matrix LoKR+DoRA다. 추가 진단에서 기존 LyCORIS generic LoRA+DoRA merge가 multiplier 0/0.5에서 표준 residual과 달랐다. 이 기존 비-LoKR 문제는 이번에 수정하지 않았고 보장 범위에 포함하지 않는다.
- 새 변경은 미커밋 상태다. 원격 main을 바꾸려면 이후 별도 게시 승인이 필요하다. 백업과 Anima BF16 제거 영향을 함께 안내한다.
- 원격 `origin/main`은 읽기 전용 `ls-remote`로 기존 `7eff3367d8cf757e1cb807a51945d3a07cf74085` 유지 확인. 검증용 `/tmp` 가상환경·패키지 캐시·복구 checkout은 검사 후 정리한다.

## 게시 준비

- 이후 사용자가 이번 변경의 커밋과 푸시를 요청했다. 검증한 MOSS 변경만 로컬 커밋으로 만든다.
- 게시 직전 원격 main은 여전히 `7eff336`이다. 로컬 기반은 `690ea7f`이므로 이력을 교체하는 게시에는 강제 업데이트와 main 보호 규칙의 별도 처리가 필요하다. 백업 브랜치와 bundle은 유지한다.
