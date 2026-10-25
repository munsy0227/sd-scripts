# 원본 main과 작업 브랜치 동기화

## 사용자 요청과 범위

- 사용자가 원본과의 동기화를 요청했고, 대상은 현재 작업 브랜치라고 명시했다.
- 포크 작업 브랜치: `codex/sd-scripts-moss-dev-integration`, 시작 커밋 `e1fa3736439e58690731715040172999eac7c89d`.
- 원본 main: `11d8cedd0c67b93a10f8b2d75f43554f6f12c794` (v0.12.2).
- 선별 패치만 더하는 대신 해당 원본 커밋을 두 번째 부모로 하는 실제 병합을 만든다. main 브랜치, 사용자 TOML, LyCORIS 저장소는 변경하지 않는다.

## 충돌 처리

- 이미 이식한 H2D-only와 다중 후보 캐시는 포크의 구현을 유지했다. JSON/layout geometry, 모델 fingerprint, 배열 검증, backup/atomic replace를 보존한다.
- `moss_train_network.py`, legacy timestep, Qwen BF16/FP32 rotary 및 외부 autocast 차단, resume step, CPU checkpoint saved-input-only 수정을 유지했다.
- 원본의 ConvRot INT8 kernels/streaming quantizer, safetensors 수정, native loader·inference·cache 도구를 연결했다.
- LyCORIS는 회전된 INT8 DiT 가중치를 해석하지 못하므로 adapter 생성 전에 명확한 오류로 차단한다. 공식 native network 모듈은 허용한다. Qwen만 INT8로 바꾸는 구성은 별도 옵션으로 처리한다.
- Qwen 양자화 여부를 텍스트 캐시 identity에 반영한다. dense 설정의 기존 identity는 유지한다.
- native INT8 scale은 FP32를 유지하도록 공통 dtype cast를 건너뛴다. Qwen checkpoint workaround는 첫 floating-point parameter를 선택한다.
- 원본 README/release history와 전체 MOSS guide를 포함하고, 포크 README 링크와 한국어 사용 설명서를 유지·갱신했다.

## 검증

- 기존 `.venv`는 변경하지 않았다. `/tmp/sd-scripts-upstream-test-deps`에 transformers 5.17.0, diffusers 0.40.0, accelerate 1.15.0과 필요한 의존성을 설치하고 기존 PyTorch 2.13.0+cu130을 사용했다.
- Linux RTX 4070. CPU 스레드 1개, Hugging Face offline, pytest cache 비활성화.
- CPU: **164 passed, 60 skipped, 1 deselected**. 캡션·캐시·MOSS 모델/트레이너·LoRA generic·offloader·safetensors·ConvRot을 포함했다. CPU per-block compile과 실제 축소 Qwen/VAE 검사도 통과했다.
- CPU에서 skip한 device/backend 검사 60개를 CUDA 접근 환경에서 실행: **58 passed, 2 skipped**. 기존 plain/H2D LoKR+DoRA BF16 gradient 비교, 두 진입점의 축소 학습·저장·재개, INT8 CUDA kernel을 포함했다. skip은 해당 하드웨어가 없는 XPU와 MPS synchronize 검사다.
- 포크 INT8 DiT 차단/native 허용, Qwen quantization cache identity, quantized encoder checkpoint workaround를 회귀 검사에 추가했다.
- 전체 5B/9B pretrained 학습, 화질·속도·최대 메모리, 장기 실행, Windows/MPS는 이번에 검증하지 않았다. timestep 시각화 검사는 기존 제한 시간 문제로 선택에서 제외했다.

임시 로그: `/tmp/sd-scripts-upstream-sync-cpu.log`, `/tmp/sd-scripts-upstream-sync-cuda.log`.
