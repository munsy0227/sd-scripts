# MOSS upstream 개선 선별 이식

## 범위

- 사용자 요청: 원본에서 좋은 변경만 가져오기.
- 기존 작업 브랜치 `codex/sd-scripts-moss-dev-integration`, 기준 `ebec22a60493471c5f4fd5a6ecb185b409ec9a76`을 유지한다.
- 원본 main `11d8cedd0c67b93a10f8b2d75f43554f6f12c794`에서 다음 세 패치를 선별 이식했다.
  - `143786a`: H2D-only block swap, 공통 offloader와 모든 내부 호출부·검사·문서.
  - `3a082f5`: H2D staging copy를 single-threaded memmove로 수정.
  - `caab8bb`: MOSS의 다중 캡션 후보 TE 캐시, dataset/strategy API와 테스트.
- ConvRot INT8은 제외했다. 현재 LyCORIS full-matrix LoKR+DoRA가 해당 정수 가중치를 해석하지 못하는 비호환이 앞선 축소 재현에서 확인됐다.
- 원본의 전체 main/dev 병합, 기본 학습 분포 변경, 사용자 학습 TOML 변경, commit/push는 하지 않았다.

## 포크 동작 보존

- `moss_train_network.py` 호환 진입점, legacy timestep 기본값, `moss`/`moss-full` 범위.
- TXT·JSON caption·layout 및 geometry 처리. 구조화 캡션의 줄바꿈은 일반 다중 후보로 분해하지 않는다.
- Qwen BF16 weight/FP32 rotary buffer 및 외부 autocast 차단.
- CPU checkpoint offload의 saved-input-only 처리와 resume step 수정.
- 캐시 버전·모델/설정 fingerprint, dtype/shape 검사, 기존 바이트 백업, atomic replace.
- 기존 버전 2 single-candidate 캐시는 fingerprint가 맞으면 재사용한다. 새 후보/캡션/설정 변경은 다시 캐시한다.
- classic block swap이 기본이며 H2D-only는 명시적으로 켠다. H2D-only 학습에는 gradient checkpointing이 필요하다.

## 검증

- 임시 checkout에서 변경을 완성하고 테스트한 뒤 실제 sd-scripts checkout에 적용한다. 기존 가상환경 패키지는 변경하지 않고 pytest만 `/tmp`에서 제공했다.
- CPU 스레드 1개, Hugging Face offline, pytest cache 비활성화.
- CPU: caption preprocessing, MOSS strategy/cache/caption/training/model/trainer/integration, common offloader, H2D config/factory 선택에서 **109 passed, 54 skipped, 6 deselected**.
- CUDA: RTX 4070, PyTorch 2.13.0+cu130, **40 passed, 50 deselected**. upstream H2D ring/pinned-memory/output/gradient 검사, MOSS의 classic/H2D BF16 LoKR+DoRA 두-step gradient 비교(`moss`와 `moss-full`), CPU/unsloth checkpoint offload, 두 진입점의 축소 BF16 저장·재개 검사를 통과했다. CPU와 CUDA 선택에는 일부 공통 검사가 있으므로 개수를 고유 총합으로 합산하지 않는다.
- 새 다중 후보 캐시 테스트는 두 번째 후보 변경과 encoder identity 변경을 감지하고, 교체 전 파일의 바이트가 백업에 보존되는지 검사한다.
- 기존 환경의 transformers에는 Qwen3.5 클래스가 없고 diffusers에는 AutoencoderKLFlux2가 없어 해당 실제 클래스 검사 및 generic Qwen 테스트를 제외했다. upstream 비교 단계에서도 같은 환경 제약을 확인했다.
- 시간 제한 내 완료하지 못했던 timestep 시각화와 선택적 compile 검사는 최종 선택에서 제외했다.
- 실제 5B/9B 가중치, 장시간 학습, 품질·성능·최대 메모리 수치, Windows/MPS는 검증하지 않았다.

임시 로그: `/tmp/sd-scripts-moss-port-cpu.log`, `/tmp/sd-scripts-moss-port-cuda.log`.
