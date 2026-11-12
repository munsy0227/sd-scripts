# MOSS·Anima 통합과 main 반영

## 요청과 복구 자료

- 사용자가 두 저장소의 원본 동기화, MOSS·과거 Anima 통합, 로컬·원격 main 반영 및 포함이 확인된 브랜치 삭제를 승인했다.
- 원본 작업 폴더와 기존 `.venv`는 수정하지 않고 `/tmp/codex-moss-anima-main/sd-scripts`에서 통합했다.
- 시작 시점 전체 refs의 검증된 bundle과 ref 목록은 `/home/munsy0227/LyCORIS/.internal/backups/main-integration-20261112-101225/`에 보존했다.

## 통합 내용

- MOSS 시작점 `08d83e48308f056e744b49c977457dca3559e009`에 원본 main `cb70b96858477e716a2096292ab8aab433277f38`(v0.12.3)을 병합했다. 병합 커밋은 `d319310`이다.
- attention은 원본의 사전 계산된 Python sequence-length metadata를 사용하고 기존 fullgraph·gradient 회귀 검사를 유지했다.
- 새 `truncated_logit_normal`과 포크의 `legacy` timestep 선택을 모두 유지했다. sampling fixture에 공식 cutoff/uniform-ratio 기본값을 보완했다(`e02bd9c`).
- ConvRot INT8 custom ops의 compile을 허용하면서 block-swap Linear 제외와 Inductor `emulate_precision_casts=True`를 유지했다. 충돌 결합 결과에 회귀 검사를 추가했다.
- Anima 백업 `7eff3367d8cf757e1cb807a51945d3a07cf74085`를 `befd60d`로 병합했다. 최신 offloader/checkpoint 구조 위에 Block·FinalLayer의 `use_fp32=False`일 때 `nullcontext()`를 쓰는 수정과 정밀도 테스트를 복원했다.
- MOSS 두 진입점, legacy TXT·JSON/layout, 캐시 identity·재개, Qwen BF16/FP32 rotary, LoKR·DoRA 및 offloader 동작을 회귀 검사했다.

## 검증

- 별도 Python 3.12.15 환경에서 기존 torch 2.13.0+cu130을 읽기 재사용했다. transformers 5.17.0, diffusers 0.40.0, accelerate 1.15.0을 임시 환경에만 설치했고 `pip check`를 통과했다.
- CPU 스레드는 1개, Hugging Face/Transformers는 offline, `PYTHONPATH`는 두 통합 작업 폴더로 고정했다.
- 최종 RTX 4070 pytest 선택: **361 passed, 10 skipped, 1 deselected**. 복원 Anima 정밀도, attention fullgraph·gradient, MOSS 학습·캡션·캐시·저장/재개, checkpoint·Inductor, offloader, generic LoRA, Prodigy, safetensors, ConvRot INT8, 원본 sampling·시각화 bin 회귀를 포함했다.
- skip은 XPU 1개, MPS 1개, CUDA 최적화의 범위 밖인 CPU half dot 8개다. 기존 장시간 `test_show_timesteps_console` 1개는 제외했다. 첫 CPU 실행의 새 sampling fixture 기본값 누락은 수정 후 개별 3개 검사와 위 최종 선택에서 통과했다.
- 실제 축소 Anima 28블록·448 full-matrix LoKR/DoRA/scalar의 CUDA BF16 검사: checkpointing off/on 모두 초기 no-op, 448/448 finite nonzero scalar gradients, optimizer 후 출력 변화, frozen base 보존, 파일 저장·재로딩 후 native state 출력 완전 일치를 통과했다.
- offline `--help` 6개: Anima 전체·network·LLLite trainer, MOSS 두 진입점, 공통 network trainer 모두 통과했다.
- 변경 Python 82개 AST 검사 및 `git diff --check origin/main..HEAD`를 통과했다.
- 최종 pytest 로그: `/tmp/codex-sd-moss-anima-cuda.log`. JUnit: `/tmp/codex-sd-moss-anima-cuda.xml`. Anima smoke: `/tmp/codex-anima-lycoris-bf16-smoke.py`와 같은 basename의 `.log`.

## 반영과 정리 기준

- 검증 후 LyCORIS, 이어 sd-scripts main을 정상 fast-forward push한다. 실행한 원격 SHA·CI·삭제 결과는 위 복구 디렉터리의 완료 기록에 보존한다.
- 최종 main의 조상임이 확인된 로컬·원격 브랜치만 삭제한다. 다른 worktree에서 사용 중인 로컬 브랜치와 고유 미병합 커밋은 보존한다.
- 전체 pretrained 학습·화질·장기 성능, Windows/XPU/MPS 실기기는 검증하지 않았다. 지정된 Knowledge 파일은 없다.
