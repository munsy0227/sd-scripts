# 공식 dev 후속 수정과 브랜치 게시

사용자의 “그것도 반영해서 이제 커밋 푸시하자” 요청에 따른 작업이다. 게시 대상은 `origin/codex/sd-scripts-moss-dev-integration`이며 main 반영이나 PR 생성은 포함하지 않는다.

## 변경과 이력

- 이전 미커밋 dev 통합을 `be216b4` (`Integrate official MOSS-Image training with LyCORIS compatibility`)로 저장했다. 기존 fork와 공식 `3e84e58...`의 정상 merge commit이다.
- 이어 공식 `upstream/dev`의 `6a3290182c786541ccf292106782b51568e46da5`를 정상 병합한다. 이 커밋은 stage 3 PR #2460과 후속 수정 두 개를 포함한다.
- 이미 반영한 stage 3 및 포크 보완 때문에 겹친 7개 파일은 검증된 직전 커밋의 내용을 보존했다. 충돌 해결 후 소스 차이가 아래 두 수정뿐임을 확인했다. 기존 validation/cache/TE 해제/offload 및 LyCORIS 범위를 바꾸지 않았다.
- `b04b1a80`: 학습 중 sample generation의 `show_progress=False`를 제거해 tqdm 진행률을 표시한다.
- `8dbdefba`: `debug_dataset`에서 conditioning/masked 이미지의 key 존재 여부 대신 실제 값을 확인한다. conditioning 값이 `None`이면 masked 이미지를 사용하고 둘 다 `None`이면 건너뛴다.
- 사용자 가이드의 공식 dev 기준을 `6a329018`로 갱신했다.

원래 staged 작업 전체는 `/home/munsy0227/LyCORIS/.internal/backups/sd-scripts-stage3-ready-20261018.patch`에 보존했다(517,918 bytes). 이전 backup과 native fixture도 보존한다. 이력 재작성이나 force push는 하지 않는다.

## 추가 검증

기존 `.venv`를 수정하지 않고 `/tmp/moss-publish-venv`와 별도 uv cache를 사용했다. PyTorch 2.13.0+cu130, transformers 5.17.0, diffusers 0.40.0, accelerate 1.15.0이며 CPU 스레드 제한과 HF offline 설정을 적용했다.

- 실제 `library.dataset.debug_dataset`를 축소 dataset으로 실행: 둘 다 None, masked fallback, conditioning 우선의 3개 경우 모두 정상. Linux 경로이며 Windows GUI 검증은 아니다.
- `test_sampling_returns_to_training[cpu/cuda]`: 2 passed. 실제 tqdm 출력이 관찰됐고, sample 저장 후 LoKR+DoRA backward 및 block-swap 학습 복귀가 정상이다.
- 새 runtime 변경은 위 두 수정뿐이므로 이전 단계의 5B 전체 모델 미검증 등 검증 경계는 그대로다. 광범위한 기존 검사를 다시 실행한 결과로 표기하지 않는다.
- Ruff, AST, whitespace 및 staged 파일 범위와 merge 상태를 확인하고 원격 ref와 로컬 HEAD를 대조해 게시를 검증한다.

추가 정적 검사에서 변경한 MOSS helper의 Ruff F와 Python AST/whitespace는 정상이었다. 공용 dataset 파일의 Ruff 진단은 직전 HEAD 및 upstream/dev와 동일하며 새 진단은 없었다. 충돌 해결 후 실제 diff는 두 소스 수정과 문서/기록 5개뿐이었다.

이번 검증용 환경/cache는 검사 후 정리한다. 원래 `.venv`와 `/tmp/moss-dev-legacy-fixture`는 유지한다. 관련 LyCORIS 문서/회귀검사는 별도 `codex/moss-sd-scripts-integration` 브랜치에 게시하며 사용자 `artifacts/`와 autostop worklog는 제외한다.
