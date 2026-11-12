# Anima BF16 LoKR·DoRA 훈련 경로 최적화

## 범위와 동작

- 사용자 요청은 Anima 훈련 최적화이며, 학습 방식은 BF16 LoKR·DoRA로 지정했다. 실제 GPU·TOML은 제공되지 않았다.
- sd-scripts 기준은 `f61bdef7a4aa5953f898c9947c96c8de6f83a5be`, LyCORIS는 `0defe85beb55acfa5687d3e17d9a3d2089ef87ba`다. 별도 `codex/anima-training-speed` 브랜치에서 구현했다.
- 일반 non-reentrant gradient checkpointing에서는 latent와 text condition에 `requires_grad=True`를 강제하지 않는다. 동결된 입력·텍스트 브리지까지 이어지던 불필요한 backward를 생략한다. 자연히 필요한 text-encoder/LLM-adapter gradient를 detach하지 않는다.
- block swapping, Unsloth, CPU offload의 기존 입력 gradient 처리는 유지한다. 모델·체크포인트 키, adapter 범위, loss·optimizer·학습률·사용자 설정은 변경하지 않는다.
- Anima per-block Inductor compile에도 MOSS와 같은 `emulate_precision_casts=True`를 전달한다. compile mode preset은 기존 공통 helper가 유지하며 다른 backend에는 이 옵션을 넘기지 않는다. 이 옵션은 BF16 수치 보존용이다.
- 근거: [PyTorch non-reentrant checkpoint 설명](https://docs.pytorch.org/docs/stable/checkpoint), [PyTorch 2.6 Inductor 설정](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/config.py). 옵션은 지원 하한인 2.6 소스에도 존재하지만 실행 검증은 아래 2.13 환경에서 수행했다.

## 검증 환경과 회귀 검사

- 별도 Python 3.12.15 환경, PyTorch 2.13.0+cu130, RTX 4070, transformers 5.17.0, diffusers 0.40.0, accelerate 1.15.0. 기존 sd-scripts `.venv`는 변경하지 않았다.
- CPU 스레드 1개, Hugging Face offline, 실제 검증 대상 소스 경로를 `PYTHONPATH`에 지정했다.
- `tests/test_anima_speed.py tests/test_anima_precision.py`: CPU **26 passed / 20 skipped**, CUDA **46 passed**. 중복되는 선택이므로 두 숫자를 합산하지 않는다.
- 새 16개 검사는 plain/swap/Unsloth/CPU-offload gradient 분기, 필요한 text gradient 보존, backend 옵션, scalar 초기/학습 중 상태의 실제 Block 출력·모든 adapter gradient를 검사한다. 작은 CPU FP32/CUDA BF16 비교는 `rtol=atol=0`으로 통과했다.
- 실제 trainer를 통한 Inductor fullgraph: eager 대비 output `rtol=2e-2, atol=2e-3`, gradient `5e-2/2e-3`, Prodigy 5step loss·parameter `5e-3/1e-5`를 모두 통과했다. 기존 MOSS 기준을 확대하지 않았다.
- step2의 native adapter+optimizer를 파일에 저장하고 3step 재개한 최종 parameter는 연속 실행과 정확히 일치했다(max abs 0). 기존 precision-option 없는 compile은 같은 output 기준을 통과하지 못했다.
- 새 테스트 Black 및 변경 Python의 Ruff 오류 검사를 통과했다.

## RTX 4070 축소 실험

입력 gradient 강제 켬/생략 비교다. 실제 frozen LLMAdapter 6층(512 dim, 8 heads)과 Anima Block 1개(512 dim, image 1,024 tokens, context 128 tokens, checkpointing)를 연결했다. LoKR·DoRA 16개, scalar 0.5 및 nonzero factor를 고정하고 warmup 3회 뒤 forward/backward 10회의 median을 측정했다. optimizer 시간은 포함하지 않았다.

| Adapter dtype | 강제 켬 | 생략 | Step 시간 감소 | Peak allocated 켬 → 생략 |
|---|---:|---:|---:|---:|
| FP32 | 37.563ms | 30.386ms | 19.1% | 220.220 → 202.448MiB |
| BF16 | 42.213ms | 33.908ms | 19.7% | 221.001 → 203.229MiB |

- Output은 완전히 같았고 frozen bridge parameter gradient는 없었다. bridge output의 gradient 필요 여부가 True에서 False로 바뀜을 확인했다.
- 큰 GPU 실험의 gradient는 같은 조건 반복에서도 bitwise 일치하지 않았다. FP32 adapter의 반복/변경 최대 절대 차이는 각각 `6.887e-9`/`3.308e-8`, BF16은 `1.490e-8`/`7.451e-9`였다. raw diagnostic을 보존하며 strict exact 통과로 표현하지 않는다.
- LLMAdapter 없이 Block만 측정한 첫 실험에서는 메모리 약 3.01MiB 감소, FP32 26.55→28.14ms/BF16 28.44→27.52ms로 시간 분포가 겹쳤다. 일관된 속도 개선 증거로 사용하지 않는다.
- 별도 작은 Block(128 dim, BF16 base/FP32 adapter)의 동일-state compile 비교는 eager 28.226ms / 기존 compile 5.485ms / 새 정밀도 옵션 compile 5.710ms다. compile 자체는 기존 기능이며 새 옵션이 5배 가속을 만든 것이 아니다. 새 옵션은 약 4.1% 비용으로 위 수치 기준을 보존했다. 최초 compile 실행 시간은 steady state와 별도 로그에 보존했다.

## 사용과 제한

- 일반 checkpointing 최적화는 `gradient_checkpointing=true`이고 swap·Unsloth·CPU offload가 꺼져 있을 때 자동 적용된다. 기존 설정을 임의로 변경하지 않았다.
- compile은 사용자가 기존 옵션 `compile=true`, `compile_backend="inductor"`를 선택할 때 적용되며 첫 실행에는 컴파일 비용이 있다. 필요한 latent/text cache를 사용할 때도 stochastic caption·augmentation 제약은 그대로 준수해야 한다.
- 위 19~20%는 frozen bridge와 Block 1개를 사용한 축소 prefix의 결과다. 전체 28블록 모델, 사용자 GPU/TOML, 실제 데이터셋, 장기 수렴·화질에 대한 개선율이 아니다. compile 실측은 FP32 adapter 조건이며 full-BF16 adapter compile은 별도 미검증이다.
- CPU offload의 기존 live block output CPU 이동 경로는 이번 수정 범위에서 제외했다. 이 옵션의 CUDA 동작·속도는 검증하지 않았으며 일반 checkpointing의 개선율을 적용하지 않는다. XPU/MPS/Windows도 미검증이다.
- 지정된 Knowledge 파일은 없다. 전체 모델 학습이나 원격 게시를 수행하지 않았다.
