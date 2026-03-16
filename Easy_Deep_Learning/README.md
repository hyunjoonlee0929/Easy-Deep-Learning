# Easy Deep Learning

Easy Deep Learning은 사용자가 CSV 데이터로 분류/회귀 문제를 빠르게 학습하고,
저장된 모델로 다른 데이터셋을 재평가할 수 있도록 만든 경량 ML 애플리케이션입니다.

## 핵심 기능

- CSV 기반 학습 파이프라인
- 분류 / 회귀 지원
- 모델 선택:
  - 딥러닝: `dnn` (NumPy 기반)
  - 전통 ML: `rf`, `svm`, `knn`, `lr`, `gbm`, `xgboost`
- 자동 전처리 (결측치 처리, 스케일링, 인코딩)
- 학습 결과/모델/전처리기 자동 저장 (`runs/{run_id}`)
- 저장된 모델로 새 테스트 데이터 평가
- Auto 모델 추천 (`--model-type auto`)
- AutoML 리더보드 생성 (`automl` 명령)
- 실행 리포트 HTML 자동 생성
- Streamlit 대시보드에서 메트릭 시각화
- Streamlit에서 리포트/Confusion Matrix/ROC 아티팩트 확인
- AutoML 리더보드 CSV 다운로드
- 이미지(CNN/ResNet) / 텍스트(GRU/LSTM/TextCNN/Transformer-lite) 모델 데모
- 오디오/비디오 데모 (WAV/프레임 시퀀스)
- Tool-Using Agent 데모 (데이터 요약 + 모델 추천)
- RAG + Auto Evaluation 데모
- 멀티모달 검색 데모 (이미지/텍스트)
- GitHub README 요약 챗봇 (룰 기반 + OpenAI 옵션)
- GitHub 저장소 구조 분석 + 실행 커맨드 제안
- AI 자동 리포트 (OpenAI 옵션 + 룰 기반)
- SHAP + PDP + ICE 해석 리포트
- 에러 분석 (오분류/잔차 상위 샘플, Residual plot)
- 실패 샘플 로컬 중요도(SHAP 기반) 요약
- 모델 개선 추천 리포트
- SHAP Interaction + Interaction PDP
- Auto Tuning (간단한 하이퍼파라미터 탐색)

## 프로젝트 구조

```text
Easy_Deep_Learning/
├── core/
│   ├── data_validator.py
│   ├── preprocessing.py
│   ├── model_engine.py
│   ├── model_registry.py
│   ├── automl.py
│   ├── trainer.py
│   ├── experiment_tracker.py
│   ├── reporting.py
│   ├── workflows.py
│   └── torch_workflows.py
├── dashboard/
│   └── app.py
├── config/
│   └── model_config.yaml
├── data/
│   ├── example_dataset.csv
│   ├── text_sample_sst2.csv
│   └── text_sample_trec.csv
│   └── text_sample.csv
└── main.py
```

## 설치

```bash
pip install -r Easy_Deep_Learning/requirements.txt
```

## FastAPI 예측 서버

```bash
uvicorn Easy_Deep_Learning.api.app:app --reload --port 8000
```

요청 예시:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"run_id":"<run_id>","records":[{"col1":1.2,"col2":"A"}]}'
```

## CLI 사용법

### 1) 학습

```bash
python Easy_Deep_Learning/main.py train \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification \
  --model-type dnn \
  --seed 42
```

모델 파라미터 커스터마이즈:

```bash
python Easy_Deep_Learning/main.py train \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification \
  --model-type rf \
  --model-params '{"n_estimators": 300, "max_depth": 6}'
```

Auto 모델 추천 사용:

```bash
python Easy_Deep_Learning/main.py train \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification \
  --model-type auto
```

AutoML 리더보드:

```bash
python Easy_Deep_Learning/main.py automl \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification \
  --max-models 6
```

### 2) 저장된 모델로 테스트

```bash
python Easy_Deep_Learning/main.py test \
  --from-run <run_id> \
  --data /path/to/new_test.csv
```

### 3) 이미지(CNN) 학습/테스트

```bash
python Easy_Deep_Learning/main.py image-train --dataset MNIST --epochs 5 --lr 0.001 --batch-size 64 --model-arch cnn
python Easy_Deep_Learning/main.py image-train --dataset SVHN --epochs 5 --lr 0.001 --batch-size 64 --model-arch resnet18
python Easy_Deep_Learning/main.py image-test --from-run <run_id>
```

### 4) 텍스트(RNN) 학습/테스트

```bash
python Easy_Deep_Learning/main.py text-train --dataset AG_NEWS_SAMPLE --epochs 3 --lr 0.001 --batch-size 64 --model-arch gru
python Easy_Deep_Learning/main.py text-train --dataset SST2_SAMPLE --epochs 3 --lr 0.001 --batch-size 64 --model-arch lstm --stopwords --ngram 2
python Easy_Deep_Learning/main.py text-train --dataset TREC_SAMPLE --epochs 3 --lr 0.001 --batch-size 64 --model-arch textcnn
python Easy_Deep_Learning/main.py text-train --dataset TREC_SAMPLE --epochs 3 --lr 0.001 --batch-size 64 --model-arch transformer --bpe --bpe-vocab-size 300
python Easy_Deep_Learning/main.py text-test --from-run <run_id>
```

### 5) Tool-Using Agent

```bash
python Easy_Deep_Learning/main.py agent \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification
```

### 6) RAG + Auto Evaluation

```bash
echo -e "Document one about ML.\nDocument two about NLP." > /tmp/docs.txt
python Easy_Deep_Learning/main.py rag --query "What is NLP?" --docs /tmp/docs.txt
```

### 7) Multimodal Search (Streamlit)

- Multimodal 탭에서 텍스트/이미지 업로드 후 검색 실행

### 8) Auto Tuning

```bash
python Easy_Deep_Learning/main.py tune \
  --data Easy_Deep_Learning/data/example_dataset.csv \
  --target-column target \
  --task-type classification \
  --model-type rf \
  --max-trials 10
```

### 9) Run Comparison Report

```bash
python Easy_Deep_Learning/main.py compare --run-ids <run_id_1>,<run_id_2>
```

커스텀 텍스트 CSV 사용:

```bash
python Easy_Deep_Learning/main.py text-train \
  --data /path/to/text.csv \
  --text-column text \
  --label-column label
```

## 실행 아티팩트

학습 실행마다 `runs/{run_id}` 생성:

- `config_snapshot.yaml`
- `validation_report.json`
- `metrics.json`
- `feature_names.json`
- `model_params.json`
- `preprocessor.joblib`
- `model.json` 또는 `model.model` 또는 `model.pt`
- `model_info.json`
- `config_hash.txt`
- `report.html`
- `auto_recommendation.json` (auto 사용 시)
- `leaderboard.json`, `best_run.json` (automl 사용 시)
- `predictions_preview.json`
- `confusion_matrix.png` (분류 모델)
- `roc_curve.png` (이진 분류 + 확률 출력 가능)
- `prediction_scatter.png` (회귀 모델)

## 대시보드

```bash
streamlit run Easy_Deep_Learning/dashboard/app.py
```

- **Train Model 탭**: 데이터 업로드 → 모델 선택/파라미터 조정 → 학습
- **Test Saved Model 탭**: run_id 선택 + 테스트 CSV 업로드 → 재평가
- **Image Models 탭**: MNIST/FashionMNIST/CIFAR10 CNN 데모
- **Text Models 탭**: 샘플 텍스트 데이터 또는 CSV 업로드로 RNN 데모
- **Agent 탭**: 데이터 요약 + 모델 추천 툴 호출 데모
- **RAG 탭**: 문서 입력 → 검색 → 답변 + 자동 평가
- **Multimodal 탭**: 이미지/텍스트 업로드 → 유사도 검색
- **Chatbot 탭**: GitHub 링크/README 입력 → 기능/사용법 요약
- **Tabular 탭**: 학습/테스트 결과 + AI 리포트 + SHAP/PDP/ICE 아티팩트 확인
