# Easy Deep Learning
## End-to-end AI platform for reproducible ML workflows and multi-modal experimentation
Easy Deep Learning은 단일 모델 데모가 아니라, 데이터 입력부터 학습/평가/해석/리포팅/서빙까지 연결하는 **실행형 AI 플랫폼**입니다.  
목표는 사용자가 복잡한 ML/딥러닝 실험을 빠르게 반복하고, 결과를 해석 가능하게 관리하며, 실제 서비스 형태(API/대시보드)로 전환할 수 있게 하는 것입니다.

## Platform Vision

- End-to-End AI Workflow
  - 데이터 로드 -> 전처리 -> 학습 -> 평가 -> 해석 -> 리포트 -> 재테스트 -> API 서빙
- Multi-Modal Workspace
  - Tabular / Image / Text / Audio / Video / RAG / Multimodal / Agent / Chatbot 통합
- Reproducible Experiment System
  - `runs/{run_id}` 기반 아티팩트 추적, 모델 재사용, 비교 리포트
- Human-Friendly UX
  - Streamlit 기반 인터랙티브 UI + Quick Start + 고급 옵션 토글

## What Makes It a Platform

1. **통합 인터페이스**
- 단일 UI에서 서로 다른 AI 작업 유형을 수행
- 학습과 추론, 리포팅, 챗봇, 요약까지 한 흐름으로 연결

2. **모듈형 엔진**
- 전처리/모델/평가/해석/리포팅/서빙을 독립 모듈로 분리
- CLI, Dashboard, API가 동일 코어를 공유

3. **실험 추적 중심**
- 실행 결과를 파일 단위로 구조화 저장
- 재현/비교/재테스트가 가능한 실험 관리 구조

4. **확장 가능 구조**
- 전통 ML + 딥러닝 + LoRA 기반 LLM fine-tuning
- 객체탐지/ASR/RAG/멀티모달 검색 등 확장 기능 포함

## Core Capabilities

- Tabular ML/DL
  - 분류/회귀, 자동 전처리, 모델 학습/평가, 저장 모델 재테스트
- Model Families
  - `dnn`, `tab_transformer`, `rf`, `svm`, `knn`, `lr`, `gbm`, `xgboost`
- Image/Text
  - CNN/ResNet/ConvNeXt/ViT, RNN/Transformer 기반 학습 및 테스트
- Audio/Video
  - 오디오 특징/ASR/분류 데모, 비디오 프레임 기반 디텍션
- Detection
  - YOLO 계열 + Faster R-CNN 기반 이미지/비디오 객체 탐지
- Interpretability
  - SHAP, PDP/ICE, 에러 분석, 상호작용 분석
- LLM Features
  - LoRA fine-tuning, 생성 테스트, 멀티턴 챗 연결
- Agent/RAG/Multimodal
  - Tool-Using Agent, RAG + 자동 평가, 이미지/텍스트 검색
- Serving
  - FastAPI `/predict`, `/llm/generate`, `/llm/chat`

## System Architecture

```text
Data Ingestion
  -> Validation / Preprocessing
  -> Training / Tuning / CV
  -> Evaluation / Explainability
  -> AI Report Generation
  -> Run Artifacts Tracking
  -> Inference API / Dashboard
```

## Project Structure

```text
Easy_Deep_Learning/
├── core/              # 학습/평가/해석/리포팅/튜닝/LLM/멀티모달 코어
├── dashboard/         # Streamlit UI
├── api/               # FastAPI 서버
├── config/            # 모델 설정
├── data/              # 샘플 데이터셋
├── runs/              # 실험 아티팩트 저장
└── main.py            # CLI 엔트리포인트
```

## Quick Start

```bash
pip install -r Easy_Deep_Learning/requirements.txt
streamlit run Easy_Deep_Learning/dashboard/app.py
```

## API Server

```bash
uvicorn Easy_Deep_Learning.api.app:app --reload --port 8000
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"run_id":"<run_id>","records":[{"col1":1.2,"col2":"A"}]}'
```

## Run Artifacts

각 실행은 `runs/{run_id}`에 저장되며, 대표 아티팩트는 아래와 같습니다.

- `config_snapshot.yaml`
- `metrics.json`
- `model_info.json`
- `model artifact` (`.json/.model/.pt`)
- `preprocessor.joblib`
- `report.html`
- `ai_report.json`
- `top_features.json`
- `error_analysis.json`
- `drift_report.json`
