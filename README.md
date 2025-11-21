# Flow-Seg

Flow-based segmentation 프로젝트

## 설치 방법

### 요구사항
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) 패키지 매니저

### 설치

```bash
# uv 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

## 프로젝트 구조

```
flow-seg/
├── configs/              # 설정 파일
│   ├── diffusion/       # Diffusion 모델 설정
│   ├── flow/            # Flow 모델 설정
│   └── supervised/      # Supervised 모델 설정
├── data/                # 데이터셋 디렉토리
├── scripts/             # 학습 스크립트
│   ├── train_diffusion_model.py
│   ├── train_flow_model.py
│   ├── train_supervised_model.py
│   └── lightning_utils.py
├── src/                 # 소스 코드
│   ├── archs/          # 모델 아키텍처
│   │   ├── diffusion_model.py
│   │   ├── flow_model.py
│   │   ├── supervised_model.py
│   │   ├── components/  # 모델 컴포넌트
│   │   └── unet/        # UNet 구현
│   ├── data/           # 데이터 로더
│   ├── losses/         # 손실 함수
│   ├── metrics/        # 평가 지표
│   ├── loggers/        # 로깅 유틸리티
│   └── utils/          # 유틸리티 함수
├── utest/              # 단위 테스트
├── pyproject.toml      # 프로젝트 설정
└── uv.lock            # 의존성 락 파일
```

