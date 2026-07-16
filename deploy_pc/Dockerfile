# Nexus 서비스(Machine A: 웹/오케스트레이터) 컨테이너 이미지
# GPU 라이브러리(torch·vllm·sentence-transformers) 제외 — 추론·임베딩은 HTTP로 위임.
FROM python:3.11-slim

# C 확장(hiredis 등) 대비 최소 빌드 도구 + curl(헬스체크용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 의존성 먼저 설치 (레이어 캐시 최적화)
COPY deploy_pc/requirements-pc.txt /app/deploy_pc/requirements-pc.txt
RUN pip install --no-cache-dir -r deploy_pc/requirements-pc.txt

# 2) 앱 코드 복사
COPY . /app

# 3) 생성물 영속 경로(볼륨 마운트 대상)
RUN mkdir -p /app/data/exports

ENV PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
EXPOSE 8600
# 실행 시 주입: NEXUS_CONFIG, NEXUS_PG_PASSWORD, NEXUS_REDIS_PASSWORD
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8600", "--log-level", "info"]
