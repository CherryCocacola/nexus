"""
오케스트레이터 패키지 — 쿼리 루프와 관련 모듈을 포함한다.

4-Tier AsyncGenerator 체인의 Tier 2에 해당하며,
모델 호출 → 이벤트 수집 → 도구 실행 → 다음 턴의 핵심 루프를 담당한다.

모듈:
  - query_loop: while(True) 에이전트 턴 루프 (핵심)
  - stream_handler: 스트리밍 중 도구 병렬 실행 최적화
  - context_manager: 4단계 컨텍스트 압축 파이프라인
  - stop_resolver: 종료/계속 판단 로직
"""
