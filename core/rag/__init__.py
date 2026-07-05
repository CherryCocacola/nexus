"""core.rag — RAG(검색 증강 생성) 파이프라인 패키지.

이 패키지는 "프로젝트 파일·외부 지식을 미리 인덱싱해 두었다가, 사용자의 질문과
관련된 부분만 골라 모델 컨텍스트에 끼워 넣는" 일련의 과정을 담당한다.
모델이 모든 문서를 통째로 읽을 수는 없으므로, 질문에 정말 필요한 청크만
검색(retrieval)해서 프롬프트에 주입(augment)하는 것이 핵심 아이디어다.

크게 두 축으로 구성된다.
  - 문서/코드 RAG: 프로젝트 파일을 청크로 나눠 임베딩으로 저장하고 검색한다.
  - 지식 RAG: 위키 등 외부 교양 지식(tb_knowledge)을 별도로 저장·검색한다.

주요 하위 모듈(자세한 설명은 각 파일의 docstring 참고):
  - indexer.py            : 프로젝트 파일을 청크로 분할해 임베딩으로 인덱싱한다.
  - retriever.py          : 쿼리와 유사한 청크를 찾아 토큰 예산 내 컨텍스트로 반환한다.
  - knowledge_store.py    : tb_knowledge 기반 장기 교양 지식 저장소.
  - knowledge_retriever.py: KNOWLEDGE_MODE 전용 지식 검색 → 프롬프트 주입.
  - symbol_indexer.py     : 다언어(Python/JS/TS/Go) 함수·클래스 심볼 추출기.
  - symbol_store.py       : tb_symbols 기반 심볼 인덱스 저장소.
  - pgvector_base.py      : KnowledgeStore·SymbolStore가 공유하는 pgvector 공통 베이스.
  - parsers/              : 언어별 심볼 파서 구현 모음.

이 파일 자체는 패키지 초기화 진입점(__init__.py)일 뿐이며, 별도의 실행 코드나
공개 심볼(re-export)은 두지 않는다. 필요한 클래스·함수는 각 하위 모듈에서 직접
import 해서 쓴다(예: `from core.rag.indexer import ...`).

작성자: 이현수 / 작성일: 2026-07-05
"""

# RAG(Retrieval-Augmented Generation) 파이프라인 패키지 마커.
# 프로젝트 파일을 인덱싱하고, 질문과 관련된 청크만 검색하여 컨텍스트에 주입한다.
