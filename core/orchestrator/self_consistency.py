# 자기일관성(Self-Consistency) 합의 로직 — 순수 함수 모듈 (Point 4.3)
"""
자기일관성 사실 검증의 "합의(consensus)" 계산을 담당하는 순수 함수 모듈.

[이 파일이 하는 일]
같은 질의를 N번 샘플링해 얻은 후보 답 N개를 받아, 다수결(majority) 또는
임베딩 클러스터링으로 "최종 승자 1개"를 고른다. 모델을 호출하지 않는 순수
로직만 담으므로(임베딩 벡터는 호출자가 만들어 넘긴다), 4-Tier 스트리밍 체인
바깥의 헬퍼로 둔다(설계 §2.2 — 체인 우회가 아님).

[핵심 관찰 — Wang et al. 2022]
오답은 흩어지고 정답은 수렴한다. 표면형 차이(콤마·마크다운·조사)를 정규화한
뒤 exact 다수결을 하면, 샘플링 노이즈로 자릿수/값이 틀어진 수치 오답을
과반의 정답이 눌러 이긴다. 단, 서술형 긴 답은 표면형이 전부 달라 exact
투표가 불가능하므로(설계 §3.3) 임베딩 클러스터 medoid를 차선책으로 쓴다.
이 방식은 "탈선 표본(완전히 다른 주제로 샌 표본)"만 걸러낼 뿐 세부 수치
오류는 못 잡는다 — 이 한계를 정직히 명시한다.

[제공 API]
  - normalize_answer(text)      : 투표 전 표면형 정규화(콤마/마크다운/조사 제거)
  - majority_vote(...)          : 짧은 사실형 답 — 정규화 exact 다수결
  - cluster_by_embedding(...)   : 서술형 긴 답 — 코사인 유사도 클러스터 medoid(폴백)
  - resolve_consensus(...)      : 위 둘을 자동 선택하는 디스패처
  - ConsensusResult             : 합의 결과 불변(frozen) dataclass

작성자: Nexus / 작성일: 2026-07-09
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ─────────────────────────────────────────────
# 정규화용 사전 컴파일 정규식 (hot path에서 매 호출 컴파일 방지)
# ─────────────────────────────────────────────
# 마크다운 굵게(**), 코드(`), 밑줄(__) 장식 문자.
_MD_DECORATION = re.compile(r"\*\*|__|`")
# 줄 맨 앞의 리스트 마커("- ", "* ", "1. " 등).
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
# 숫자 사이의 천단위 콤마(1,443 → 1443). 앞뒤가 모두 숫자일 때만 제거.
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d)")
# 문장 끝 조사/서술 어미 + 문장부호 꼬리("~입니다.", "~이다!" 등).
_TAIL_JOSA = re.compile(r"(?:입니다|이에요|예요|에요|이다|다)?[.?!,~\s]*$")
# 전각 숫자 → 반각 매핑 테이블.
_FULLWIDTH_DIGITS = {ord("０") + i: ord("0") + i for i in range(10)}


def normalize_answer(text: str) -> str:
    """
    투표 전 표면형 차이를 제거해 "같은 답"이 같은 문자열이 되게 한다(설계 §3.1).

    수행 단계(순서 중요):
      1. 앞뒤 공백·마크다운 장식(**, `, __)·리스트 마커 제거
      2. 전각 숫자 → 반각, 라틴 문자 소문자화
      3. 숫자 천단위 콤마 제거("5,100" → "5100")
      4. 근사 표현 "약" 제거, 내부 공백 전부 제거
      5. 문장 끝 조사/서술 어미·문장부호 꼬리 제거

    한국어 수사("오천만") 파싱은 과설계라 1차 구현 범위에서 제외한다(설계 §3.1).
    그래서 "5,100만 명"과 "약 5100만명"은 같은 값으로 정규화되지만("5100만명"),
    "오천백만"은 별개로 남는다 — 실패 시 문자열 비교로 자연 폴백된다.
    """
    s = text.strip()
    # 1) 마크다운 장식/리스트 마커 제거
    s = _LIST_MARKER.sub("", s)
    s = _MD_DECORATION.sub("", s)
    # 2) 전각 숫자 → 반각, 라틴 소문자화
    s = s.translate(_FULLWIDTH_DIGITS).lower()
    # 3) 천단위 콤마 제거
    s = _THOUSANDS_COMMA.sub("", s)
    # 4) 근사 표현 "약" 제거 + 내부 공백 전부 제거(표면형 통일)
    s = s.replace("약", "")
    s = re.sub(r"\s+", "", s)
    # 5) 문장 끝 조사/어미·부호 꼬리 제거
    s = _TAIL_JOSA.sub("", s)
    return s


@dataclass(frozen=True)
class ConsensusResult:
    """
    합의 결과 불변(frozen) 객체.

    필드:
      winner            : 최종 채택 텍스트(정규화 폼이 아니라 '원문' — 자연스러운 표출용).
      method            : "majority"(정규화 다수결) | "embedding"(클러스터 medoid) |
                          "fallback_first"(합의 실패 → 후보 0번 채택).
      agreement         : 최다 그룹/클러스터의 표 수.
      total             : 전체 표본 수.
      consensus_reached : min_agreement를 달성했는지 여부. False면 호출자가
                          SYSTEM_WARNING/로그로 "합의 실패"를 관측한다(설계 §3.4).
    """

    winner: str
    method: str
    agreement: int
    total: int
    consensus_reached: bool


def majority_vote(candidates: list[str], min_agreement: int) -> ConsensusResult:
    """
    짧은 사실형 답을 정규화 후 exact 다수결로 확정한다(설계 §3.2).

    동작:
      - 각 후보를 normalize_answer로 정규화해 같은 문자열끼리 그룹핑.
      - 최다 그룹의 표 수가 min_agreement 이상이면 그 그룹을 승자로 채택하고,
        그 그룹의 '원문 중 가장 긴 후보'를 winner로 돌려준다(정규화 폼이 아니라
        원문을 보여줘야 자연스럽기 때문).
      - min_agreement 미만(예: 3표가 전부 다름)이면 합의 실패 →
        후보 0번을 그대로 채택하고 consensus_reached=False로 표시(설계 §3.4).

    빈 리스트가 들어오면 빈 승자 + 실패로 방어 반환한다(호출자 크래시 방지).
    """
    total = len(candidates)
    if total == 0:
        return ConsensusResult("", "fallback_first", 0, 0, False)

    # 정규화 폼 → 그 그룹에 속한 원문 후보 리스트(원문 순서 보존).
    groups: dict[str, list[str]] = {}
    # 정규화 폼의 '최초 등장 순서'를 기록 — 동률일 때 먼저 나온 그룹을 우선한다.
    first_seen: dict[str, int] = {}
    for i, cand in enumerate(candidates):
        key = normalize_answer(cand)
        if key not in groups:
            groups[key] = []
            first_seen[key] = i
        groups[key].append(cand)

    # 최다 그룹 선택 — 표 수 내림차순, 동률이면 먼저 등장한 그룹.
    best_key = max(
        groups,
        key=lambda k: (len(groups[k]), -first_seen[k]),
    )
    best_members = groups[best_key]
    agreement = len(best_members)

    if agreement >= min_agreement:
        # 승자 그룹의 원문 중 가장 긴 것을 표출용으로 채택(길이 동률이면 먼저 나온 것).
        winner = max(best_members, key=len)
        return ConsensusResult(winner, "majority", agreement, total, True)

    # 합의 실패 — 후보 0번 채택(재샘플링 금지, 설계 §3.4).
    return ConsensusResult(candidates[0], "fallback_first", agreement, total, False)


def _cosine(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도. 영벡터/차원 불일치는 0.0으로 방어."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cluster_by_embedding(
    candidates: list[str],
    embeddings: list[list[float]],
    min_agreement: int,
    similarity_threshold: float = 0.90,
) -> ConsensusResult:
    """
    서술형 긴 답의 차선책 — 임베딩 코사인 유사도로 최대 클러스터의 medoid 채택(설계 §3.3).

    ★정직한 한계★: 이 방식이 잡는 것은 "한 표본만 완전히 다른 주제로 샌 경우"
    (탈선 표본 배제)까지다. 문장 안의 개별 수치 오류(연도 하나 틀림)는 문장
    전체 유사도에 묻혀 걸러지지 않는다. 그래서 기본 정책은 G3 게이트로 서술형을
    아예 SC 대상에서 빼는 것이고(설계 §1.2), 이 경로는 실험용 폴백일 뿐이다.

    동작:
      - 후보 i를 기준으로 similarity_threshold 이상인 후보들을 한 클러스터로 묶어,
        가장 큰 클러스터를 고른다(그리디 — 각 후보 기준 이웃 수 최대).
      - 그 클러스터 안에서 다른 후보들과의 평균 유사도가 최대인 후보(medoid)를 채택.
      - 클러스터 크기가 min_agreement 이상이면 합의 성공으로 본다.

    임베딩이 후보와 개수가 안 맞으면(호출 실패 등) majority_vote로 폴백한다.
    """
    total = len(candidates)
    if total == 0:
        return ConsensusResult("", "fallback_first", 0, 0, False)
    # 임베딩 개수 불일치 → 안전하게 다수결로 폴백(임베딩 서버 부분 실패 방어).
    if len(embeddings) != total:
        return majority_vote(candidates, min_agreement)

    # 후보별 유사 이웃 집합(자기 자신 포함)을 구한다.
    neighbors: list[list[int]] = []
    for i in range(total):
        group = [
            j
            for j in range(total)
            if i == j or _cosine(embeddings[i], embeddings[j]) >= similarity_threshold
        ]
        neighbors.append(group)

    # 가장 큰 이웃 집합을 대표 클러스터로 삼는다(동률이면 먼저 나온 것).
    best_i = max(range(total), key=lambda i: len(neighbors[i]))
    cluster = neighbors[best_i]
    agreement = len(cluster)

    # 클러스터 내부에서 평균 유사도가 최대인 후보(medoid)를 승자로.
    def _avg_sim(idx: int) -> float:
        others = [j for j in cluster if j != idx]
        if not others:
            return 0.0
        return sum(_cosine(embeddings[idx], embeddings[j]) for j in others) / len(others)

    medoid = max(cluster, key=_avg_sim)
    reached = agreement >= min_agreement
    method = "embedding" if reached else "fallback_first"
    winner = candidates[medoid] if reached else candidates[0]
    return ConsensusResult(winner, method, agreement, total, reached)


def resolve_consensus(
    candidates: list[str],
    *,
    min_agreement: int,
    short_answer_max_chars: int,
    embeddings: list[list[float]] | None = None,
    similarity_threshold: float = 0.90,
) -> ConsensusResult:
    """
    후보 특성에 따라 다수결/임베딩 클러스터를 자동 선택하는 디스패처(설계 §3.2~3.3).

    선택 규칙:
      - 후보 전원이 short_answer_max_chars 이하 → majority_vote(정규화 exact 다수결).
        (사실형 짧은답의 기본 경로 — 임베딩 호출 없이 순수 계산)
      - 하나라도 초과(서술형) + embeddings 제공 → cluster_by_embedding(폴백 경로).
      - 서술형인데 embeddings 미제공 → majority_vote로 폴백(정규화가 우연히
        일치하면 다수결, 아니면 후보 0번).

    임베딩 계산(model_provider.embed)은 호출자(query_loop)가 수행해 넘긴다 —
    이 모듈은 순수 로직만 유지해 체인 밖 헬퍼로 남기 위함이다.
    """
    if not candidates:
        return ConsensusResult("", "fallback_first", 0, 0, False)

    all_short = all(len(c) <= short_answer_max_chars for c in candidates)
    if all_short:
        return majority_vote(candidates, min_agreement)
    if embeddings is not None:
        return cluster_by_embedding(
            candidates, embeddings, min_agreement, similarity_threshold
        )
    return majority_vote(candidates, min_agreement)
