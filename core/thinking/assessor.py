"""
복잡도 평가기(ComplexityAssessor) — 사용자 입력이 얼마나 "어려운" 요청인지
0.0~1.0 사이의 숫자 하나로 정량화하는 모듈이다.

왜 필요한가:
  Nexus는 모든 요청에 똑같은 깊이로 사고하지 않는다. 간단한 질문에 값비싼
  다중 에이전트 추론을 돌리면 시간·토큰이 낭비되고, 반대로 어려운 설계 작업에
  단순 응답을 하면 품질이 떨어진다. 그래서 "먼저 난이도를 재고 → 거기에 맞는
  사고 전략을 고르는" 방식을 쓴다. 이 파일은 그 첫 단계인 "난이도 측정"을 맡는다.

어떻게 측정하나 (4가지 신호를 조합):
  1. 키워드 매칭   — "architecture", "리팩토링" 같은 단어에 미리 정한 가중치를 더함
  2. 메시지 길이   — 긴 요청일수록 요구사항이 복잡할 확률이 높음
  3. 코드 블록 유무 — 코드가 붙어 있으면 기술적 분석이 필요함
  4. 증폭 패턴     — "multi-file", "전면 재작성" 등 범위를 키우는 표현은 곱셈으로 증폭
  (+ 이전 대화가 길면 맥락이 복잡하다고 보고 약간 가산)

이 스코어의 소비처:
  결과 스코어는 상위의 ThinkingStrategy 선택 로직으로 전달되어 다음처럼 쓰인다.
  (구간 경계값은 이 파일이 아니라 전략 선택 쪽에서 정의·사용한다)
    - < 0.3  → DIRECT       (단순 응답)
    - < 0.6  → HIDDEN_COT   (2-pass 분석)
    - < 0.8  → SELF_REFLECT (3-pass 검증)
    - >= 0.8 → MULTI_AGENT  (다중 에이전트)

주요 구성:
  - ComplexityAssessor            : 유일한 공개 클래스
  - ComplexityAssessor.assess()   : 외부에서 호출하는 메인 진입점
  - _score_keywords / _score_length / _has_code_block /
    _calculate_amplifier / _score_context : assess()가 쓰는 내부 헬퍼들

의존:
  - core.message.Message (대화 컨텍스트 타입 힌트에만 사용, 실제 필드 접근은 없음)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re

from core.message import Message

logger = logging.getLogger("nexus.thinking.assessor")


class ComplexityAssessor:
    """
    사용자 입력의 복잡도를 0.0~1.0 스코어로 평가하는 클래스.

    상태(인스턴스 변수)를 갖지 않는 순수 계산기라, 한 번 만들어 두고 여러 요청에
    재사용해도 안전하다. 판단 기준이 되는 값들(키워드 사전, 증폭 패턴, 길이 임계값)은
    모두 아래의 클래스 변수로 선언되어 있어, 튜닝할 때 이 표만 고치면 된다.

    평가 기준 5가지 (자세한 계산은 assess() 참고):
      1. 키워드 매칭 — 특정 작업 유형 키워드의 가중치를 더함 (덧셈)
      2. 메시지 길이 — 긴 메시지는 복잡한 요구사항일 가능성이 높음 (덧셈)
      3. 코드 블록   — 코드가 포함되면 기술적 분석이 필요함 (덧셈)
      4. 증폭 패턴   — "multi-file", "across modules" 등 복합 작업 표현 (곱셈)
      5. 대화 컨텍스트 — 이전 대화가 길수록 복잡한 맥락임 (덧셈)
    """

    # ─── 키워드 사전: 작업 유형별 복잡도 가중치 ───
    # 요청 문장에 이 단어가 (부분 문자열로) 들어 있으면 해당 가중치를 더한다.
    # 한글/영어를 나란히 등록해 어느 언어로 써도 잡히게 했다. 가중치가 높을수록
    # "본질적으로 어려운 작업"이라는 뜻. 값을 조정하면 전체 난이도 감이 바뀐다.
    COMPLEXITY_KEYWORDS: dict[str, float] = {
        # 높은 복잡도 (0.3~0.4) — 구조적 변경, 설계 작업
        "architecture": 0.4,
        "아키텍처": 0.4,
        "refactor": 0.3,
        "리팩토링": 0.3,
        "리팩터링": 0.3,
        "debug": 0.3,
        "디버그": 0.3,
        "디버깅": 0.3,
        "migrate": 0.3,
        "마이그레이션": 0.3,
        "optimize": 0.3,
        "최적화": 0.3,
        "design": 0.3,
        "설계": 0.3,
        "performance": 0.3,
        "성능": 0.3,
        # 중간 복잡도 (0.2) — 구현 작업
        "implement": 0.2,
        "구현": 0.2,
        "fix": 0.2,
        "수정": 0.2,
        "버그": 0.2,
        "bug": 0.2,
        "integrate": 0.2,
        "통합": 0.2,
        "test": 0.2,
        "테스트": 0.2,
        "security": 0.2,
        "보안": 0.2,
        "concurrent": 0.2,
        "동시성": 0.2,
        "async": 0.2,
        "비동기": 0.2,
        # 낮은 복잡도 (0.1) — 단순 작업
        "create": 0.1,
        "생성": 0.1,
        "add": 0.1,
        "추가": 0.1,
        "update": 0.1,
        "변경": 0.1,
        "remove": 0.1,
        "삭제": 0.1,
        "rename": 0.1,
        "explain": 0.1,
        "설명": 0.1,
        "read": 0.05,
        "확인": 0.05,
    }

    # ─── 증폭 패턴: 복합 작업을 나타내는 표현 ───
    # 여기 등록된 정규식이 문장에 걸리면 "이미 계산된 점수를 배수로 곱해" 키운다.
    # 키워드가 '무엇을' 하는지라면, 증폭 패턴은 '얼마나 넓은 범위로' 하는지를 잡는다.
    # 예: "리팩토링"(0.3) 자체는 중간이지만 "전체 모듈 전면 재작성"이면 범위가 폭발한다.
    # 형식은 (정규식 패턴, 증폭 배수) 튜플. 여러 개가 걸려도 곱하지 않고 최댓값만 쓴다
    # (_calculate_amplifier 참고).
    AMPLIFIER_PATTERNS: list[tuple[str, float]] = [
        (r"multi[- ]?file", 1.4),
        (r"여러\s*파일", 1.4),
        (r"across\s+(modules?|files?|packages?)", 1.3),
        (r"전체\s*(모듈|파일|패키지)", 1.3),
        (r"integration", 1.3),
        (r"end[- ]?to[- ]?end", 1.3),
        (r"from\s+scratch", 1.2),
        (r"처음부터", 1.2),
        (r"전면\s*재작성", 1.5),
        (r"rewrite", 1.4),
        (r"모든\s*(파일|모듈|컴포넌트)", 1.3),
        (r"시스템\s*전체", 1.4),
        (r"backward[- ]?compat", 1.2),
        (r"하위\s*호환", 1.2),
        (r"thread[- ]?safe", 1.2),
        (r"스레드\s*안전", 1.2),
    ]

    # ─── 메시지 길이 기반 스코어 보정 상수 ───
    # 긴 메시지일수록 복잡한 요구사항일 확률이 높다는 경험칙을 반영한다.
    # (글자수 임계값, 더할 보너스) 튜플을 '큰 값부터' 나열해 둔 점이 중요하다.
    # _score_length가 위에서부터 훑다가 처음 넘어서는 구간의 보너스 하나만 쓰고
    # 멈추기 때문에, 순서가 뒤바뀌면 항상 작은 보너스만 걸리는 버그가 생긴다.
    _LENGTH_THRESHOLDS: list[tuple[int, float]] = [
        (500, 0.15),  # 500자 이상: +0.15
        (200, 0.10),  # 200자 이상: +0.10
        (100, 0.05),  # 100자 이상: +0.05
    ]

    def assess(
        self,
        message: str,
        context: list[Message] | None = None,
    ) -> float:
        """
        입력을 받아 복잡도 스코어(0.0~1.0)를 계산해 돌려주는 메인 진입점.

        외부에서 실제로 호출하는 유일한 메서드다. 아래 6단계를 순서대로 밟는데,
        핵심은 "덧셈으로 기본 점수를 쌓은 뒤(1~3, 5단계), 증폭 패턴만 곱셈으로
        적용(4단계)"한다는 점이다. 그래서 4단계의 위치(덧셈 이후, 컨텍스트 이전)가
        결과에 영향을 준다 — 순서를 함부로 바꾸지 말 것.

        Args:
            message: 사용자 입력 텍스트(원문). 비어 있거나 공백뿐이면 0.0을 반환.
            context: 이전 대화 메시지 목록. None이면 컨텍스트 보정을 건너뛴다.

        Returns:
            0.0(매우 단순) ~ 1.0(매우 복잡) 사이로 클램프된 float 스코어.

        평가 절차:
          1. 키워드 매칭 → 기본 스코어 합산      (_score_keywords)
          2. 메시지 길이 보정 (덧셈)             (_score_length)
          3. 코드 블록이 있으면 +0.1            (_has_code_block)
          4. 증폭 패턴 배수 적용 (곱셈)          (_calculate_amplifier)
          5. 대화 컨텍스트가 있으면 보정 (덧셈)   (_score_context)
          6. 마지막에 0.0~1.0 범위로 잘라냄(clamp)
        """
        # 빈 메시지(또는 공백만 있는 경우)는 잴 것이 없으므로 곧장 최소값 0.0 반환.
        if not message or not message.strip():
            return 0.0

        # 키워드/증폭 패턴 매칭은 소문자 기준으로 한다. 한글에는 대소문자가 없지만,
        # 영어 단어가 대문자로 섞여 들어와도("Refactor") 놓치지 않게 통일해 둔다.
        # 주의: 길이·코드블록 판정은 원문(message)을 그대로 써야 하므로 여기서
        #       lower_msg만 따로 만든다(원문을 덮어쓰지 않는다).
        lower_msg = message.lower()

        # ── 1단계: 키워드 매칭 → 기본 스코어 ──
        # 여기서 나온 값이 이후 곱셈(4단계)의 밑바탕이 된다.
        score = self._score_keywords(lower_msg)

        # ── 2단계: 메시지 길이 보정 (원문 길이 기준으로 덧셈) ──
        score += self._score_length(message)

        # ── 3단계: 코드 블록(```)이 있으면 +0.1 가산 ──
        if self._has_code_block(message):
            score += 0.1

        # ── 4단계: 증폭 패턴 적용 (여기까지 쌓인 점수에 배수를 곱함) ──
        # 곱셈이라, 앞에서 점수가 0이면 아무리 증폭 표현이 있어도 0에 머문다.
        amplifier = self._calculate_amplifier(lower_msg)
        score *= amplifier

        # ── 5단계: 대화 컨텍스트 보정 (이전 대화가 길면 맥락이 복잡하다고 보고 가산) ──
        if context:
            score += self._score_context(context)

        # ── 6단계: 최종적으로 0.0~1.0 범위 밖으로 튀지 않도록 잘라낸다 ──
        result = max(0.0, min(1.0, score))

        # 디버깅용 로그: 최종 점수와 함께 각 단계가 얼마나 기여했는지 남긴다.
        # 튜닝할 때 "어느 신호 때문에 점수가 튀었는지" 추적하는 용도라, 여기서
        # 헬퍼들을 다시 호출(재계산)하는 비용은 감수한다(디버그 레벨이라 평소엔 꺼짐).
        logger.debug(
            f"복잡도 평가: score={result:.2f}, "
            f"keywords={self._score_keywords(lower_msg):.2f}, "
            f"length={self._score_length(message):.2f}, "
            f"amplifier={amplifier:.2f}, "
            f"context_bonus={self._score_context(context) if context else 0:.2f}"
        )

        return result

    def _score_keywords(self, lower_msg: str) -> float:
        """
        키워드 사전을 훑어 매칭되는 단어들의 가중치를 모두 더한 기본 스코어를 낸다.

        중요한 규칙: 같은 키워드가 문장에 여러 번 나와도 한 번만 센다. 예를 들어
        "리팩토링 리팩토링 리팩토링"이라고 해서 0.9가 되지는 않고 0.3에 그친다.
        (단어 반복만으로 난이도를 부풀리는 것을 막기 위함.) 반대로 서로 다른
        키워드가 여러 개 걸리면 각각의 가중치가 정상적으로 합산된다.

        Args:
            lower_msg: 이미 소문자로 변환된 입력 문장.

        Returns:
            매칭된 서로 다른 키워드들의 가중치 합(0.0 이상).
        """
        score = 0.0
        matched: set[str] = set()  # 이미 점수에 반영한 키워드를 기억(중복 방지)
        for keyword, weight in self.COMPLEXITY_KEYWORDS.items():
            # 이미 매칭된 키워드는 건너뛴다 (중복 방지)
            if keyword in matched:
                continue
            # 부분 문자열 포함 검사 — 단어 경계를 따지지 않는 단순 in 매칭이다.
            if keyword in lower_msg:
                score += weight
                matched.add(keyword)
        return score

    def _score_length(self, message: str) -> float:
        """
        메시지 글자수에 따라 길이 보너스를 반환한다(가장 큰 구간 하나만).

        _LENGTH_THRESHOLDS가 큰 임계값부터 정렬돼 있어서, 위에서부터 내려오며
        처음으로 length가 넘어서는 구간을 만나면 그 보너스를 즉시 반환하고 끝낸다.
        즉 보너스는 '누적'이 아니라 '한 개'만 적용된다. 예: 250자 → +0.10 (500 구간은
        못 넘고 200 구간을 넘으므로). 어떤 구간에도 못 미치면 0.0.

        Args:
            message: 원문 메시지(소문자 변환 전, 실제 글자수가 필요하므로 원문 사용).

        Returns:
            해당 길이 구간의 보너스, 또는 100자 미만이면 0.0.
        """
        length = len(message)
        for threshold, bonus in self._LENGTH_THRESHOLDS:
            if length >= threshold:
                return bonus
        return 0.0

    @staticmethod
    def _has_code_block(message: str) -> bool:
        """
        메시지에 마크다운 코드 펜스(```)가 들어 있는지 검사한다.

        여는/닫는 쌍을 엄밀히 따지지 않고, 백틱 3개가 한 번이라도 등장하면 True로 본다.
        코드가 붙어 있다는 건 대개 "이 코드를 봐 달라/고쳐 달라"는 기술적 요청이라
        복잡도를 한 단계 올릴 근거가 된다. 상태가 필요 없어 staticmethod로 둔다.

        Args:
            message: 원문 메시지(코드 펜스는 원문에서 찾아야 하므로 원문 사용).

        Returns:
            코드 펜스가 하나라도 있으면 True, 없으면 False.
        """
        return "```" in message

    def _calculate_amplifier(self, lower_msg: str) -> float:
        """
        증폭 패턴들을 모두 검사해 '가장 큰' 증폭 배수 하나를 반환한다.

        여러 패턴이 동시에 걸려도 배수를 서로 곱하지 않고 최댓값만 쓴다. 이렇게 하면
        표현을 여러 개 나열했다고 점수가 기하급수로 폭발하는 것을 막을 수 있다.
        아무 패턴도 안 걸리면 1.0(= 증폭 없음, 곱해도 점수 불변)을 돌려준다.

        Args:
            lower_msg: 소문자로 변환된 입력 문장. (re.IGNORECASE도 함께 줘서
                       혹시 모를 대소문자 차이에 이중으로 안전하게 대비한다.)

        Returns:
            1.0 이상의 곱셈 배수. 1.0이면 증폭 없음.
        """
        max_amplifier = 1.0
        for pattern, multiplier in self.AMPLIFIER_PATTERNS:
            # 부분 매칭(re.search)이면 충분 — 문장 어디든 패턴이 있으면 걸린다.
            if re.search(pattern, lower_msg, re.IGNORECASE):
                max_amplifier = max(max_amplifier, multiplier)
        return max_amplifier

    @staticmethod
    def _score_context(context: list[Message]) -> float:
        """
        지금까지 오간 대화 턴 수에 따라 컨텍스트 보너스를 반환한다.

        대화가 길게 이어졌다는 건 그만큼 얽힌 맥락 위에서 작업 중이라는 신호라,
        같은 요청이라도 난이도를 조금 더 높게 본다. 아래처럼 '가장 높은 구간 하나'만
        적용하며(누적 아님), 5턴 미만이면 0.0이다. turn_count는 context 리스트의
        길이(메시지 개수)로 단순 계산한다.

        Args:
            context: 이전 대화 메시지 목록(비어 있지 않다고 가정하고 호출됨).

        Returns:
            - 20턴 이상: +0.2
            - 10턴 이상: +0.15
            - 5턴 이상:  +0.1
            - 그 미만:   0.0
        """
        turn_count = len(context)
        if turn_count >= 20:
            return 0.2
        elif turn_count >= 10:
            return 0.15
        elif turn_count >= 5:
            return 0.1
        return 0.0
