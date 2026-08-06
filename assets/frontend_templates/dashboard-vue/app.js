// ⭐ 이 파일의 SITE 객체만 수정하면 대시보드 전체 내용과 색상이 바뀝니다.
//    index.html·styles.css는 열지 마세요 — 수정할 것이 없습니다.
const SITE = {
  // 색상 — 아래 값만 바꾸면 전체 배색이 바뀝니다(CSS 파일은 건드리지 않습니다).
  colors: {
    brand: "#2563eb",    // 주 브랜드 색(활성 메뉴·강조)
    brand2: "#0ea5e9",   // 보조 색(그라데이션·차트 막대)
    ink: "#111827",      // 본문 글자색
    muted: "#6b7280",    // 흐린 글자색
    bg: "#f3f4f6",       // 페이지 배경
    surface: "#ffffff",  // 카드/패널 배경
    line: "#e5e7eb",     // 구분선
    ok: "#16a34a",       // 상승·정상
    warn: "#d97706",     // 주의
    bad: "#dc2626",      // 하락·오류
  },

  brandName: "운영 대시보드",
  subtitle: "실시간 서비스 현황",
  // 좌측 메뉴 — active: true 인 항목이 현재 화면으로 표시됩니다.
  menu: [
    { icon: "📊", label: "개요", active: true },
    { icon: "🧾", label: "요청 로그" },
    { icon: "👥", label: "사용자" },
    { icon: "⚙️", label: "설정" },
  ],
  // 상단 지표 카드 — delta 는 증감 표시(양수면 상승색, 음수면 하락색).
  kpis: [
    { label: "오늘 요청", value: "12,480", delta: "+8.2%", tone: "ok" },
    { label: "평균 응답", value: "1.24초", delta: "-0.11초", tone: "ok" },
    { label: "오류율", value: "0.42%", delta: "+0.05%", tone: "bad" },
    { label: "활성 사용자", value: "1,932", delta: "+126", tone: "ok" },
  ],
  // 가로 막대 — value 는 0~100 사이 백분율입니다.
  usageTitle: "기능별 사용 비중",
  usage: [
    { label: "문서 분석", value: 42 },
    { label: "지식 검색", value: 27 },
    { label: "문서 생성", value: 19 },
    { label: "이미지 생성", value: 12 },
  ],
  // 표 — columns 순서대로 rows 의 값이 들어갑니다.
  tableTitle: "최근 처리 내역",
  columns: ["시각", "사용자", "작업", "상태"],
  rows: [
    ["09:41", "김민수", "입찰공고문.hwp 분석", "완료"],
    ["09:38", "이서연", "월간 보고서 생성", "완료"],
    ["09:35", "박지훈", "지식 검색", "완료"],
    ["09:31", "최수빈", "이미지 생성", "실패"],
  ],
  // 상태 문자열이 아래 목록에 있으면 경고색으로 표시됩니다.
  badStatuses: ["실패", "오류", "중단"],
  footer: "© 2026 IDINO. All rights reserved.",
};

// ── 아래는 구조 코드 — 수정 금지 ──

// SITE.colors 를 CSS 변수로 주입한다.
// 색을 styles.css 에 두면 값을 '교체'하는 대신 새 :root 를 덧붙이는 실수가 잦다.
// CSS 는 뒤 규칙이 이기므로 옛 색이 남아 "바꿨는데 안 바뀐다"가 된다. 인라인 스타일은
// 스타일시트보다 우선하므로 항상 SITE 값이 이긴다.
const CSS_VAR_BY_KEY = {
  brand: "--brand", brand2: "--brand-2", ink: "--ink", muted: "--muted",
  bg: "--bg", surface: "--surface", line: "--line",
  ok: "--ok", warn: "--warn", bad: "--bad",
};
for (const [key, value] of Object.entries(SITE.colors || {})) {
  const cssVar = CSS_VAR_BY_KEY[key];
  if (cssVar && value) document.documentElement.style.setProperty(cssVar, value);
}

const { createApp } = Vue;

createApp({
  data() {
    return { site: SITE };
  },
  methods: {
    // 표의 상태 칸에 색을 입힌다(정상/경고 구분).
    statusClass(value) {
      return (SITE.badStatuses || []).includes(String(value)) ? "pill pill-bad" : "pill pill-ok";
    },
    // 마지막 열만 상태로 취급한다.
    isStatusCell(index) {
      return index === (SITE.columns.length - 1);
    },
  },
}).mount("#app");
