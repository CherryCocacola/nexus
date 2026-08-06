// ⭐ 이 파일의 SITE 객체만 수정하면 페이지 전체 내용과 색상이 바뀝니다.
//    index.html·styles.css는 열지 마세요 — 수정할 것이 없습니다.
const SITE = {
  // 색상 — 아래 값만 바꾸면 페이지 전체 배색이 바뀝니다(CSS 파일은 건드리지 않습니다).
  colors: {
    brand: "#4f46e5",    // 주 브랜드 색(버튼·강조)
    brand2: "#7c3aed",   // 보조 브랜드 색(그라데이션 끝)
    ink: "#1f2937",      // 본문 글자색
    muted: "#6b7280",    // 흐린 글자색
    bg: "#f9fafb",       // 페이지 배경
    surface: "#ffffff",  // 카드/패널 배경
    line: "#e5e7eb",     // 구분선
  },
  brandName: "IDINO NOVA",
  heroBadge: "On-Premise · Air-gapped",
  tagline: "에어갭 환경을 위한 완전 자율 AI 플랫폼",
  ctaText: "도입 문의",
  features: [
    { icon: "📄", title: "문서 생성", desc: "보고서·제안서를 docx/pptx/hwpx로 자동 생성합니다." },
    { icon: "🎨", title: "이미지 생성", desc: "텍스트 프롬프트로 고품질 이미지를 만듭니다." },
    { icon: "👁️", title: "비전 분석", desc: "이미지 속 객체·표·문자를 이해하고 설명합니다." },
    { icon: "📚", title: "지식 RAG 검색", desc: "사내 지식베이스에서 근거 기반으로 답합니다." },
  ],
  specs: [
    { label: "AI 모델", value: "A.X-4.0 72B" },
    { label: "GPU", value: "2×B200" },
    { label: "네트워크", value: "에어갭(폐쇄망) 지원" },
  ],
  contactTitle: "문의하기",
  footer: "© 2026 IDINO. All rights reserved.",
};

// ── 아래는 구조 코드 — 수정 금지 ──

// SITE.colors 를 CSS 변수로 주입한다.
// 왜 JS에서 주입하나: 색을 styles.css의 :root에 두면, 값을 '교체'하는 대신 새 :root
// 블록을 위에 덧붙이는 실수가 잦았다. CSS는 뒤에 오는 규칙이 이기므로 옛 색이 그대로
// 남아 "바꿨는데 안 바뀐다"가 된다. 여기서 documentElement 인라인 스타일로 넣으면
// 스타일시트 규칙보다 우선하므로 항상 SITE 값이 이긴다.
// (JS가 실패해도 styles.css의 :root 기본값이 남아 화면은 깨지지 않는다.)
const CSS_VAR_BY_KEY = {
  brand: "--brand", brand2: "--brand-2", ink: "--ink",
  muted: "--muted", bg: "--bg", surface: "--surface", line: "--line",
};
for (const [key, value] of Object.entries(SITE.colors || {})) {
  const cssVar = CSS_VAR_BY_KEY[key];
  if (cssVar && value) document.documentElement.style.setProperty(cssVar, value);
}

const { createApp } = Vue;

createApp({
  data() {
    return {
      site: SITE,
      form: { name: "", email: "", message: "" },
    };
  },
  methods: {
    submitForm() {
      alert(`문의가 접수되었습니다.\n이름: ${this.form.name}\n이메일: ${this.form.email}`);
      this.form = { name: "", email: "", message: "" };
    },
  },
}).mount("#app");
