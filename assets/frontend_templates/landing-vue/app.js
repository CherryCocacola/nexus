// ⭐ 이 파일의 SITE 객체만 수정하면 페이지 전체 내용이 바뀝니다.
//    (index.html·styles.css의 구조 코드는 수정하지 마세요. 색상만 styles.css 맨 위 :root에서.)
const SITE = {
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
