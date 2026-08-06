// ⭐ 이 파일의 SITE 객체만 수정하면 신청서 전체 내용과 색상이 바뀝니다.
//    index.html·styles.css는 열지 마세요 — 수정할 것이 없습니다.
const SITE = {
  // 색상 — 아래 값만 바꾸면 전체 배색이 바뀝니다(CSS 파일은 건드리지 않습니다).
  colors: {
    brand: "#4338ca",    // 주 브랜드 색(버튼·포커스)
    brand2: "#6366f1",   // 보조 색
    ink: "#111827",      // 본문 글자색
    muted: "#6b7280",    // 흐린 글자색
    bg: "#f5f6fa",       // 페이지 배경
    surface: "#ffffff",  // 카드 배경
    line: "#e5e7eb",     // 구분선
    bad: "#dc2626",      // 오류 표시
  },

  formTitle: "서비스 이용 신청서",
  formSubtitle: "아래 항목을 채워 제출해 주세요. * 표시는 필수입니다.",
  submitText: "신청서 제출",
  doneTitle: "제출이 완료되었습니다",
  doneText: "담당자가 확인 후 연락드립니다. 아래는 제출하신 내용입니다.",
  resetText: "새 신청서 작성",

  // 폼 구성 — sections 안의 fields 만 채우면 화면·검증·요약이 자동으로 만들어집니다.
  // field.type: "text" | "email" | "tel" | "number" | "date" | "textarea" | "select" | "checkbox"
  // select 는 options 배열이 필요합니다. required: true 면 필수 항목입니다.
  sections: [
    {
      title: "신청자 정보",
      fields: [
        { key: "name", label: "이름", type: "text", required: true, placeholder: "홍길동" },
        { key: "org", label: "소속", type: "text", required: true, placeholder: "○○부서" },
        { key: "email", label: "이메일", type: "email", required: true, placeholder: "name@example.com" },
        { key: "phone", label: "연락처", type: "tel", placeholder: "010-0000-0000" },
      ],
    },
    {
      title: "신청 내용",
      fields: [
        { key: "plan", label: "이용 구분", type: "select", required: true,
          options: ["신규 이용", "이용 연장", "인원 추가"] },
        { key: "seats", label: "이용 인원", type: "number", required: true, placeholder: "10" },
        { key: "startDate", label: "시작 희망일", type: "date", required: true },
        { key: "purpose", label: "이용 목적", type: "textarea", placeholder: "어떤 업무에 사용할 예정인지 적어 주세요." },
      ],
    },
    {
      title: "동의",
      fields: [
        { key: "agree", label: "개인정보 수집·이용에 동의합니다.", type: "checkbox", required: true },
      ],
    },
  ],

  footer: "© 2026 IDINO. All rights reserved.",
};

// ── 아래는 구조 코드 — 수정 금지 ──

// SITE.colors 를 CSS 변수로 주입한다.
// 색을 styles.css 에 두면 값을 '교체'하는 대신 새 :root 를 덧붙이는 실수가 잦다.
// CSS 는 뒤 규칙이 이기므로 옛 색이 남는다. 인라인 스타일은 스타일시트보다 우선한다.
const CSS_VAR_BY_KEY = {
  brand: "--brand", brand2: "--brand-2", ink: "--ink",
  muted: "--muted", bg: "--bg", surface: "--surface", line: "--line", bad: "--bad",
};
for (const [key, value] of Object.entries(SITE.colors || {})) {
  const cssVar = CSS_VAR_BY_KEY[key];
  if (cssVar && value) document.documentElement.style.setProperty(cssVar, value);
}

const { createApp } = Vue;

// SITE 정의에서 모든 필드를 한 줄로 펼친다(검증·요약에서 반복해 쓴다).
const ALL_FIELDS = SITE.sections.flatMap((s) => s.fields);

createApp({
  data() {
    // 필드 종류에 맞는 빈 값으로 폼 상태를 만든다(체크박스만 false).
    const form = {};
    for (const f of ALL_FIELDS) form[f.key] = f.type === "checkbox" ? false : "";
    return { site: SITE, form, errors: {}, submitted: false };
  },
  methods: {
    // 필수 항목만 검사한다. 체크박스는 '체크됨'이어야 통과.
    validate() {
      const errors = {};
      for (const f of ALL_FIELDS) {
        if (!f.required) continue;
        const v = this.form[f.key];
        const empty = f.type === "checkbox" ? !v : !String(v).trim();
        if (empty) errors[f.key] = "필수 항목입니다.";
      }
      this.errors = errors;
      return Object.keys(errors).length === 0;
    },
    submit() {
      if (!this.validate()) {
        // 첫 오류 항목으로 스크롤해 사용자가 무엇을 놓쳤는지 바로 보이게 한다.
        const first = ALL_FIELDS.find((f) => this.errors[f.key]);
        if (first) {
          const el = document.getElementById("f-" + first.key);
          if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        return;
      }
      this.submitted = true;
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    reset() {
      for (const f of ALL_FIELDS) this.form[f.key] = f.type === "checkbox" ? false : "";
      this.errors = {};
      this.submitted = false;
    },
    // 제출 요약에 쓸 (라벨, 값) 목록. 빈 값은 건너뛴다.
    summary() {
      return ALL_FIELDS
        .map((f) => ({
          label: f.label,
          value: f.type === "checkbox" ? (this.form[f.key] ? "동의함" : "") : this.form[f.key],
        }))
        .filter((row) => String(row.value).trim() !== "");
    },
  },
}).mount("#app");
