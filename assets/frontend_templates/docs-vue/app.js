// ⭐ 이 파일의 SITE 객체만 수정하면 문서 전체 내용과 색상이 바뀝니다.
//    index.html·styles.css는 열지 마세요 — 수정할 것이 없습니다.
const SITE = {
  // 색상 — 아래 값만 바꾸면 전체 배색이 바뀝니다(CSS 파일은 건드리지 않습니다).
  colors: {
    brand: "#0f766e",    // 주 브랜드 색(활성 목차·링크)
    brand2: "#14b8a6",   // 보조 색
    ink: "#111827",      // 본문 글자색
    muted: "#6b7280",    // 흐린 글자색
    bg: "#ffffff",       // 페이지 배경
    surface: "#f8fafc",  // 코드·노트 배경
    line: "#e5e7eb",     // 구분선
  },

  docTitle: "IDINO NOVA 사용자 매뉴얼",
  docSubtitle: "v1.0 · 2026-08",
  tocTitle: "목차",

  // 문서 본문 — sections 배열 하나만 채우면 목차가 자동으로 만들어집니다.
  // blocks 의 종류:
  //   { type: "p",    text: "단락" }
  //   { type: "list", items: ["항목1", "항목2"] }
  //   { type: "steps", items: ["1단계", "2단계"] }     번호가 붙습니다
  //   { type: "code", text: "명령어" }
  //   { type: "note", text: "강조할 안내" }
  //   { type: "table", columns: ["열1","열2"], rows: [["a","b"]] }
  sections: [
    {
      id: "intro",
      title: "1. 시작하기",
      blocks: [
        { type: "p", text: "이 문서는 서비스의 기본 사용법을 안내합니다. 처음이라면 순서대로 읽어 주세요." },
        { type: "note", text: "사내망에서만 접속할 수 있습니다. 외부망에서는 열리지 않습니다." },
      ],
    },
    {
      id: "install",
      title: "2. 접속 방법",
      blocks: [
        { type: "steps", items: [
          "브라우저에서 서비스 주소를 엽니다.",
          "발급받은 계정으로 로그인합니다.",
          "왼쪽 메뉴에서 사용할 기능을 고릅니다.",
        ]},
        { type: "code", text: "http://192.168.0.10:8600" },
      ],
    },
    {
      id: "features",
      title: "3. 주요 기능",
      blocks: [
        { type: "p", text: "자주 쓰는 기능은 다음과 같습니다." },
        { type: "list", items: [
          "문서 분석 — PDF·Word·한글 파일을 올려 내용을 확인합니다.",
          "문서 생성 — 정리한 내용을 보고서 파일로 내려받습니다.",
          "지식 검색 — 사내 자료에서 근거를 찾아 답합니다.",
        ]},
      ],
    },
    {
      id: "faq",
      title: "4. 자주 묻는 질문",
      blocks: [
        { type: "table",
          columns: ["질문", "답변"],
          rows: [
            ["파일이 안 올라갑니다", "20MB를 넘는지, 지원 형식인지 확인해 주세요."],
            ["한글(.hwp)도 되나요", "지원합니다. 열리지 않으면 HWPX로 저장해 올려 주세요."],
          ]},
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
  muted: "--muted", bg: "--bg", surface: "--surface", line: "--line",
};
for (const [key, value] of Object.entries(SITE.colors || {})) {
  const cssVar = CSS_VAR_BY_KEY[key];
  if (cssVar && value) document.documentElement.style.setProperty(cssVar, value);
}

const { createApp } = Vue;

createApp({
  data() {
    return { site: SITE, activeId: (SITE.sections[0] || {}).id || "" };
  },
  mounted() {
    // 스크롤 위치에 따라 목차의 현재 항목을 표시한다.
    const onScroll = () => {
      let current = this.activeId;
      for (const s of this.site.sections) {
        const el = document.getElementById(s.id);
        if (el && el.getBoundingClientRect().top <= 90) current = s.id;
      }
      this.activeId = current;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  },
}).mount("#app");
