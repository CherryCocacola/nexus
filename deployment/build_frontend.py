# 프런트(정적 UI)만 떼어 IIS 등 별도 웹서버에 올릴 배포 폴더를 만든다.
"""
프런트 분리 배포 빌드 — IIS(192.168.10.215 등) 정적 호스팅용.

■ 먼저 알아야 할 것: 여기서 "빌드"는 컴파일이 아니다
    NOVA 프런트는 번들러(webpack/vite)를 쓰는 프레임워크 앱이 아니라, FastAPI 가
    함께 서빙하던 **정적 파일 묶음**(index.html + static/ 39개, 약 1MB)이다.
    그래서 이 스크립트가 하는 일은 셋뿐이다.
      ① 배포 폴더로 자산 복사
      ② API 주소·키를 담은 config.js 생성 (앱 코드는 손대지 않는다)
      ③ IIS 용 web.config 생성 (MIME·기본문서·캐시)

■ 프런트는 API 주소를 어떻게 아는가
    앱의 fetch 27곳은 전부 상대경로(`/v1/...`, `/health`)다. index.html 의 fetch
    래퍼 한 곳이 `window.NOVA_CONFIG.apiBase` 를 앞에 붙인다. 그래서 배포별로
    config.js 만 바꾸면 되고 앱 코드는 그대로다. 통합 배포(112)에서는 apiBase 가
    비어 있어 종전과 동일하게 same-origin 으로 나간다.

■ 왜 경로를 상대경로로 바꾸는가
    원본은 `/static/...` 절대경로라 사이트 루트에 올려야만 동작한다. IIS 하위
    애플리케이션(예: /nova/)에 올리면 전부 404 가 난다. 빌드 때 `static/...` 로
    바꿔 두면 어느 경로에 올려도 동작한다(이 UI 는 클라이언트 라우팅이 없어
    문서 URL 이 바뀌지 않으므로 상대경로가 안전하다).

■ 교차 오리진 전제 (중요)
    프런트(192.168.10.215)와 API(192.168.21.112:8600)가 다른 오리진이므로,
    **API 서버에서 CORS 를 열어야** 한다. 112 컨테이너에 환경변수를 주입한다.
        NEXUS_EXTRA_CORS_ORIGINS=http://192.168.10.215
    LAN 사설 주소만 받아들이며 공인 주소는 무시된다(web/middleware.py 참고).

■ ★보안 — API 키가 브라우저로 내려간다
    config.js 의 apiKey 는 페이지를 여는 누구나 볼 수 있다. 통합 배포에서도 같았지만,
    프런트를 별도 서버로 빼면 **닿을 수 있는 사람의 범위가 넓어진다.**
    공개 구간에 둘 계획이라면 이 방식 그대로는 안 된다(역프록시가 서버측에서 키를
    붙이거나, 사용자 로그인 기반 토큰으로 바꿔야 한다).

사용법:
    python -m deployment.build_frontend --api-base http://192.168.21.112:8600
    python -m deployment.build_frontend --api-base ... --out D:/deploy/nova-front

작성자: 이현수 / 작성일: 2026-08-14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_STATIC = REPO / "web" / "static"
DEFAULT_OUT = REPO / "dist" / "nova-front-iis"

# 배포에 넣지 않는 것 — 개발용 페이지와 파이썬 부산물.
EXCLUDE_NAMES = {"chrome.html", "__pycache__"}

# IIS 가 기본으로 모르는(또는 서버마다 빠져 있는) 확장자. 없으면 404/깨진 아이콘이 된다.
MIME_MAP = {
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
}


def _config_js(api_base: str, api_key: str) -> str:
    """배포 환경 설정 파일 본문."""
    return (
        "// NOVA 프런트 분리 배포 설정 — deployment/build_frontend.py 가 생성했다.\n"
        "// 이 파일만 고치면 API 주소·키를 바꿀 수 있다(앱 코드 재빌드 불필요).\n"
        "//\n"
        "// ★apiKey 는 브라우저로 내려가는 값이라 이 페이지에 닿는 사람은 모두 볼 수 있다.\n"
        "//   공개 구간에 두려면 이 방식 대신 서버측에서 키를 붙이는 구조가 필요하다.\n"
        "window.NOVA_CONFIG = {\n"
        f"  apiBase: {json.dumps(api_base)},\n"
        f"  apiKey: {json.dumps(api_key)},\n"
        "};\n"
    )


def _web_config() -> str:
    """IIS 정적 호스팅용 web.config.

    - 기본 문서를 index.html 로
    - 폰트/JSON MIME 등록(서버마다 빠져 있어 아이콘이 깨진다)
    - index.html·config.js 는 캐시 금지(배포 후 즉시 반영되게)
    - vendor 자산은 장기 캐시(해시가 아니라 경로 고정이므로 1일로 둔다)
    """
    mime = "\n".join(
        f'        <remove fileExtension="{ext}" />\n'
        f'        <mimeMap fileExtension="{ext}" mimeType="{mt}" />'
        for ext, mt in MIME_MAP.items()
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!-- NOVA 프런트 정적 호스팅용 IIS 설정 — build_frontend.py 가 생성했다.
     이 사이트는 정적 파일만 서빙한다. API 호출은 브라우저가 직접 다른 오리진
     (config.js 의 apiBase)으로 보내므로 IIS 에 프록시(ARR) 설정이 필요 없다. -->
<configuration>
  <system.webServer>
    <defaultDocument>
      <files>
        <clear />
        <add value="index.html" />
      </files>
    </defaultDocument>

    <staticContent>
{mime}
      <!-- 기본 캐시: 자산은 1일. index.html·config.js 는 아래에서 따로 끈다. -->
      <clientCache cacheControlMode="UseMaxAge" cacheControlMaxAge="1.00:00:00" />
    </staticContent>

    <!-- 배포 직후 옛 화면이 뜨는 것을 막는다 — 실제로 겪은 문제다. -->
    <location path="index.html">
      <system.webServer>
        <staticContent>
          <clientCache cacheControlMode="DisableCache" />
        </staticContent>
      </system.webServer>
    </location>
    <location path="static/config.js">
      <system.webServer>
        <staticContent>
          <clientCache cacheControlMode="DisableCache" />
        </staticContent>
      </system.webServer>
    </location>

    <httpProtocol>
      <customHeaders>
        <add name="X-Content-Type-Options" value="nosniff" />
        <add name="X-Frame-Options" value="SAMEORIGIN" />
        <add name="Referrer-Policy" value="same-origin" />
      </customHeaders>
    </httpProtocol>
  </system.webServer>
</configuration>
"""


def _readme(api_base: str) -> str:
    """배포 담당자가 폴더만 받아도 끝까지 갈 수 있게, 산출물에 함께 넣는 안내문."""
    return f"""# IDINO NOVA 프런트 — IIS 배포 안내

이 폴더는 **정적 파일만** 들어 있습니다. 서버 런타임(.NET/Node)이 필요 없습니다.
API 는 브라우저가 직접 아래 주소로 호출합니다.

    API 주소: {api_base}

## 1. IIS 설정

1. 이 폴더 전체를 서버로 복사합니다 (예: `C:\\inetpub\\nova-front`).
2. IIS 관리자에서 사이트(또는 하위 응용 프로그램)를 만들고 실제 경로를 이 폴더로 지정합니다.
3. **응용 프로그램 풀 → .NET CLR 버전 = "관리되는 코드 없음"** 으로 둡니다(정적 전용).
4. `web.config` 는 이미 들어 있습니다. 기본 문서(index.html), 폰트 MIME,
   캐시 정책이 설정돼 있습니다.

> 하위 경로(`/nova/`)에 올려도 동작합니다 — 자산 경로가 전부 상대경로입니다.

## 2. ★API 서버에 CORS 허용 (이 단계를 빼면 화면만 뜨고 아무것도 안 됩니다)

프런트와 API 가 서로 다른 주소이므로, API 서버가 이 프런트 주소를 허용해야 합니다.
API 서버(nexus-web 컨테이너)에 환경변수를 주입하고 재시작합니다.

    NEXUS_EXTRA_CORS_ORIGINS=http://<이 프런트의 주소>

예: `NEXUS_EXTRA_CORS_ORIGINS=http://192.168.10.215`
쉼표로 여러 개를 넣을 수 있습니다. **사설망(LAN) 주소만 허용되고 공인 주소는 무시됩니다.**

## 3. 확인

브라우저로 열고 `F12 → Console` 에 다음이 없으면 정상입니다.

- `Access-Control-Allow-Origin` 관련 오류 → 2번 CORS 미설정
- `net::ERR_CONNECTION` → 프런트 서버에서 API 주소로 통신이 안 됨(방화벽/라우팅)
- 폰트/아이콘 404 → `web.config` 의 MIME 설정이 적용되지 않음

화면 좌상단 상태 점이 **녹색**이면 API 연결까지 정상입니다.

## 4. 설정 변경

`static/config.js` 만 고치면 됩니다(재빌드 불필요). 고친 뒤 브라우저에서 `Ctrl+F5`.

## ★보안 주의

`static/config.js` 의 `apiKey` 는 **브라우저로 내려갑니다.** 이 페이지에 접근할 수 있는
사람은 누구나 그 값을 볼 수 있고, 그 키로 API 를 직접 호출할 수 있습니다.
사내망 한정으로만 쓰고, 공개 구간에 노출하지 마십시오.
"""


def _to_relative_paths(html: str) -> str:
    """`/static/...` 절대경로를 `static/...` 상대경로로 바꾼다.

    사이트 루트가 아니라 하위 애플리케이션(/nova/)에 올려도 동작하게 하기 위함이다.
    `/v1/...` 같은 API 경로는 손대지 않는다 — 그쪽은 fetch 래퍼가 apiBase 를 붙인다.

    HTML 속성뿐 아니라 **JS 문자열 리터럴**도 바꿔야 한다. Mermaid 번들은
    `s.src = '/static/vendor/mermaid/...'` 처럼 스크립트가 동적으로 붙이는데,
    속성만 처리하면 이 한 줄이 남아 배포 후에야 404 로 드러난다(빌드 검사가 잡았다).
    """
    # ★치환식에서 \1 을 빠뜨리면 여는 따옴표가 통째로 사라진다(실제로 겪었다).
    #   src="/static/x" → src=static/x"  로 HTML 이 조용히 깨지고, 아래 절대경로
    #   검사마저 통과해 버려 배포 후에야 드러난다. 캡처를 반드시 되돌린다.
    return re.sub(r"([\"'`])/static/", r"\1static/", html)


def build(api_base: str, api_key: str, out_dir: Path) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_static = out_dir / "static"
    out_static.mkdir(parents=True)

    # ① 자산 복사 (index.html 과 chrome.html 은 따로 다룬다)
    copied = 0
    for src in SRC_STATIC.rglob("*"):
        if any(part in EXCLUDE_NAMES for part in src.parts):
            continue
        if not src.is_file() or src.name == "index.html":
            continue
        dst = out_static / src.relative_to(SRC_STATIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # ② index.html — 경로를 상대경로로
    html = (SRC_STATIC / "index.html").read_text(encoding="utf-8")
    html = _to_relative_paths(html)
    (out_dir / "index.html").write_text(html, encoding="utf-8", newline="\n")

    # manifest.json 도 절대경로를 쓰므로 함께 바꾼다.
    mf = out_static / "manifest.json"
    if mf.exists():
        data = json.loads(mf.read_text(encoding="utf-8"))
        data["start_url"] = "./"
        for icon in data.get("icons", []):
            icon["src"] = icon["src"].lstrip("/").replace("static/", "", 1)
            icon["src"] = "./" + icon["src"]
        mf.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )

    # ③ 설정·IIS 파일 생성
    (out_static / "config.js").write_text(
        _config_js(api_base, api_key), encoding="utf-8", newline="\n"
    )
    (out_dir / "web.config").write_text(_web_config(), encoding="utf-8", newline="\n")
    (out_dir / "README-배포.md").write_text(_readme(api_base), encoding="utf-8", newline="\n")

    # 남은 절대경로가 있으면 배포 후에야 404 로 드러난다 — 여기서 잡는다.
    leftovers = sorted(set(re.findall(r'["\']/(?:static)/[\w./-]+', html)))

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    return {
        "out": out_dir,
        "files": len(list(p for p in out_dir.rglob("*") if p.is_file())),
        "copied_assets": copied,
        "bytes": total,
        "leftover_abs_paths": leftovers,
        "index_sha256": hashlib.sha256((out_dir / "index.html").read_bytes()).hexdigest()[:12],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NOVA 프런트 분리 배포 빌드(IIS 등 정적 호스팅용)")
    ap.add_argument(
        "--api-base",
        required=True,
        help="API 서버 주소 (예: http://192.168.21.112:8600). 끝 슬래시 없이.",
    )
    ap.add_argument(
        "--api-key",
        default="nexus-b200-test-key-001",
        help="테넌트 API 키. ★브라우저로 내려가는 값이다.",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="출력 폴더")
    args = ap.parse_args()

    api_base = args.api_base.rstrip("/")
    if not api_base.startswith(("http://", "https://")):
        print("[오류] --api-base 는 http:// 또는 https:// 로 시작해야 합니다.")
        return 1

    r = build(api_base, args.api_key, Path(args.out))

    print("=" * 66)
    print("NOVA 프런트 분리 배포 빌드 완료")
    print("=" * 66)
    print(f"  출력 폴더   : {r['out']}")
    print(f"  파일 수     : {r['files']}개 (자산 {r['copied_assets']}개 + index/config/web.config)")
    print(f"  크기        : {r['bytes'] / 1024:,.0f} KB")
    print(f"  API 주소    : {api_base}")
    print(f"  index 해시  : {r['index_sha256']}")
    if r["leftover_abs_paths"]:
        print("  ★남은 절대경로(하위 경로 배포 시 404):")
        for p in r["leftover_abs_paths"]:
            print("     ", p)
    else:
        print("  절대경로    : 없음 (하위 경로에 배포해도 동작)")

    print("\n다음 단계")
    print("  1) 폴더를 IIS 서버로 복사하고 사이트/앱의 실제 경로로 지정")
    print("  2) 응용 프로그램 풀: '관리되는 코드 없음'(정적 전용)")
    print("  3) ★API 서버(112)에 CORS 오리진 주입 후 재시작:")
    print("        NEXUS_EXTRA_CORS_ORIGINS=http://<IIS 주소>")
    print("  4) 브라우저에서 열고 F12 콘솔에 CORS 오류가 없는지 확인")
    print("\n주의 — config.js 의 apiKey 는 브라우저로 내려갑니다"
          "(페이지를 여는 사람은 모두 볼 수 있음).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
