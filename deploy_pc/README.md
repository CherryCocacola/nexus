# Nexus 분리구조 배포 — 서비스 PC (오케스트레이터)

이 PC(= **Machine A / 서비스**)가 Nexus 오케스트레이터·웹·CLI를 실행하고,
무거운 **추론·임베딩·데이터**는 **B200 GPU 백엔드**를 SSH 터널로 호출한다.

```
[이 PC = 서비스]                         [B200 = GPU 백엔드]
 오케스트레이터(쿼리루프/도구/권한)  ──SSH 터널──▶  vLLM(8001) · 임베딩(8002)
 + RAG 검색로직 + 웹/CLI                            PostgreSQL(5440) · Redis(6340)
 ※ GPU 불필요                                       ※ 모델·데이터 상주
```

터널이 B200의 내부 포트를 **이 PC의 localhost** 로 매핑하므로,
`config/nexus_config.yaml` 은 백엔드를 전부 `127.0.0.1` 로 가리킨다(공개 노출 없음 = 안전).

---

## 최초 셋업 (새 PC/서버에서 1회)

**Windows**
```powershell
.\deploy_pc\setup.ps1
```
**Linux/macOS**
```bash
bash deploy_pc/setup.sh
```
→ `.venv_pc` 생성 + 오케스트레이터 의존성 설치(GPU 라이브러리 제외).

그다음 아래 3개를 환경에 맞게 확인/수정:
- **`.env`** — `NEXUS_PG_PASSWORD`, `NEXUS_REDIS_PASSWORD` (B200 백엔드 비번)
- **`config/nexus_config.yaml`** — 백엔드 호스트. 터널을 쓰면 `127.0.0.1` 그대로. 백엔드를 직접(LAN) 가리키려면 여기 수정.
- **`deploy_pc/tunnel.ps1` / `tunnel.sh`** — SSH 키 경로, Bastion 주소/포트.

---

## 실행 (매번)

**1) 터널 (창 1 — 열어둔 채 유지)**
```powershell
.\deploy_pc\tunnel.ps1      # Windows  (passphrase 입력)
```
```bash
bash deploy_pc/tunnel.sh    # Linux/macOS
```

**2) 서비스 (창 2)**
```powershell
.\deploy_pc\start_web.ps1   # 웹  → http://127.0.0.1:8600
.\deploy_pc\start_cli.ps1   # 또는 대화형 CLI
```
```bash
bash deploy_pc/start_web.sh
bash deploy_pc/start_cli.sh
```

---

## 접속 / 인증

- **CLI**: 인증 불필요 — 바로 대화.
- **웹 API**: `web_auth` 가 켜져 있어 `/v1/chat` 는 헤더 `Authorization: Bearer <API키>` 필요.
  - 테스트 키(기본): `nexus-b200-test-key-001` (`config/tenants.yaml` 의 default 테넌트).
  - 외부 앱/멀티테넌트: `config/tenants.yaml` 에 테넌트별 `api_keys` 추가 → 그 키로 호출하면 격리 세션(agent) 생성.
- **웹 UI(브라우저)**: `web/static/index.html` 에 API 키 자동주입 패치가 적용돼 있어,
  **인증을 켠 채로 브라우저에서 바로 채팅**된다(키 입력 불필요). 키 변경은 index.html 의 `API_KEY` 값 수정.
  ※ 반드시 **`http://127.0.0.1:8600`** 로 접속. `localhost` 는 IPv6(::1)로 풀려 다른 서비스(Docker 등)로 갈 수 있음.

예) API 호출 테스트:
```bash
curl http://127.0.0.1:8600/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer nexus-b200-test-key-001" \
  --data-binary '{"message":"질문"}'
```

---

## 다른 서버로 이전 (relocation)

1. 이 저장소를 새 서버에 복제(또는 복사).
2. `setup.ps1` / `setup.sh` 실행.
3. `.env`·`config/nexus_config.yaml`·`tunnel.*` 의 키/주소만 새 환경에 맞게 수정.
4. `tunnel` → `start_web`/`start_cli`.

백엔드(B200)를 다른 GPU 서버로 옮긴 경우엔 `tunnel.*` 의 Bastion 주소만 바꾸면 된다.
백엔드가 LAN에 직접 노출돼 있으면 터널 없이 `config/nexus_config.yaml` 의 호스트를 그 주소로 지정해도 된다.

---

## 파일

| 파일 | 용도 |
|---|---|
| `requirements-pc.txt` | 오케스트레이터 전용 의존성(GPU 제외) |
| `setup.ps1` / `setup.sh` | venv + 의존성 설치(최초 1회) |
| `tunnel.ps1` / `tunnel.sh` | B200 백엔드 SSH 터널(4포트) |
| `start_web.ps1` / `start_web.sh` | 웹(오케스트레이터) 기동 |
| `start_cli.ps1` / `start_cli.sh` | 대화형 CLI 기동 |

> 상위 폴더의 `.env`, `config/nexus_config.yaml`, `config/tenants.yaml` 이 실제 설정.
> `.venv_pc/` 는 OS별로 다르므로 이전 시 **복사하지 말고 setup 재실행**.
