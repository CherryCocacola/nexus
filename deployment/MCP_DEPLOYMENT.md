# Nexus MCP 서버 배포 가이드 (DB 서버 192.168.10.39)

Nexus 의 LAN MCP 서버(`db` / `diag` / `kowiki` / `docingest`)를 **DB 서버
192.168.10.39** 에 배포·기동하는 절차를 정리한다. 에어갭(폐쇄망) 환경을 전제로
하며, 모든 통신은 LAN 주소만 사용한다.

> **안전 원칙**
> - 배포 스크립트(`scripts/_ssh_deploy_mcp.py`)는 **운영자가 검토 후 직접 실행**한다.
> - 스크립트 기본 모드는 `--dry-run`(원격을 건드리지 않고 출력만)이다.
>   실제 배포는 `--no-dry-run` 을 명시해야 한다(fail-closed).
> - 의존성은 **오프라인 wheel** 로 사전 설치한다. 스크립트는 `pip install` 을 하지
>   않는다(에어갭 규칙, anti-pattern #10).

---

## 0. 배포 대상 토폴로지

| 서버 | 기본 포트 | 접속 대상 | 비고 |
|---|---|---|---|
| `db` | 8810 | PostgreSQL `localhost`(.39 컨테이너) | read-only SELECT 전용 |
| `diag` | 8811 | GPU(.28) SSH / DB(.39) socket / 임베딩(.28:8002) | 인프라 도달성·지연 점검 |
| `kowiki` | 8813 | 임베딩 192.168.21.112:8002 + pgvector(localhost) | tb_knowledge 검색(read-only) |
| `docingest` | 8814 | 임베딩 192.168.21.112:8002 + PostgreSQL(localhost) | parse/search(읽기) + ingest(쓰기) |

- `db` MCP 는 **PostgreSQL 을 localhost 로** 접속한다. .39 에서 도는 `docutil-postgres`
  컨테이너가 호스트 포트(예: 5440)로 노출돼 있으므로, `config/nexus_config.yaml`
  의 `postgresql.host` 를 .39 기준으로 `localhost`(또는 `127.0.0.1`), `port` 를 해당
  호스트 포트로 맞춘다.
- `kowiki`/`docingest` 의 임베딩은 GPU 서버(.28:8002)로 **LAN HTTP** 호출한다.
- `docingest` 는 .39 에 GPU 가 없으므로 자동으로 **경량 파서(pdfplumber/Tesseract)**
  만 사용한다(`docingest_server.py` 의 `_gpu_available()` 폴백 — 별도 플래그 불필요).

---

## 1. 사전 준비 — Python 3.11 + 오프라인 의존성

1. **Python 3.11+** 가 .39 에 설치돼 있어야 한다.
   - 권장: Nexus 전용 venv 를 만들어 격리한다.
     ```bash
     python3.11 -m venv /home/idino/nexus-venv
     ```
2. **오프라인 wheel 로 의존성 설치(에어갭).** 외부 인터넷이 없으므로 사전 준비한
   wheel 디렉토리에서 설치한다.
   - 공통(db/kowiki/docingest 모두): `fastapi`, `uvicorn`, `asyncpg`, `httpx`,
     `pydantic`, `pyyaml`, `paramiko`(diag 용)
   - docingest 파서(경량 CPU): `python-pptx`, `pdfplumber`, `pytesseract`(+ OS 패키지
     `tesseract-ocr`), `python-docx`(.hwp LibreOffice 변환 경로)
   - 설치 예시(오프라인 wheel 디렉토리 `/opt/wheels`):
     ```bash
     /home/idino/nexus-venv/bin/pip install --no-index --find-links=/opt/wheels \
       fastapi uvicorn asyncpg httpx pydantic pyyaml paramiko \
       python-pptx pdfplumber pytesseract python-docx
     ```
   - GPU 전용 고품질 파서(`docling`/`paddleocr`/`torch`)는 .39 에 **설치하지 않는다**.
     GPU 가 없으면 자동으로 경량 파서로 폴백한다.
3. 설치 확인은 배포 스크립트의 의존성 점검 단계가 대신 해 준다(아래 2-3 참조).

---

## 2. 코드 배포

### 2-1. 배포 스크립트 사용 (권장 — dry-run 우선)

스크립트: `scripts/_ssh_deploy_mcp.py` (Machine A 에서 실행, paramiko 로 .39 접속)

```bash
# 1) 먼저 dry-run 으로 무엇을 할지 확인 (원격을 건드리지 않음 — 기본 모드)
python scripts/_ssh_deploy_mcp.py --servers db diag kowiki docingest

# 2) 검토 후 실제 배포 (코드 업로드 + 의존성 점검 + 기동 + 헬스체크)
python scripts/_ssh_deploy_mcp.py --servers db diag kowiki docingest \
  --no-dry-run \
  --remote-dir /home/idino/nexus \
  --python /home/idino/nexus-venv/bin/python \
  --api-key '<운영_Bearer_키>'
```

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `--servers` | 기동할 서버 목록(기본 `db diag`). 예: `--servers db kowiki docingest` |
| `--dry-run` / `--no-dry-run` | 실행 여부. 기본 `--dry-run`(출력만). 실제 배포는 `--no-dry-run` |
| `--host` / `--user` / `--password` | SSH 접속 정보(기본 .39 / idino) |
| `--remote-dir` | 원격 코드 디렉토리(기본 `/home/idino/nexus`) |
| `--python` | 원격 파이썬 인터프리터(venv 면 그 경로) |
| `--api-key` | MCP Bearer 인증 키(기본 placeholder `local-key`) |
| `--code-mode` | `upload`(SFTP, 기본) / `pull`(원격 git pull) / `skip` |

- `--code-mode upload`(기본): `mcp_servers/`, `core/`, `config/` 를 SFTP 로 올린다.
  rsync/git 없이 paramiko 만으로 동작한다.
- `--code-mode pull`: .39 에 이미 Nexus 레포가 git clone 되어 있을 때만 사용한다.
  사내 git 원격(LAN)이 설정돼 있어야 한다(에어갭 — 인터넷 원격 불가).

### 2-2. 수동 git pull (레포가 .39 에 있는 경우)

```bash
cd /home/idino/nexus
git rev-parse --abbrev-ref HEAD
git pull --ff-only
```

### 2-3. 의존성 점검

배포 스크립트가 기동 전에 `fastapi/uvicorn/asyncpg/httpx/pptx/pdfplumber` import
가능 여부를 출력한다. `MISSING` 이 있으면 1번으로 돌아가 wheel 로 설치한다.

---

## 3. MCP 서버 기동

### 3-1. 빠른 기동 — nohup (스크립트 기본)

배포 스크립트가 각 서버를 다음과 같이 기동한다(run.py 규약):

```bash
cd /home/idino/nexus
PYTHONPATH=/home/idino/nexus setsid nohup \
  /home/idino/nexus-venv/bin/python -m mcp_servers.run db \
  --host 0.0.0.0 --port 8810 --api-key '<키>' \
  </dev/null >mcp_db.log 2>&1 &
echo $! > mcp_db.pid
```

- 로그: `mcp_<name>.log`, PID: `mcp_<name>.pid` (원격 디렉토리 기준).
- 중복 기동 방지를 위해 스크립트는 기동 전에 `pkill -f 'mcp_servers.run <name>'` 을 한다.

### 3-2. 운영 권장 — systemd unit

장기 운영은 nohup 대신 systemd 로 관리한다(부팅 자동 기동·재시작·로그 일원화).
아래는 `db` 예시(`/etc/systemd/system/mcp-db.service`). 다른 서버는 이름/포트만
바꾼다(`mcp-diag` 8811, `mcp-kowiki` 8813, `mcp-docingest` 8814).

```ini
[Unit]
Description=Nexus MCP db server (read-only PostgreSQL)
After=network.target docker.service

[Service]
Type=simple
User=idino
WorkingDirectory=/home/idino/nexus
Environment=PYTHONPATH=/home/idino/nexus
# API 키는 환경파일로 분리 권장: EnvironmentFile=/etc/nexus/mcp.env (MCP_API_KEY=...)
ExecStart=/home/idino/nexus-venv/bin/python -m mcp_servers.run db \
  --host 0.0.0.0 --port 8810 --api-key local-key
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mcp-db.service
sudo systemctl status mcp-db.service
journalctl -u mcp-db.service -f      # 로그 확인
```

> docingest 서버는 GPU 가 없는 .39 에서 경량 파서만 사용한다. 별도 환경변수는
> 불필요하다. (Windows 개발 환경의 PaddleOCR/OpenMP 충돌 회피용
> `KMP_DUPLICATE_LIB_OK` 는 Linux 배포에서 불필요.)

---

## 4. 방화벽 — 88xx 포트는 Machine A 에서만 접근 허용 (LAN 한정)

MCP 포트(8810/8811/8813/8814)는 **Nexus 오케스트레이터 호스트(Machine A)에서만**
접근할 수 있어야 한다. 외부/그 외 호스트의 접근은 차단한다(에어갭 + 최소 노출).

```bash
# Machine A 의 IP 가 192.168.x.y 라고 가정 — 해당 IP 에서만 88xx 허용
for p in 8810 8811 8813 8814; do
  sudo ufw allow from 192.168.x.y to any port $p proto tcp
done
# 그 외 모든 호스트의 88xx 접근 차단(명시 거부)
for p in 8810 8811 8813 8814; do
  sudo ufw deny $p/tcp
done
sudo ufw status numbered
```

> `--host 0.0.0.0` 은 LAN 바인드일 뿐이며, 실제 노출 통제는 방화벽이 담당한다.
> `/health` 는 인증 없이 응답하므로, 방화벽으로 출처를 제한하는 것이 중요하다.

---

## 5. Nexus 측 설정 — `config/nexus_config.yaml` (mcp 섹션)

배포 후 Machine A 의 설정을 활성화한다. **기본은 fail-closed(`enabled: false`)** 이므로
명시적으로 켜야 연결된다.

```yaml
mcp:
  enabled: true                 # 전역 마스터 스위치 — 활성
  connect_timeout_sec: 5.0
  servers:
    - name: "db"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8810"
      enabled: true
      trust: { read_only: true }
    - name: "diag"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8811"   # .28 → .39 로 변경
      enabled: true
      trust: { read_only: true }
    - name: "docingest"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8814"   # .28 → .39 로 변경
      enabled: true
      trust: { read_only: true }              # ingest(쓰기) 쓰려면 allow_write: true 명시
    - name: "kowiki"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8813"   # .28 → .39 로 변경
      enabled: true
      trust: { read_only: true }
      expose_to_worker: false                 # 그대로 유지(KNOWLEDGE 모드 자동 RAG 가 담당)
```

- 현재 레포 기본값은 `db` 만 `.39`, 나머지(`diag`/`docingest`/`kowiki`)는 `.28` 을
  가리킨다. **이번 배포에서 4개 모두 `.39` 로 옮기므로 위처럼 base_url 을 변경한다.**
- `kowiki` 는 `expose_to_worker: false` 를 **유지**한다. 검색은 KNOWLEDGE 모드 자동 RAG
  주입이 담당하므로 Worker 도구로 중복 노출하지 않는다(5090 8K 컨텍스트 overflow 방지).
- MCP 인증 키는 서버 기동 시 `--api-key` 와 **동일**해야 한다(framework 의 Bearer 검증).

---

## 6. 검증

1. **서버 단독 헬스체크**(.39 에서, 또는 Machine A 에서 LAN 으로):
   ```bash
   for p in 8810 8811 8813 8814; do
     echo "port $p:"; curl -s -m 5 http://192.168.10.39:$p/health; echo
   done
   ```
   각각 `{"status":"healthy", ...}` 가 나와야 한다.

2. **Nexus 재기동 후 연결 확인**: Machine A 에서 Nexus 를 재기동하고 `/metrics`
   엔드포인트에서 `mcp.connected`(연결된 서버 수/목록)를 확인한다.

3. **e2e 검증 스크립트 변형**: `scripts/_verify_mcp_live.py` 는 로컬에서 서버를 띄워
   클라이언트로 list_tools→call_tool 을 확인한다. .39 배포 검증에는 이를 변형해
   `McpClient("http://192.168.10.39:88xx", api_key=...)` 로 원격 서버에 직접
   list_tools/call_tool 을 던지는 방식을 쓴다(전부 read-only 라 운영 데이터 불변).

4. **도구 노출 확인**: Nexus `/v1/tools`(또는 도구 목록 경로)에서 db/diag/docingest
   도구가 보이는지 확인한다(kowiki 는 `expose_to_worker: false` 라 Worker 도구 풀에는
   노출되지 않음 — 정상).

---

## 7. 롤백

가장 빠른 비활성화는 **Machine A 설정의 fail-closed 스위치**를 끄는 것이다.

```yaml
mcp:
  enabled: false      # 전역 즉시 비활성 — 어떤 MCP 서버에도 연결하지 않음
```

- 또는 개별 서버만: 해당 `servers[].enabled: false`.
- 설정 변경 후 Nexus 재기동(또는 핫리로드) 시 MCP 연결이 끊기고, 어댑터가 도구를
  더 이상 노출하지 않는다(권한 파이프라인 Layer 1 에서 제거).
- .39 의 서버 프로세스 정지(선택):
  - systemd: `sudo systemctl stop mcp-db.service mcp-diag.service ...`
  - nohup: `pkill -f 'mcp_servers.run'` (또는 `kill $(cat mcp_<name>.pid)`)

---

## 부록 — 에어갭 체크리스트

- [ ] 외부 인터넷 호출 없음(모든 URL 이 192.168.x / localhost)
- [ ] 런타임 `pip install` 없음 — 의존성은 오프라인 wheel 로 사전 설치
- [ ] 모델/임베딩은 GPU 서버(.28)에 사전 배치, LAN 으로만 호출
- [ ] MCP 포트는 방화벽으로 Machine A 만 허용
- [ ] MCP `--api-key` 와 Nexus 설정 키 일치
- [ ] 배포 스크립트는 운영자가 dry-run 검토 후 `--no-dry-run` 으로 실행
