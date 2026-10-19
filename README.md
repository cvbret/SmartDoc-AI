# SmartDoc-AI

SmartDoc-AI 是一个企业文档智能问答 / RAG backend。用户可以上传企业文档，系统完成文档解析、chunking、embedding、向量召回和 BGE reranking，再结合 DeepSeek 与会话历史生成回答。项目包含固定 Ground Truth、RAGAS 与 stable-context-ID 离线评测及 paired comparison。

## 项目特性

### 文档处理

- 支持 PDF、DOCX、TXT、Markdown
- 统一通过 `parse_document()` 解析为文本
- 通过 `split_text()` 切分文档，再生成 Embedding 并保存到 ChromaDB
- DOCX 当前提取普通段落和表格文本

### 文档管理

- Upload
- List
- Detail
- Update
- Delete

### RAG 问答

- Document parsing
- Text chunking
- `BAAI/bge-small-zh-v1.5` Embedding
- ChromaDB vector retrieval
- `BAAI/bge-reranker-base` 二阶段重排，Top-K 候选 → Top-N context
- DeepSeek `deepseek-chat`

### Conversation

- Redis 保存会话消息
- 支持 multi-turn history
- 通过 TTL 自动过期，默认有效期为 3600 秒

### Engineering

- FastAPI
- PostgreSQL / SQLAlchemy
- Alembic migrations
- Global exception handling
- Logging
- pytest
- PostgreSQL integration tests
- GitHub Actions CI
- Docker / Docker Compose

## 技术栈

| 类别 | 技术 |
| --- | --- |
| Backend | FastAPI、Uvicorn、Python 3.11 |
| AI / RAG | Sentence Transformers、BAAI/bge-small-zh-v1.5、ChromaDB、DeepSeek |
| Database | PostgreSQL、SQLAlchemy、Alembic |
| Conversation | Redis |
| Infrastructure | Docker、Docker Compose |
| Testing | pytest、PostgreSQL integration tests、GitHub Actions |

## 系统架构

```mermaid
flowchart LR
    Client[Client] --> FastAPI[FastAPI]
    FastAPI --> PostgreSQL[(PostgreSQL)]
    FastAPI --> Redis[(Redis)]
    FastAPI --> RAG[RAG Service]
    Documents[Documents] --> Chunk[Parse and Chunk]
    Chunk --> Embedding[BGE-small Embedding]
    Embedding --> Chroma[(ChromaDB)]
    RAG --> Query[Question Embedding]
    Query --> Retrieval[Chroma Top-K]
    Chroma --> Retrieval
    Retrieval --> Reranker[BGE Reranker]
    Reranker --> Context[Top-N Context]
    Context --> DeepSeek[DeepSeek]
    GroundTruth[Fixed Corpus and 20-sample Ground Truth] --> Eval[Ephemeral Baseline / Reranker Runs]
    Eval --> Metrics[RAGAS + ID Metrics]
    Metrics --> Comparison[Saved JSON Paired Comparison]
```

## 文档处理流程

```mermaid
flowchart TD
    Upload[UploadFile] --> Parse[parse_document]
    Parse --> Formats[PDF / DOCX / TXT / Markdown]
    Formats --> Split[split_text]
    Split --> Embed[embed_chunks]
    Embed --> Save[save_chunks]
    Save --> Chroma[(ChromaDB)]
    Chroma --> Status[PostgreSQL status = completed]
```

实际处理过程中，系统会先创建 PostgreSQL 文档记录，完成解析、切分、Embedding 和 ChromaDB 保存后，再将文档状态更新为 `completed`。

## RAG 问答流程

`Question → BGE-small → Chroma Top-K → BGE Reranker → Top-N Context → shared prompt + history → DeepSeek`

默认 K=10、N=3。离线 baseline 使用相同 K/N 和候选池，保留 Chroma 原始排序；reranker 模式仅增加 BGE 重排。

聊天请求会先将用户消息写入 Redis，再读取 session history；RAG 检索得到的 context 与 history 一起构造 prompt，交给 DeepSeek 生成 answer。

## 固定数据集离线评测

使用 TASK-004 固定 Ground Truth：20 个问题、5 个 chunks，dataset/corpus version 1.0。
Baseline 和 Reranker 各执行一次完整正式运行，共 40 次样本评估；六项指标均为 20 个成功配对，无失败或跳过。
两侧使用相同模型、prompt、数据顺序、K=10、Top-3；20/20 候选池的集合和顺序一致。

| Metric | Baseline | Reranker | Delta | Paired N | Win | Tie | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| faithfulness | 0.927579 | 0.894762 | -0.032817 | 20 | 2 | 13 | 5 |
| answer_relevancy | 0.843333 | 0.840149 | -0.003185 | 20 | 11 | 3 | 6 |
| context_precision_llm | 0.950000 | 1.000000 | 0.050000 | 20 | 3 | 17 | 0 |
| context_recall_llm | 1.000000 | 1.000000 | 0.000000 | 20 | 0 | 20 | 0 |
| context_precision_id | 0.400000 | 0.400000 | 0.000000 | 20 | 0 | 20 | 0 |
| context_recall_id | 1.000000 | 1.000000 | 0.000000 | 20 | 0 | 20 | 0 |

Delta 为 reranker − baseline 的绝对分差，仅使用双方成功的同一样本；`abs(delta) < 1e-6` 视为 tie。
四项 LLM semantic metrics 使用 RAGAS 0.4.3 collections API；两项 ID metrics 独立计算参考 ID overlap。

Reranker 改变了 16/20 个最终 context 排列：11 个改变集合，5 个仅改变顺序，4 个完全相同。
参考覆盖改善/不变/退化为 **0/20/0**，ID Recall 均为 1.0。观察到 LLM Context Precision 均值增加 0.05，但 Faithfulness 和 Answer Relevancy 均值下降，未证明回答质量整体改善。
EVAL-018 在最终 contexts 完全相同时仍出现 LLM Context Precision 变化，说明该分数差异不能全部归因于 reranking。

限制：只有 20 samples、5 chunks；K=10 大于 corpus size，实际候选为 5。Answer/Judge 同用 `deepseek-chat`，可能共享偏差；LLM metrics 有运行波动。这是每模式一次的观察比较，不支持统计显著性或大规模泛化结论。ID precision 的 1/3 表示 Top-3 中命中一个参考 context，不是回答准确率。

完整设置、逐样本差异、负向案例与正式结果见 [Evaluation Report](backend/evaluation/results/TASK006_EVALUATION_REPORT.md) 和 [Comparison JSON](backend/evaluation/results/task006_comparison.json)。

在 backend 目录重现（全量评估调用付费 API；未来运行应使用新文件名保留本次证据）：

```powershell
New-Item -ItemType Directory -Force evaluation/results
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode baseline --output evaluation/results/task006_baseline.json
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode reranker --output evaluation/results/task006_reranker.json
..\venv\Scripts\python.exe -B -m scripts.compare_rag_evaluations --baseline evaluation/results/task006_baseline.json --reranker evaluation/results/task006_reranker.json --output evaluation/results/task006_comparison.json --report evaluation/results/TASK006_EVALUATION_REPORT.md
```

Comparison 只读取保存的 JSON，不调用 API、不加载模型。依赖、凭据配置与指标定义见 [Evaluation README](backend/evaluation/README.md)。

## 项目目录

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── exceptions/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   ├── alembic/
│   ├── tests/
│   │   └── integration/
│   ├── Dockerfile
│   ├── pytest.ini
│   └── requirements.txt
├── compose.yaml
└── README.md
```

## 快速启动：Docker Compose

前提：已安装并启动 Docker Desktop，且可使用 Docker Compose。

首次启动时，在项目根目录执行：

```powershell
Copy-Item .env.compose.example .env.compose
Copy-Item backend\.env.example backend\.env
```

然后根据模板填写 PostgreSQL 配置和 DeepSeek API key 等必要配置，不要将真实密码或 API key 提交到 Git。由于 `smartdoc_postgres_data` 是 external volume，首次启动前还需要创建一次：

```bash
docker volume create smartdoc_postgres_data
```

该 volume 一般只需首次创建一次。

在项目根目录执行：

```bash
docker compose up --build
```

Compose 会启动 `backend`、`postgres` 和 `redis`。Backend 容器启动后会先执行 Alembic migration，迁移成功后再启动 Uvicorn。

访问地址：

- FastAPI：<http://localhost:8000>
- Swagger：<http://localhost:8000/docs>

当前 Compose 端口映射：

- Backend：`localhost:8000` → `8000`
- PostgreSQL：`localhost:5433` → `5432`
- Redis：`localhost:6380` → `6379`

Backend 容器内部通过 `postgres:5432` 和 `redis:6379` 访问依赖服务。PostgreSQL 数据使用外部持久化 volume `smartdoc_postgres_data`；ChromaDB 数据通过 `./backend/chroma` 挂载到容器。

## 本地开发启动

从项目根目录创建并激活虚拟环境：

```bash
python -m venv venv
```

Windows PowerShell：

```powershell
.\venv\Scripts\Activate.ps1
```

安装 Backend 依赖并启动：

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

本地启动前请准备 `backend/.env`，并确保 PostgreSQL、Redis 和 DeepSeek 配置可用。

## 环境变量

应用配置由 `backend/app/core/config.py` 中的 Settings 管理，默认从当前运行目录的 `.env` 读取。

### `backend/.env`

本地 Backend 运行配置。仓库已提供不含真实密钥的 `backend/.env.example`，可按需复制后填写本地配置。

| 变量 | 必填 | 默认值 / 说明 |
| --- | --- | --- |
| `APP_ENV` | 否 | `development` |
| `LOG_LEVEL` | 否 | `INFO` |
| `DATABASE_URL` | 是 | PostgreSQL 连接 URL |
| `REDIS_HOST` | 否 | `localhost`；Compose 内由环境变量覆盖为 `redis` |
| `REDIS_PORT` | 否 | `6380`；Compose 内由环境变量覆盖为 `6379` |
| `REDIS_DB` | 否 | `0` |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API key，不应提交到 Git |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` |
| `EMBEDDING_MODEL` | 否 | `BAAI/bge-small-zh-v1.5` |
| `CHROMA_COLLECTION_NAME` | 否 | `smartdoc` |
| `SESSION_EXPIRE_SECONDS` | 否 | `3600` |

### `backend/.env.test`

仅用于 PostgreSQL integration tests。配置 `TEST_DATABASE_URL`，并指向独立的 `smartdoc_test` 数据库。仓库已提供 `backend/.env.test.example`。

### `.env.compose`

项目根目录下的 Compose 配置文件，用于提供 PostgreSQL 容器的 `POSTGRES_USER`、`POSTGRES_PASSWORD` 和 `POSTGRES_DB`。仓库已提供 `.env.compose.example`，请在本地填写实际值，不要将密码提交到 Git。

## 数据库迁移

数据库 schema 使用 Alembic 管理。进入 `backend` 目录后执行：

```bash
python -m alembic upgrade head
```

正式初始化和升级应使用 Alembic migration，不使用 `Base.metadata.create_all()` 代替迁移流程。Docker Compose 的 Backend 启动命令会自动执行 `python -m alembic upgrade head`。

## API

主要 API 如下：

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/documents/upload` | 上传并处理文档 |
| `GET` | `/api/documents` | 分页查询文档列表 |
| `GET` | `/api/documents/{document_id}` | 查询文档详情 |
| `PUT` | `/api/documents/{document_id}` | 更新文档并重新处理 |
| `DELETE` | `/api/documents/{document_id}` | 删除文档及其向量 |
| `POST` | `/api/chat` | 基于文档 context 和会话历史进行问答 |

## Supported Document Formats

当前支持：

- PDF
- TXT
- Markdown
- DOCX

DOCX 当前支持提取：

- Paragraph text
- Table text

当前不支持：

- DOCX image OCR
- `.doc`
- Scanned PDF OCR

## Testing

在 `backend` 目录执行：

```bash
python -m pytest -m "not integration"
python -m pytest
```

当前测试套件共 50 个测试。integration tests 使用独立的 `smartdoc_test` PostgreSQL 数据库；测试通过 transaction rollback 清理测试数据，不保留测试记录。

## GitHub Actions CI

`.github/workflows/ci.yml` 在 push 和 pull request 时运行：

```text
push / pull request
        ↓
Ubuntu runner
        ↓
PostgreSQL 17 service
        ↓
安装 requirements.txt
        ↓
Alembic upgrade head
        ↓
pytest
```

当前 workflow 只执行 Backend 测试，没有声明 deploy 或 CD 流程。

## Docker

Docker Compose 当前管理以下服务：

- `backend`
- `postgres`
- `redis`

Backend 使用 `backend/Dockerfile` 构建，依赖来自 `backend/requirements.txt`。容器内通过服务名访问 `postgres:5432` 和 `redis:6379`；PostgreSQL 使用持久化 volume 保存数据。

## Known Limitations

- PostgreSQL 与 ChromaDB 之间没有分布式事务保证，跨存储操作失败时可能需要后续补偿或对账。
- ChromaDB 当前使用本地 persistent storage，主要面向单实例环境。
- DOCX 不解析图片文字。
- 扫描型 PDF 没有 OCR。
- Embedding model 初始化较重。

## Future Improvements

- Cross-storage compensation / reconciliation
- Lazy model initialization
- Docker image optimization
- OCR
- Production deployment
- Scalable vector database
