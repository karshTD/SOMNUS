```markdown
# 🌙 SOMNUS

**Self-Optimizing Memory Network for Unifying Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CockroachDB](https://img.shields.io/badge/CockroachDB-Ready-6933FF.svg)](https://www.cockroachlabs.com/)
[![AWS](https://img.shields.io/badge/AWS-Ready-FF9900.svg)](https://aws.amazon.com/)

---

## 📖 The One-Liner

> *"SOMNUS is a brain-inspired AI agent that sleeps, dreams, and consolidates memories — just like a human."*

---

## 🧠 What Is SOMNUS?

SOMNUS is an AI agent with a dual-memory system inspired by neuroscience:

| Memory System | Analogy | Storage | What It Does |
|---------------|---------|---------|--------------|
| **Hippocampus** | Sticky Notes | S3 (Fast) | Stores raw experiences as they happen |
| **Neocortex** | Hardcover Book | CockroachDB (Slow) | Stores consolidated patterns and rules |

The magic happens during **sleep consolidation** — the agent replays recent experiences, extracts patterns, and stores them permanently.

---

## 🎯 Why SOMNUS?

### The Problem
Most AI agents today are stateless. They don't learn from experience. They forget everything between conversations.

### The Solution
SOMNUS learns continuously:

1. **While Awake:** The agent watches, predicts, and stores surprising events
2. **While Sleeping:** The agent replays, consolidates, and extracts patterns
3. **The Next Day:** The agent recognizes patterns instantly — zero panic, zero hesitation

### The Demo
Press the **"Force REM Sleep"** button and watch the agent dream in real-time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SOMNUS ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         WAKE LOOP (Active Inference)                 │ │
│  │                                                                       │ │
│  │  Simulator → Predict → Compare → Surprised? → Action → Store in S3   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     SLEEP CYCLE (Consolidation)                      │ │
│  │                                                                       │ │
│  │  Read S3 → Bedrock Summarize → Embed → Write to CockroachDB         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      DUAL MEMORY SYSTEM                              │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐              ┌─────────────────────────────────┐ │ │
│  │  │   HIPPOCAMPUS   │              │           NEOCORTEX            │ │ │
│  │  │     (S3)        │              │      (CockroachDB + pgvector)  │ │ │
│  │  │  Raw Episodes   │              │      Semantic Rules/Vectors    │ │ │
│  │  │  TTL Expiration │              │      Permanent Storage         │ │ │
│  │  └─────────────────┘              └─────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         MCP SERVER (Introspection)                   │ │
│  │                                                                       │ │
│  │  "Why did you predict that?" → Schema → Provenance → Episodes        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Fast Memory** | AWS S3 |
| **Slow Memory** | CockroachDB + pgvector |
| **AI/Reasoning** | AWS Bedrock (Claude 3.5 Sonnet + Titan Embeddings) |
| **Orchestration** | AWS Lambda |
| **Agent Runtime** | Python 3.10+ / ECS Fargate |
| **Introspection** | MCP Server (JSON-RPC) |
| **Dashboard** | Streamlit |
| **Containerization** | Docker + Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CockroachDB Cloud account ([free tier](https://www.cockroachlabs.com/))
- AWS account ([free tier](https://aws.amazon.com/free/))
- `ccloud` CLI installed

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/somnus.git
cd somnus
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Fill in your credentials:

```bash
# CockroachDB
COCKROACH_DB_URL="postgresql://user:password@cluster.cockroachlabs.cloud:26257/defaultdb"

# AWS
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
AWS_DEFAULT_REGION="us-east-1"
S3_HIPPOCAMPUS_BUCKET="somnus-hippocampus-yourname"

# Optional
MCP_SERVER_URL="http://localhost:8765"
```

### 3. Initialize Database

```bash
python substrate/migrate.py
```

### 4. Run SOMNUS

```bash
# Terminal 1: Agent + MCP Server
python main.py

# Terminal 2: Dashboard
streamlit run dashboard.py
```

### 5. Trigger Sleep (Manual)

```bash
python main.py --sleep-once
```

Or click the **"Force REM Sleep"** button in the dashboard.

### 6. Docker (Optional)

```bash
docker-compose up --build
```

- Agent + MCP: `http://localhost:8765`
- Dashboard: `http://localhost:8501`

---

## 🧪 Running Tests

```bash
# Unit tests (no cloud dependencies)
pytest tests/test_consolidation.py tests/test_neuromod.py -v

# Integration tests (requires live AWS + CockroachDB)
pytest tests/test_connections.py -v
```

---

## 📂 Project Structure

```
somnus/
├── main.py                 # Orchestrator: agent + MCP + wake loop
├── dashboard.py            # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── core/
│   ├── simulator.py        # Fake telemetry + anomaly trigger
│   ├── neuromod.py         # Prediction error calculation
│   ├── agent.py            # SomnusAgent wake loop
│   ├── actions.py          # Remediation skills
│   └── state_store.py      # IPC for dashboard ↔ agent
├── memory/
│   ├── hippocampus.py      # Fast episodic writes to S3
│   └── cortex.py           # Slow semantic recall via pgvector
├── mcp/
│   ├── server.py           # MCP JSON-RPC server
│   └── introspection.py    # Schema, stats, agent state
├── infra/
│   ├── aws_client.py       # S3 + Bedrock (Titan/Claude)
│   └── ccloud.py           # Meta-plasticity via ccloud CLI
├── sleep_cycle/
│   └── lambda_handler.py   # REM: S3 → Bedrock → CockroachDB
├── eval/
│   └── baseline.py         # Baseline RAG vs SOMNUS dual memory
├── substrate/
│   └── migrate.py          # CREATE EXTENSION vector + table
└── tests/
    ├── test_connections.py
    ├── test_consolidation.py
    └── test_neuromod.py
```

---

## 🎬 The Demo (3 Minutes)

| Time | Scene | What Happens |
|------|-------|--------------|
| **0:00-1:00** | **Awake State** | Agent watches, predicts, detects anomaly, stores in S3 |
| **1:00-1:30** | **Sleep State** | Press "Force REM Sleep" → Agent dreams, consolidates |
| **1:30-2:30** | **Next Day** | Mutated anomaly → Agent recognizes pattern instantly |
| **2:30-3:00** | **Meta-plasticity** | Agent grows its own brain via `ccloud cluster scale` |

---

## 🔬 Key Concepts

### 1. Dual-Memory System

| Memory Type | Storage | Speed | Lifespan | Purpose |
|-------------|---------|-------|----------|---------|
| **Hippocampus** | S3 | Fast | Temporary (TTL) | Raw episodes, immediate recall |
| **Neocortex** | CockroachDB | Slow | Permanent | Semantic rules, long-term knowledge |

### 2. Surprise-Gated Encoding

The agent only stores events that surprise it:

```python
if prediction_error > SURPRISE_THRESHOLD:
    hippocampus.write(event)  # Store in fast memory
```

### 3. Sleep Consolidation

The agent replays recent experiences, interleaved with existing knowledge:

```python
new_episodes = hippocampus.sample(batch=32)
old_schemas = cortex.sample(batch=32)

for item in shuffle(new_episodes + old_schemas):
    cortex.merge_or_spawn(item)
```

### 4. Meta-plasticity

Knowledge hardens with repetition:

```python
alpha = ALPHA_BASE * (2 ** -schema.stability)
schema.update(item, alpha=alpha)
schema.stability += 1
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [CockroachDB](https://www.cockroachlabs.com/) — Distributed, always-on database
- [AWS](https://aws.amazon.com/) — Cloud infrastructure
- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity search
- [Streamlit](https://streamlit.io/) — Dashboard framework
- [Complementary Learning Systems Theory](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11172494/) — The neuroscience behind SOMNUS

---

## 🏆 Built For

The [CockroachDB × AWS Hackathon](https://www.cockroachlabs.com/blog/agentic-ai-hackathon/).

---

**🌙 SOMNUS: Sleep. Dream. Learn. Grow.**

---
