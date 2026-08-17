''''mermaid



%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#7B2FBE',
    'primaryTextColor': '#FFFFFF',
    'primaryBorderColor': '#5B1FA3',
    'lineColor': '#9B59B6',
    'secondaryColor': '#2E86C1',
    'tertiaryColor': '#1ABC9C',
    'clusterBkg': '#1A1A2E',
    'clusterBorder': '#7B2FBE',
    'edgeLabelBackground': '#1A1A2E',
    'nodeBorder': '#4A148C'
  }
}}%%

graph TB
    subgraph USER["👤 USER INTERFACE"]
        DASH["📊 Streamlit Dashboard<br>• Real-time telemetry<br>• Prediction errors<br>• S3 write count<br>• Force REM Sleep button"]
    end

    subgraph WAKE["🌅 WAKE LOOP (Active Inference)"]
        direction TB
        SIM["🔄 Simulator<br>• CPU/RPS telemetry<br>• Anomaly triggers"]
        PRED["🧠 Predictor<br>• Simple regression<br>• Bedrock for complex<br>• Rolling history"]
        NEURO["⚡ Neuromodulation<br>• Prediction error (0-1)<br>• Surprise threshold: 0.35"]
        ACT["🎯 Action Engine<br>• Block IP<br>• Scale resources<br>• Rate limit"]
        
        SIM --> PRED
        PRED --> NEURO
        NEURO -->|"Surprised?"| ACT
    end

    subgraph FAST["⚡ FAST MEMORY (Hippocampus)"]
        S3["📦 Amazon S3<br>• Raw episode storage<br>• JSON format<br>• TTL expiration<br>• Fast writes"]
    end

    subgraph SLOW["🐘 SLOW MEMORY (Neocortex)"]
        CRDB["CockroachDB + pgvector<br>• semantic_memory table<br>• 1536-dim embeddings<br>• Cosine similarity (<=>)<br>• Permanent storage"]
    end

    subgraph SLEEP["🌙 SLEEP CYCLE (Consolidation)"]
        direction LR
        LAMBDA["⚡ AWS Lambda<br>• Triggered by EventBridge<br>• Orchestrates sleep"]
        SUMMARIZE["🤖 Bedrock Claude<br>• Summarizes anomalies<br>• Extracts patterns"]
        EMBED["📐 Bedrock Titan<br>• Generates embeddings<br>• 1536-dim vectors"]
        
        LAMBDA --> SUMMARIZE
        SUMMARIZE --> EMBED
    end

    subgraph INTROSPECT["🔍 MCP SERVER (Introspection)"]
        MCP["🧠 MCP JSON-RPC<br>• 'Why did you predict?'<br>• Schema traversal<br>• Provenance tracking"]
        PROV["🔗 Provenance Graph<br>• Schema → Episodes<br>• Trace reasoning"]
    end

    subgraph META["🧬 META-PLASTICITY"]
        CCLOUD["☁️ ccloud CLI<br>• Cluster scaling<br>• Auto-provisioning"]
        HARDEN["📈 Stability Counter<br>• alpha = base * 2^-stability<br>• Knowledge hardens"]
    end

    subgraph INFRA["🏗️ INFRASTRUCTURE"]
        ECS["🐳 ECS/Fargate<br>• Agent runtime<br>• 24/7 wake loop"]
        EVAL["📊 Evaluation<br>• Baseline RAG<br>• Two curves"]
    end

    %% Connections
    WAKE -->|"Store surprising events"| FAST
    FAST -->|"Read raw episodes"| SLEEP
    SLEEP -->|"Write consolidated rules"| SLOW
    SLOW -->|"Query for patterns"| WAKE
    INTROSPECT -->|"Traverse"| SLOW
    INTROSPECT -->|"Explain"| USER
    META -->|"Scale cluster"| CRDB
    META -->|"Monitor density"| SLOW
    INFRA -->|"Host"| WAKE
    INFRA -->|"Host"| SLEEP

    %% Styling
    classDef awake fill:#2E86C1,stroke:#1B4F72,color:#fff
    classDef fast fill:#1ABC9C,stroke:#0E6655,color:#fff
    classDef slow fill:#7B2FBE,stroke:#4A148C,color:#fff
    classDef sleep fill:#E67E22,stroke:#A04000,color:#fff
    classDef mcp fill:#E74C3C,stroke:#78281F,color:#fff
    classDef meta fill:#F1C40F,stroke:#7D6608,color:#000
    classDef user fill:#2ECC71,stroke:#1E8449,color:#000

    class SIM,PRED,NEURO,ACT awake
    class S3 fast
    class CRDB slow
    class LAMBDA,SUMMARIZE,EMBED sleep
    class MCP,PROV mcp
    class CCLOUD,HARDEN meta
    class DASH user
    class ECS,EVAL infra


''''
