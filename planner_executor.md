graph TD
    subgraph Data Ingestion & Storage
        A[patients.csv] -->|Load| B[DuckDB: patients table];
        C[index.csv Dictionary] -->|Process & Embed| D[ChromaDB: schema_store];
    end

    subgraph User Interface
        E[User Query] --> F[LangChain REPL CLI];
    end

    subgraph Agentic Pipeline Planner/Executor
        F --> G{Stage 1: Planner Agent Ollama};
        G -->|Call retrieve_context| H[Tool: retrieve_context];
        H -->|RAG Lookup| D;
        D -->|Schema Context| H;
        H -->|Context & Schema| G;
        G -->|Generates PlannerResult JSON| I[PlannerResult: SQL Query];
        
        I --> J{Stage 2: Executor Agent Ollama};
        J -->|Call run_sql SQL| K[Tool: run_sql];
        K -->|Execute Query| B;
        B -->|SQL Result| K;
        K -->|Result Observation| J;
        J -->|Synthesizes Final Answer| L[Final Answer];
    end
    
    L --> F;
    L --> E;