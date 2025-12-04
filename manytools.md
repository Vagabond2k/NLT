graph TD
    subgraph Data Ingestion & Storage
        A[patients.csv] -->|Loaded into| B[DuckDB: patients table];
        C[index.csv Dictionary] -->|Processed & Embedded| D[ChromaDB: schema_store];
    end

    subgraph User Interface & Orchestration
        E[User Query] --> F[LangChain REPL CLI];
        F --> G[Ollama: Llama 3.1 8B Agent];
    end

    subgraph Agent Tools & Data Flow
        subgraph Semantic RAG Knowledge
            G -->|Call schema_qa / lookup| H[Tool: schema_qa / schema_lookup];
            H -->|Query Schema| D;
        end
        
        subgraph Data Diagnostics Exploration
            G -->|Call peek_rows / distinct_values| I[Tool: peek_rows / distinct_values];
            I -->|Access Data/Encodings| B;
        end
        
        subgraph Analytical Operation Computation
            G -->|Call duckdb_query / column_profile| J[Tool: duckdb_query / column_profile];
            J -->|Execute SQL| B;
        end
        
        subgraph Post-Processing
            G -->|Call python_calc| K[Tool: python_calc];
        end
        
        D -->|Schema Context| G;
        B -->|Diagnostic Results| I;
        B -->|SQL Results| J;
        K -->|Arithmetic Result| G;
    end
    
    G -->|Synthesizes Final Answer| L[Final Answer];
    L --> F;