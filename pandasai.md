graph TD
    subgraph Data Ingestion & Setup
        A[patients.csv] -->|Loaded as| B[Pandas DataFrame: df];
        C[dictionary.csv] -->|Processed into| D[Python Dict: field_descriptions];
    end

    subgraph LLM Agent & Execution Loop
        E[User Query] --> F[REPL CLI Interface];
        F --> G[PandasAI Agent / LangChain];
        
        G -->|Configures Agent| H{PandasAI Agent Config};
        H -->|Injects Context| G;
        D --> H;
        B --> G;
        
        G -->|Sends Prompt + Context| I[OLLAMA: Llama3.1 8B];
        I -->|Generates Pandas/Python Code| G;
        
        G -->|Executes Code| J[Code Execution Engine PandasAI];
        J -->|Accesses Data| B;
        J -->|Numerical Result| G;
    end
    
    G -->|Synthesizes Final Answer| K[Final Answer];
    K --> F;

    style D fill:#e0f7fa,stroke:#00bcd4
    style G fill:#f3e5f5,stroke:#9c27b0
    style J fill:#bbdefb,stroke:#2196f3