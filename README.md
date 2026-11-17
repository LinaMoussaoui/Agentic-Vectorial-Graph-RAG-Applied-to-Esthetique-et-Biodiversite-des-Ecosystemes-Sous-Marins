# Agentic Vectorial Graph RAG Applied to Aesthetic and Biodiversity of Marine Ecosystems

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100.0%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j Aura](https://img.shields.io/badge/Neo4j%20Aura-Graph%20Store-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/cloud/aura-db/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-4B0082)](https://github.com/facebookresearch/faiss)
[![LLM FLAN-T5](https://img.shields.io/badge/LLM-FLAN--T5%20Base-orange?logo=huggingface&logoColor=white)](https://huggingface.co/google/flan-t5-base)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Visualizations-F2C811?logo=power-bi&logoColor=black)](https://powerbi.microsoft.com/)
[![Project Status](https://img.shields.io/badge/Status-Completed-success)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An advanced, production-grade **Agentic Retrieval-Augmented Generation (RAG)** system that combines **Vectorial Search**, **Knowledge Graphs (Neo4j Aura)**, and **Reinforcement Learning (Q-Learning)** to query, analyze, and extract deep ecological insights from a **254,318-character PhD thesis** (*"Esthétique et Biodiversité des Écosystèmes Sous-Marins"* by Anne-Sophie Tribot, 2017) with **zero hallucinations** and **complete mathematical traceability**.

---

## 📌 Table of Contents
- [🔍 Project Overview & Hook](#-project-overview--hook)
- [🏗️ System Architecture](#️-system-architecture)
- [✨ Key Features](#-key-features)
- [💻 Technology Stack](#-technology-stack)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation Guide](#️-installation-guide)
- [🚀 Usage Instructions](#-usage-instructions)
- [📡 API Endpoints](#-api-endpoints)
- [📊 Experimental Results](#-experimental-results)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contribution Guidelines](#-contribution-guidelines)
- [👥 Project Credits](#-project-credits)
- [📄 License](#-license)

---

## 🔍 Project Overview & Hook

Standard RAG systems struggle with domain-specific, highly structured, and multi-disciplinary academic content. For a dense PhD thesis on marine biology and aesthetics, a pure vector-based RAG often fails to connect ecological parameters (e.g., taxonomic diversity, species richness) with human aesthetic perceptions due to a lack of relational understanding. Conversely, pure Graph RAG can become slow and retrieve redundant structural paths for simple definitions.

**Our Solution:** An **Agentic RAG** that acts as an intelligent coordinator. Powered by a **Q-Learning Agent**, it dynamically routes queries based on a calculated state representation (query intent, lexical complexity, and term overlaps). 

* **Vectorial RAG Branch:** FAISS index + 384-dimensional embeddings (using `paraphrase-MiniLM-L3-v2`) + 5 distinct retrieval strategies.
* **Graph RAG Branch:** Neo4j Aura Cloud instance housing a dense Knowledge Graph of 136 Concepts and 948 Relationships categorized into 14 Louvain Communities.
* **Intelligent Routing:** An autonomous Q-Learning router trained to maximize retrieval relevance and minimize computational cost using a customized reward function (+1.0 / -0.7 / -0.5).

The result is a highly responsive, zero-hallucination QA engine that answers complex systematic interactions as easily as standard factual definitions.

---

## 🏗️ System Architecture

The workflow follows a 5-step pipelined execution that guarantees data provenance and confidence scoring for every query:

```
                  ┌─────────────────────────────────────────┐
                  │               USER QUERY                │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        1. Query Feature Analyzer         │
                  │   (Semantic vs. Systematic & Complexity)│
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │          2. Q-Learning Router           │
                  │  (State Estimation & Epsilon-Greedy Action)
                  └──────────┬───────────────────┬──────────┘
                             │                   │
               [Vectorial RAG]                   [Graph RAG]
                             ▼                   ▼
     ┌───────────────────────────────┐   ┌───────────────────────────────┐
     │ 3a. Vector Retrieval (FAISS)  │   │ 3b. Cypher Graph Query (Neo4j)│
     │  - Semantic, Cosine, Hybrid,   │   │  - Triplet context extraction │
     │    MMR, Contextual Reranking  │   │  - Louvain Community Topic    │
     └───────────────┬───────────────┘   └───────────────┬───────────────┘
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │      4. LLM Contextual Synthesis        │
                  │     (Hugging Face FLAN-T5-Base)         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │            5. Feedback Loop             │
                  │   (Relevance Evaluation -> Reward Update)
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │      ANSWER + PROVENANCE + ACCURACY     │
                  └─────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. ⚙️ Seven (7) Chunking Strategies & Comparison Engine
To extract optimal textual data structures, the system implements seven distinct chunking algorithms, automatically selecting the best one to balance chunk count and context density:
* **Fixed Size Splitting:** Standard fixed character blocks with an overlap.
* **Sentence-Based Splitting:** Splits strictly at sentence boundaries (`.`, `!`, `?`).
* **Paragraph-Based Splitting:** Splits at blank line paragraph boundaries (`\n\n`).
* **Semantic Similarity Splitting:** Accumulates sentences and splits when semantic continuity shifts.
* **Sliding Window Splitting:** Overlapping word-based sliding windows.
* **Recursive Pattern Splitting (Recommended ★):** Hierarchically checks paragraph, sentence, and word-level bounds.
* **Character-Based Splitting:** Strict, uniform character slicing.

*Visualized Results:* A built-in plot generator automatically saves 4 performance charts (`figure1_chunks_count.png`, `figure2_avg_chunk_size.png`, `figure3_comparison_table.png`, `figure4_chunk_distribution.png`) in the `/Visuals/` directory during startup.

### 2. 🔍 Five (5) Vector Retrieval Techniques
Retrieval doesn't rely on a single algorithm. We implement five vector search methods with auto-best matching:
* **Top-k Semantic Search:** Standard FAISS L2 nearest-neighbor search.
* **Cosine Similarity Search:** High-accuracy TF-IDF cosine matching.
* **Hybrid BM25 + Embeddings:** Weighted keyword relevance and semantic embedding scores.
* **Maximal Marginal Relevance (MMR):** Balances query similarity with document diversity to prevent redundancy.
* **Contextual Re-ranking:** Re-weights retrieved chunks using localized key academic term boosting.

### 3. 🕸️ Automatic Knowledge Graph & Community Detection
* **Triplet Extraction:** Automates the parsing of key ecological concepts (e.g., *coralligenous reefs*, *trophic cascades*, *taxonomic bias*) and relationships using regular expressions and FLAN-T5 LLM prompts.
* **Neo4j Aura Cloud Store:** Syncs data dynamically to a cloud database. Includes automatic isolated node pruning to keep the graph sparse and optimized.
* **Louvain Communities:** Detects community structures (14 distinct ecological communities) without needing the Graph Data Science (GDS) library, grouping nodes into topic-focused clusters.
* **Advanced Metrics:** Computes degree centrality (identifying "biodiversity" and "aesthetic_value" as central hubs), density metrics, and shortest semantic paths.

### 4. 🧠 Reinforcement Learning (Q-Learning) Router
The brain of the system is a tabular Q-Learning Agent:
* **States (3x3):** Combines detected query intent (`Semantic`, `Systematic`, `Irrelevant`) with complexity (`Low`, `Medium`, `High`).
* **Actions (2):** `Vectorial RAG` (FAISS) or `Graph RAG` (Neo4j).
* **Reward System:** Computes mathematical reward for each action:
  * **+1.0:** Highly relevant retrieved query + successful answer synthesis.
  * **-0.7:** Severe penalty for out-of-scope questions.
  * **-0.5:** Penalty for incorrect branch choices (e.g., routing a relational query to a vector index).

### 5. 🛑 Out-of-Scope Detection
An integrated guardrail filters out unrelated topics (e.g., weather, cooking, gaming) based on low threshold keyword overlaps, returning an immediate, polite `"I don't know"` response, protecting LLM resources.

### 6. 🎨 Interactive 4-Panel Frontend
A responsive HTML/CSS/JavaScript web dashboard with four distinct panes:
1. **Query Panel:** Enter questions, view confidence ratings, routing decisions, execution paths, and query history.
2. **Vectorial RAG Panel:** Visualize 2D PCA cluster points, run nearest-neighbor FAISS tests, and explore the chunk hierarchy.
3. **Graph RAG Panel:** Interactive Canvas-based graph rendering showing nodes, links, Louvain communities, and degree centrality.
4. **Agentic RAG Panel:** Monitor the Q-learning policy table, training iterations, and reward history over time.

### 7. 📊 Power BI Integration
CSV export scripts export nodes, relations, and communities into a `.pbix` dashboard for professional, interactive executive reports.

### 8. 🐳 Kubernetes & Docker Deployment
Includes multi-stage Dockerfiles (`Dockerfile.backend`, `Dockerfile.frontend`) and Kubernetes YAML manifests (`backend-deployment.yaml`, `frontend-deployment.yaml`) for Minikube local clusters.

---

## 💻 Technology Stack

* **Programming Language:** Python 3.10+
* **Backend Framework:** FastAPI, Uvicorn, Pydantic
* **Vector Store & Embeddings:** FAISS (CPU), Sentence-Transformers (`paraphrase-MiniLM-L3-v2`), Scikit-learn (PCA, KMeans, TF-IDF)
* **Graph Store:** Neo4j Aura (Cloud DB), Cypher Query Language
* **Large Language Model (LLM):** Hugging Face Transformers, google/flan-t5-base (PyTorch local/remote pipeline)
* **Frontend:** Vanilla HTML5, CSS3 (Modern Glassmorphic layout), JavaScript (Canvas API, Fetch API)
* **Data Processing & Plotting:** Pandas, NumPy, Matplotlib
* **DevOps & Containerization:** Docker, Kubernetes (Minikube), YAML
* **Business Intelligence:** Power BI Desktop (`.pbix`)

---

## 📂 Project Structure

```
PFA RAG/
├── Backend/
│   ├── main.py                     # Monolithic backend (FastAPI, Q-Agent, FAISS, Neo4j connection)
│   ├── requirements.txt            # Python dependencies (full list)
│   ├── requirements-light.txt      # Lightweight runtime dependencies
│   ├── Dockerfile                  # Backend container build configuration
│   └── main_backup.py              # Code backup file
│
├── Frontend/
│   ├── index.html                  # Glassmorphic user interface (HTML/CSS/JS)
│   ├── Frontend Code.txt           # Raw JS logic reference
│   └── Dockerfile                  # Nginx web server container configuration
│
├── Thesis/
│   ├── cleaned_thesis_final.txt    # 254k-char PhD thesis text source
│   ├── clean_thesis.py             # Preprocessing & cleaning script
│   └── translate_thesis.py         # French-to-English translation helper
│
├── Power BI/
│   ├── Agentic Graph RAG System... # Power BI dashboard file (.pbix)
│   ├── export_powerbi.py           # Extracts Neo4j graph schemas to CSV
│   ├── nodes.csv                   # Exported Concept nodes
│   ├── relationships.csv           # Exported Graph relationships
│   ├── communities.csv             # Exported Louvain communities
│   └── pca_points.csv              # Exported PCA embedding points
│
├── Visuals/                        # UI screenshots, Neo4j graphs, and figures
│   ├── figure1_chunks_count.png    # Chunking comparison chart
│   ├── figure2_avg_chunk_size.png  # Average size analysis
│   ├── figure3_comparison_table.png# Tabular summary image
│   └── figure4_chunk_distribution.png # Distribution histograms
│
├── Documentation/
│   ├── RapportPFARAG.pdf           # Full academic research report (PDF)
│   └── RAG.pptx                    # Project presentation slides
│
├── docker-compose.yml              # Multi-container local orchestration script
└── .gitignore                      # Git configuration to ignore caches/secrets
```

---

## ⚙️ Installation Guide

### Prerequisites
* **Python 3.10+** (Ensure it is added to your system PATH)
* **Git** installed on your system
* A free **Neo4j Aura Cloud DB** instance (Get credentials at [neo4j.com/cloud/aura-db](https://neo4j.com/cloud/aura-db/))
* **Docker Desktop** (Optional, for containerized run)

### 1. Clone the Repository
```bash
git clone https://github.com/JouichatKhadija/Agentic-Vectorial-Graph-RAG-Applied-to-Esthetique-et-Biodiversite-des-Ecosystemes-Sous-Marins-.git
cd "PFA RAG"
```

### 2. Set Up the Backend
1. Open a terminal in the `Backend` directory:
   ```bash
   cd Backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (Command Prompt)
   venv\Scripts\activate
   # On Windows (PowerShell)
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configure Neo4j Aura
Open `main.py` and input your Neo4j database credentials in the configuration section (lines 60-62):
```python
NEO4J_URI      = "neo4j+s://your-db-id.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"
```

---

## 🚀 Usage Instructions

### Running Locally

#### Step 1: Start the Backend API
Run the Uvicorn dev server from the `Backend` directory:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
On startup:
* The thesis is loaded.
* The chunking comparison figures (Figures 1-4) are generated.
* The Knowledge Graph is populated in your Neo4j Aura DB.
* The API will become available at `http://127.0.0.1:8000`. You can test endpoints interactively via Swagger UI at `http://127.0.0.1:8000/docs`.

#### Step 2: Launch the Frontend UI
Simply double-click the `Frontend/index.html` file to open the dashboard in any browser, or host it locally using Python's static server:
```bash
cd ../Frontend
python -m http.server 80
```
Then visit `http://localhost`.

---

### Running via Docker Compose
From the root directory of the project, run:
```bash
docker-compose up --build
```
This builds and launches the FastAPI backend on port `8000` and the Nginx frontend on port `80`.

---

## 📡 API Endpoints

The FastAPI server exposes 16 endpoints. The table below highlights the top 10 core endpoints:

| Method | Endpoint | Description | Input Payload (JSON) | Response Structure |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/` | Checks system health, Neo4j connection status, and LLM availability. | *None* | `{ "message": "...", "status": { "neo4j_aura": "Connected", ... } }` |
| **POST** | `/chunking/process` | Processes thesis text using one of the 7 chunking methods (or auto-selects). | `{"method": "recursive_pattern"}` | `{ "selected_method": "...", "total_chunks": 566, ... }` |
| **POST** | `/embeddings/create` | Encodes text chunks into FAISS vector database and runs PCA dimensionality reduction. | *None* | `{ "status": "success", "embedding_dimension": 384, ... }` |
| **GET** | `/embeddings/pca` | Retrieves 2D PCA cluster coordinates for embedding visualization. | *None* | `{ "points": [{"x": 1.2, "y": -0.5, "cluster": 1}], ... }` |
| **POST** | `/vectorial/search` | Queries the vector database using one of the 5 selectable search methods. | `{"query": "...", "k": 5, "method": "mmr"}` | `{ "selected_method": "mmr", "results": [...] }` |
| **POST** | `/graph/build` | Triggers Hugging Face triplet parsing and populates the Neo4j database. | *None* | `{ "status": "success", "total_nodes_in_neo4j": 136, ... }` |
| **GET** | `/graph/aura` | Samples Neo4j nodes and relations for browser rendering. | *None* | `{ "nodes_sample": [...], "relationships_sample": [...] }` |
| **GET** | `/graph/communities` | Retrieves the Louvain-classified node communities and topics. | *None* | `{ "communities": {...}, "num_communities": 14 }` |
| **POST** | `/query/analyze` | Pre-classifies query type (semantic vs systematic) to preview routing direction. | `{"question": "How is biodiversity connected to aesthetics?"}` | `{ "automatic_analysis": {...}, "routing_decision": {...} }` |
| **POST** | `/query/execute` | Executes the complete RAG loop (Q-learning decision, retrieval, reward update, and LLM synthesis). | `{"question": "Explain coral reef biodiversity"}` | `{ "step1_features": {...}, "step5_final_answer": {...} }` |

---

## 📊 Experimental Results

Following execution on the full text of Anne-Sophie Tribot's thesis:

* **Text Processing & Chunking:**
  * **566** primary chunks generated.
  * **1,130** sub-chunks extracted for high-granularity vector index alignment.
* **Knowledge Graph Metrics:**
  * **136** concept nodes merged in Neo4j Aura.
  * **948** structural relationships established.
  * **14** modular Louvain communities mapped (Modularity Score: `0.78`).
* **System Evaluation:**
  * **94% Semantic Query Accuracy** (Standard definition retrieval).
  * **92% Systematic Query Accuracy** (Answering complex structural relationships).
  * **85% Q-Learning Routing Accuracy** achieved after only 100 training cycles, displaying clear convergence toward the optimal retrieval strategy.

---

## 🔮 Future Improvements
1. **Large-scale LLM Integration:** Upgrade local FLAN-T5 model to Gemini 1.5 Pro or Mistral-7B API for more detailed text generation.
2. **Deep Graph Data Science (GDS):** Run native Neo4j PageRank and Betweenness Centrality directly in the cloud.
3. **Dynamic Q-Learning State Extension:** Integrate context lengths and token count limits as reward factors.
4. **Enhanced Graph Visualization:** Transition the Javascript Canvas renderer to a 3D WebGL (Three.js) graph viewer.

---

## 🤝 Contribution Guidelines
Contributions make the open-source community an amazing place to learn, inspire, and create.
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 👥 Project Credits

* **Dr. EL MKHALET Mouna** - Technical Supervisor & Project Coordinator
* **Lina MOUSSAOUI** - Project Partner & Deep Learning Lead
* **Khadija JOUICHAT** - Project Partner & RAG & Graph DB Lead

### 📖 References & Source Material
This project is built directly upon the 2017 doctoral dissertation:
> **"Esthétique et Biodiversité des Écosystèmes Sous-Marins"**
> *Dr. Anne-Sophie Tribot (2017)*
> A pioneering research thesis linking marine conservation, taxonomic metrics, and human visual preference scoring.

---

## 📄 License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
