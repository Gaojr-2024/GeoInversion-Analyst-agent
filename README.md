# GeoInversion Analyst Agent (v1.0)

**GeoInversion Analyst Agent** is an intelligent, automated system designed for geophysical literature analysis. Powered by **LangChain** and multimodal LLMs (e.g., Qwen-VL, GPT-4o), it streamlines the workflow from inversion image interpretation to literature retrieval, filtering, downloading, and deep analysis.

This repository hosts **v1.0** of the agent, featuring a modular architecture and robust error handling.

---

## ✨ Key Features

-   **🧠 Multimodal Image Analysis**: Automatically interprets geophysical inversion images (velocity, resistivity, etc.) to generate initial geological hypotheses.
-   **🔍 Intelligent Literature Search**: Dynamically generates search queries based on image analysis and retrieves relevant papers from **OpenAlex**.
-   **⭐ Relevance Scoring**: Uses LLMs to evaluate and score papers based on title, abstract, and geological context, ensuring high-quality references.
-   **⬇️ Smart Downloader**:
    -   **Dual Mode**: Supports **Direct Access** for Open Access (OA) papers and **Institutional Login** (configurable) for subscription access.
    -   **Auto-Fallback**: Automatically switches modes if connection fails.
-   **📄 Deep PDF Extraction**: Extracts figures, captions, and full text from downloaded PDFs for granular analysis.
-   **📝 Automated Reporting**: Generates comprehensive, citation-backed Markdown reports using a Map-Reduce strategy.

---

## 🚀 Quick Start

### 1. Environment Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Gaojr-2024/GeoInversion-Analyst-agent.git
    cd GeoInversion-Analyst-agent
    ```

2.  **Create and activate a new virtual environment**:
    It is highly recommended to use a clean virtual environment to avoid dependency conflicts.
    ```bash
    # Create virtual environment named 'venv'
    python -m venv venv

    # Activate (Windows)
    .\venv\Scripts\activate

    # Activate (Linux/Mac)
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    playwright install
    ```

4.  **Configure Environment Variables**:
    - Create a `.env` file in the root directory (copy from `.env.example` if available, or create new).
    - Required keys:
    ```env
    # Required: Alibaba Cloud / DashScope API Key for Qwen models
    ALIBABA_API_KEY=sk-xxxxxx
    
    # Optional: Other model keys if configured
    # OPENAI_API_KEY=sk-xxxxxx
    ```

---

## 📂 Workflow Guide

The project consists of two main parts that can be run sequentially.

### Part 1: From Image to Paper Download
**Goal**: Analyze an inversion image, search for relevant literature, and download PDFs.

1.  **Prepare Input**:
    - Place your inversion image (e.g., `image.png`) in `data/input/`.
    - Place your geographic info JSON (e.g., `test_geo_info.json`) in `data/input/`.
    - *Note*: If inputs are missing, the system will try to use examples from `examples/`.

2.  **Run Command**:
    ```bash
    # Run the integrated workflow
    python integrated_workflow.py --image data/input/image.png --geo-info data/input/test_geo_info.json
    ```

3.  **Outputs** (in `data/processed/`):
    - `1_image_analysis/`: Image analysis report and hypotheses.
    - `2_literature_search/`: Search queries, scored paper list (`literature_review_scored.md`).
    - `3_paper_download/`: Downloaded PDF files (in `pdfs/` subfolder) and download report.

### Part 2: Deep Literature Analysis
**Goal**: Extract figures and text from the downloaded PDFs, analyze them, and generate a comprehensive report.

1.  **Prerequisites**:
    - Ensure Part 1 has run successfully and PDFs are in `data/processed/3_paper_download/pdfs`.
    - Alternatively, manually place PDF files in that directory.

2.  **Run Command**:
    ```bash
    python main2.py
    ```

3.  **Outputs** (in `data/processed/`):
    - `4_pdf_extraction/`: Extracted figures and text for each paper.
    - `5_final_report/`: Final comprehensive report (`LangChain_Comprehensive_Report.md`) and intermediate summaries.

---

## 🛠️ Project Structure

```text
GeoInversion-Analyst-agent/
├── integrated_workflow.py  # [Part 1 Entry] Image Analysis -> Search -> Download
├── main2.py                # [Part 2 Entry] Extraction -> Analysis -> Report
├── config.py               # Global Configuration
├── requirements.txt        # Python Dependencies
├── upload_to_github.bat    # Helper script for GitHub upload
├── agents/                 # Core Agent Logic
│   ├── workflow1.py        # Image Analysis Agent (Multimodal)
│   ├── download.py         # Literature Search & Download Orchestrator
│   ├── run_sep_download.py # Standalone/Sub-process Download Logic
│   ├── input_processor.py  # Input Data Processing
│   └── multimodal_analyzer.py # Vision Model Interaction
├── app/                    # Application Components (Chains, Pipelines)
│   ├── chains/             # LangChain Chains
│   ├── core/               # Core Utilities
│   └── tools/              # Custom Tools (PDF Extraction etc.)
├── data/                   # Data Directory (Ignored by Git except examples)
│   ├── input/              # Place input images/json here
│   └── processed/          # Generated outputs (organized by step)
└── prompts/                # LLM Prompt Templates
```
