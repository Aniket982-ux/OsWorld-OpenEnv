---
title: OsWorld-OpenEnv
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---
# OsWorld Data Cleaning Environment

> [!TIP]
> This environment is specifically designed for Training and Benchmarking Large Language Models (LLMs) on complex, multi-step Data Engineering and Cleaning tasks.

---

## 🧠 Architecture Overview

The **OsWorld Data Cleaning Environment** is built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework, providing a robust Markov Decision Process (MDP) for agentic research. 

- **State Representation**: Comprehensive environment state including workspace file contents and task descriptions.
- **Action Space**: Structured Pydantic actions for file inspection, Python execution within a secure sandbox, and automated utility tools.
- **Semantic Grading**: A multi-component Φ (Phi) score that evaluates content, schema, validity, and constraints.
- **Reward Shaping**: Potential-based reward shaping with regression penalties and efficiency-scaled terminal bonuses.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the `uv` package manager installed:
```bash
pip install uv
```

### 2. Installation
Clone the repository and sync dependencies:
```bash
uv sync
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
# Required for inference.py
HF_TOKEN=your_huggingface_token
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
API_BASE_URL=https://router.huggingface.co/v1

# Required for baseline.py (OpenRouter fallback)
OPENROUTER_API_KEY=your_key
```

---

## 📋 Task Scenarios

The environment features **15 procedurally generated variants** across three difficulty tiers. Each `reset()` call cycles through these tiers to challenge the agent's generalization capabilities.

### 🌟 Easy Tier (4 Variants)
| Variant | Description |
| :--- | :--- |
| **Duplicate Removal** | Standardize column names (`id`, `name`) and deduplicate records. |
| **Format Normalization** | Whitespace stripping and lowercase string standardization. |
| **Type Coercion** | Converting semantic strings (e.g., "18 yrs") into native Python types. |
| **Column Rename** | Pure schema alignment for mixed-case and non-standard headers. |

### ⚡ Medium Tier (7 Variants)
| Variant | Description |
| :--- | :--- |
| **Missing Imputation** | Intelligent null handling and redundant column pruning. |
| **Schema Repair** | Advanced mapping of unstructured headers to target specs. |
| **Constraint Enforcement** | Value clamping (e.g., `[0, 100]`) and cross-row uniqueness. |
| **Multi-File Join** | Clean-and-Join operations across `users.csv` and `orders.csv`. |
| **JSON Normalization** | Flattening ragged, deeply nested JSON into tabular CSVs. |
| **SQL Extraction** | Loading SQL dumps into SQLite and exporting a normalized join. |
| **HTML Scraping** | Parsing messy HTML tables into clean structured formats. |

### 🔥 Hard Tier (4 Variants)
| Variant | Description |
| :--- | :--- |
| **Pipeline Recovery** | Complex multi-failure recovery (nulls + dups + corrupt strings). |
| **Adversarial Data** | Resolving biologically/semantically impossible data values. |
| **Cascading Pipeline** | Multi-file dependency logic (currency conversion rates). |
| **Log Parsing** | Extracting structured metrics from unstructured system logs. |

---

## 📊 Evaluation & Mechanics

### Semantic Grading System (Φ)
Unlike simple string matching, our grader uses a weighted combination of four orthogonal components:

| Component | Weight | Metric |
| :--- | :--- | :--- |
| **Content** | 40% | F1-Score row matching via soft-normalized merge. |
| **Schema** | 20% | Column Jaccard overlap + strict order bonus. |
| **Validity** | 20% | Null-ratio, type correctness, and formatting quality. |
| **Constraints** | 20% | Uniqueness and valid range satisfaction. |

> [!IMPORTANT]
> **Anti-Cheat**: An `extra_row_penalty` is applied to prevent agents from dumping excessive junk rows to game the precision/recall metrics.

### Reward Shaping
The environment provides a shaped reward $R$ calculated as:
$$R = \text{Step Penalty} + \Delta\Phi + \text{Regression Penalty} + \text{Terminal Bonus}$$

---

## 🚀 Usage

### Benchmarking (Hackathon Mode)
Run the strictly validated benchmarking script to produce standard `[START]`, `[STEP]`, `[END]` logs:
```bash
uv run inference.py
```

### Evaluation Suite
Verify the robustness of the graders and reward shaping:
```bash
uv run python eval.py
```

### Standalone Server
Run the FastAPI environment server for custom agent development:
```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 📂 Project Structure
```text
OsWorld/
├── Build_process/         # Technical deep-dives and design docs
├── server/                # Core Environment Implementation
│   ├── tasks.py           # 15 Procedural Task Generators
│   ├── graders.py         # Semantic Phi-Grader Logic
│   └── OsWorld_environment.py # State Machine & Sandbox Logic
├── models.py              # Pydantic Action/Observation Specs
├── client.py              # High-performance WebSocket Client
├── inference.py           # Standard Benchmarking Script
└── openenv.yaml           # OpenEnv Manifest
```

---
*Developed for the OpenEnv Hackathon | Built for Scalable Agent Research.*