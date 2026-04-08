# Episode Scenarios and Difficulty Modes

The Data Cleaning Environment presents AI agents with simulated programmatic challenges of varying complexity. The agent is exposed to `data.csv` files via the environment state and must issue Python execution commands to clean the data.

## Task Architecture

There are **15 task variants** grouped into 3 difficulty tiers. The environment cycles through tiers automatically on each `reset()` call, advancing the variant within each tier sequentially.

### 1. Easy Tier (4 variants)
Fundamental schema and targeted parsing corrections.

#### Variant A: Duplicate Removal
- **Goal:** Standardize column names to `id` and `name` and remove duplicate records.

#### Variant B: Format Normalization
- **Goal:** Rename columns, strip whitespace and normalize strings to lowercase.

#### Variant C: Type Coercion
- **Goal:** Convert semantic strings (e.g., '18 yrs', 'Yes') into true integers and booleans.

#### Variant D: Column Rename Only
- **Goal:** Minimal intervention task; just pure schema standardisation for 'Identifier', 'StudentName', 'TestScore'.

---

### 2. Medium Tier (7 variants)
Tasks combining structural issues with distinct semantic data quality problems.

#### Variant A: Missing Value Imputation
- **Goal:** Rename columns, drop extra columns, and fill missing numeric values with `0`.

#### Variant B: Schema Repair
- **Goal:** Standardize completely non-standard column mappings and drop extraneous flag columns.

#### Variant C: Constraint Enforcement
- **Goal:** Enforce unique `id` and clamp values to valid ranges like `[0, 100]`.

#### Variant D: Multi-File Join
- **Goal:** Schema align `orders.csv` and `users.csv`, inner join on `user_id`, and output a unified tabular structure.

#### Variant E: JSON Normalization
- **Goal:** Flatten deeply nested and ragged structure dictionaries into a clean, standardized tabular dataframe.

#### Variant F: SQL Extraction
- **Goal:** Load SQL dumps into SQLite and exporting a normalized join of `users` and `purchases`.

#### Variant G: HTML Scraping
- **Goal:** Parse messy HTML tables into clean structured formats with standardized column names.

---

### 3. Hard Tier (4 variants)
Multi-step, destructive data recovery scenarios.

#### Variant A: Corrupted Pipeline Recovery
- **Goal:** A dataset fully compromised with string formatting out of alignment, duplicate rows, missing values, and out-of-range constraints.

#### Variant B: Adversarial Corruption
- **Goal:** Syntactically intact but semantically impossible constraints. Clipping boundary logic required.

#### Variant C: Cascading Pipeline
- **Goal:** Multi-file dependency logic. Extract rates, compute new columns, fill bounds, and save to a final composite file.

#### Variant D: Log Parsing
- **Goal:** Extract structured records from unstructured system logs, ignoring lines with invalid metrics.

## Agent Constraints
- The agent only "sees" the files exposed in the `files` observation dict.
- The agent receives a detailed `current_task` description listing all required fixes.
- The agent must use `execute_python` action type with pandas code to manipulate the CSV.
- All actions execute inside a sandboxed Python scope injected with core libraries (`pandas`, `io`, `traceback`).
- The agent is scored by a multi-component grader (see `02_grading_mechanics.md`).