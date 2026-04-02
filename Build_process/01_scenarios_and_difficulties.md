# Episode Scenarios and Difficulty Modes

The Data Cleaning Environment presents AI agents with simulated programmatic challenges of varying complexity. The agent is exposed to `data.csv` files via the environment state and must issue Python execution commands to clean the data.

## Task Architecture

There are **6 task variants** grouped into 3 difficulty tiers. The environment cycles through variants automatically on each `reset()` call.

### 1. Easy Tier (2 issues per task)

Tasks with a foundational schema issue combined with a focused data quality check.

#### Variant A: Duplicate Removal
- **Scenario:** The target CSV contains exact row duplicates and wrong column casing. 
- **Goal:** Standardize column names to `id` and `name` and remove duplicate records.
- **Start State:** 
  ```csv
  ID,Name
  1,alice
  2,bob
  3,charlie
  4,dave
  2,bob
  3,charlie
  ```
- **Expected End State:** Unique records for each entity with correct column names.

#### Variant B: Format Normalization
- **Scenario:** String fields contain inconsistent formatting — trailing spaces, mixed casing, and wrong column casing.
- **Goal:** Rename columns, strip whitespace and normalize to lowercase.
- **Start State:**
  ```csv
  ID,Name
  1, Alice
  2,BOB
  3, Charlie
  4,  dave
  5,Eve 
  ```
- **Expected End State:** All names stripped and lowercased, columns renamed.

---

### 2. Medium Tier (3-5 issues per task)

Tasks combining structural issues with distinct semantic data quality problems.

#### Variant A: Missing Value Imputation
- **Scenario:** The target CSV has missing values (nulls/NaN) in data columns, wrong column casing, and an extra column.
- **Goal:** Rename columns to `id` and `val`, drop the extra column, and fill missing values with `0`.
- **Start State:**
  ```csv
  Id,Val,extra
  1,10,junk
  2,,junk
  3,30,junk
  4,,junk
  5,50,junk
  6,60,junk
  ```
- **Expected End State:** Correct schema and all blank cells filled with `0`.

#### Variant B: Schema Repair
- **Scenario:** Column names are completely non-standard (e.g., `IDENTIFIER` instead of `id`) and an extra junk column exists.
- **Goal:** Rename columns to match expected schema (`id`, `val`) and drop the extra column.
- **Start State:**
  ```csv
  IDENTIFIER,VALUE,flag
  1,10,y
  2,20,y
  3,30,y
  4,40,y
  5,50,y
  ```
- **Expected End State:** Columns renamed to `id,val`, extra column removed.

#### Variant C: Constraint Enforcement
- **Scenario:** The dataset has duplicate IDs, values outside valid ranges, wrong casing, and an extra column.
- **Goal:** Rename columns, drop extra column, enforce unique `id` (keep first occurrence) and clamp `val` to `[0, 100]`.
- **Start State:**
  ```csv
  Id,Val,extra
  1,10,junk
  1,25,junk
  2,150,junk
  3,-10,junk
  4,80,junk
  5,200,junk
  ```
- **Expected End State:** Unique IDs, values clamped to the valid range.

---

### 3. Hard Tier (7-9 issues combined)

#### Variant A: Corrupted Pipeline Recovery
- **Scenario:** The dataset is fully compromised with **all issue types combined**: three non-standard column names, an extra column, duplicate rows, missing values, inconsistent formatting, and out-of-range constraints.
- **Goal:** Execute a multi-step semantic cleaning strategy addressing all 9 issues.
- **Start State:**
  ```csv
  ID,Name,Value,extra
  1, Alice ,10,junk
  1, ALICE ,10,junk
  2, Bob ,40,junk
  3,CHARLIE ,150,junk
  4, dave,,junk
  5,  EVE,-5,junk
  6,frank ,80,junk
  7, Grace ,90,junk
  ```
- **Expected End State:** A fully clean dataset with correct schema, unique records, filled values, normalized strings, and clamped ranges.

## Agent Constraints
- The agent only "sees" the files exposed in the `files` observation dict.
- The agent receives a detailed `current_task` description listing all required fixes.
- The agent must use `execute_python` action type with pandas code to manipulate the CSV.
- All actions execute inside a sandboxed Python scope injected with core libraries (`pandas`, `io`, `traceback`).
- The agent is scored by a multi-component grader (see `02_grading_mechanics.md`).