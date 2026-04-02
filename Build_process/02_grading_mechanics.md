# Multi-Component Semantic Grader

Unlike traditional code-evaluation systems that rely on brittle binary grading (exact byte-for-byte text matching), this environment implements an autonomous **Multi-Component Semantic Grader** with anti-cheat protections.

## The Problem With Simple Grading

### String Matching (Too Brittle)
If the environment expected `1,alice` but the agent produced `1.0,alice`, exact checks would fail even though the task was semantically solved.

### Merge-Only Scoring (Too Weak)
A simple inner-merge count can be exploited:
- Agents can **duplicate correct rows** to inflate join counts
- Agents can **delete problematic rows** and still score well on the remaining matches
- **Schema mismatches** (wrong column names) are not penalized
- **Constraint violations** (duplicate IDs, out-of-range values) are ignored

## The Multi-Component Architecture

The grader computes a composite score $\Phi$ from four orthogonal components minus an anti-cheat penalty:

$$\Phi = 0.4 \cdot \text{content} + 0.2 \cdot \text{schema} + 0.2 \cdot \text{validity} + 0.2 \cdot \text{constraints} - \text{penalty}$$

### 1. Content Score (40% weight) — F1-Based Row Matching

Uses precision and recall via inner merge to evaluate row correctness:

```python
merged = agent_df.merge(expected_df, how='inner')
recall    = matched / expected_rows   # Punishes deleting correct rows
precision = matched / agent_rows      # Punishes adding junk rows
content   = F1(precision, recall)     # Harmonic mean
```

Both sides are deduplicated before merging to prevent score inflation.

### 2. Schema Score (20% weight) — Column Correctness

Evaluates whether the agent produced the correct column structure:

- **Jaccard similarity** of column name sets (handles missing/extra columns)
- **Order bonus** (+0.2) if columns are in the exact expected order
- Missing columns → lower Jaccard
- Extra columns → lower Jaccard

### 3. Validity Score (20% weight) — Data Quality

Checks the cleanliness of the actual cell values:

- **Null check**: No nulls in required columns (per task constraints)
- **Type correctness**: Numeric columns should parse as numbers
- **Formatting**: String columns should be stripped and lowercase (no trailing spaces, consistent casing)

Each check produces a ratio (0.0 to 1.0) and they are averaged.

### 4. Constraint Score (20% weight) — Rule Satisfaction

Verifies that the output respects domain rules defined per task:

- **Uniqueness**: Specified ID columns have no duplicate values
- **Range compliance**: Numeric columns fall within defined bounds (e.g., `val` in `[0, 100]`)
- **Required fields**: All expected columns are present

### 5. Extra Row Penalty — Anti-Cheat

Penalizes agents that add junk rows beyond the expected count:

$$\text{penalty} = \min\left(0.3, \; 0.1 \times \frac{\max(0, \; n_{agent} - n_{expected})}{n_{expected}}\right)$$

Capped at 0.3 to prevent the total score from going excessively negative.

## Anti-Cheat Matrix

| Exploit Attempt | What Happens |
|----------------|-------------|
| Delete all rows | content recall = 0 → $\Phi \approx 0.0$ |
| Add fake/junk rows | precision drops + extra row penalty |
| Duplicate correct rows | deduplication before merge prevents inflation |
| Output partial dataset | recall drops proportionally |
| Wrong column names | schema Jaccard drops |
| Keep nulls in required fields | validity score drops |
| Values out of range | constraint score drops |

## Score Publishing

The internal score (from `0.0` to `1.0`) is passed through the Pydantic type models into the OpenEnv `OsworldObservation.score` field, guaranteeing safe telemetric transmission to the client.