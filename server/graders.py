"""
Multi-component Semantic Grader for the OsWorld Data Cleaning Environment.

Phi = 0.4 * content_score    (F1 exact row matching)
    + 0.2 * schema_score     (column names + order)
    + 0.2 * validity_score   (nulls, types, formatting)
    + 0.2 * constraint_score (uniqueness, ranges)
    - extra_row_penalty      (anti-cheat)
"""

import pandas as pd
import io
from typing import Dict, Any


class SemanticGrader:
    """
    Grades data cleaning using four orthogonal components plus anti-cheat penalty.
    """

    W_CONTENT = 0.4
    W_SCHEMA = 0.2
    W_VALIDITY = 0.2
    W_CONSTRAINT = 0.2

    def get_score(
        self,
        files: Dict[str, str],
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        """Compute the multi-component Phi score."""
        content = files.get("data.csv", "")

        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception:
            return 0.0

        # Empty agent output -> 0
        if len(df) == 0 and len(expected_df) > 0:
            return 0.0

        content_score = self._content_score(df, expected_df)
        schema_score = self._schema_score(df, expected_df, constraints)
        validity_score = self._validity_score(df, expected_df, constraints)
        constraint_score = self._constraint_score(df, constraints)
        extra_penalty = self._extra_row_penalty(df, expected_df)

        phi = (
            self.W_CONTENT * content_score
            + self.W_SCHEMA * schema_score
            + self.W_VALIDITY * validity_score
            + self.W_CONSTRAINT * constraint_score
            - extra_penalty
        )

        return min(1.0, max(0.0, round(phi, 4)))

    # -- Content Score (F1) ------------------------------------------

    def _content_score(self, df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        """
        F1-based row matching via inner merge on common columns.
        Uses EXACT values — no normalization. Formatting differences
        (whitespace, casing) correctly reduce this score.
        Numeric coercion only (int/float comparison).
        """
        common_cols = list(set(df.columns) & set(expected_df.columns))
        if not common_cols:
            return 0.0

        try:
            df_cmp = df[common_cols].copy()
            exp_cmp = expected_df[common_cols].copy()

            # Only coerce numeric types — leave strings exactly as-is
            for col in common_cols:
                if exp_cmp[col].dtype in ("int64", "float64"):
                    df_cmp[col] = (
                        pd.to_numeric(df_cmp[col], errors="coerce")
                        .fillna(-999)
                        .astype(int)
                    )
                    exp_cmp[col] = exp_cmp[col].astype(int)
                # Strings: no strip(), no lower() — exact match required

            df_dedup = df_cmp.drop_duplicates()
            exp_dedup = exp_cmp.drop_duplicates()

            merged = df_dedup.merge(exp_dedup, how="inner")
            n_matched = len(merged)
            n_expected = len(exp_dedup)
            n_agent = len(df_dedup)

            if n_expected == 0:
                return 1.0 if n_agent == 0 else 0.0
            if n_agent == 0:
                return 0.0

            recall = n_matched / n_expected
            precision = n_matched / n_agent
            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            return min(1.0, f1)
        except Exception:
            return 0.0

    # -- Schema Score ------------------------------------------------

    def _schema_score(
        self,
        df: pd.DataFrame,
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        """Column name correctness + optional order bonus."""
        expected_cols = set(expected_df.columns)
        agent_cols = set(df.columns)

        if not expected_cols and not agent_cols:
            return 1.0
        if not expected_cols or not agent_cols:
            return 0.0

        intersection = expected_cols & agent_cols
        union = expected_cols | agent_cols
        jaccard = len(intersection) / len(union)

        order_bonus = 0.0
        if constraints.get("expected_col_order", False):
            if list(df.columns) == list(expected_df.columns):
                order_bonus = 0.2

        return min(1.0, 0.8 * jaccard + order_bonus)

    # -- Validity Score ----------------------------------------------

    def _validity_score(
        self,
        df: pd.DataFrame,
        expected_df: pd.DataFrame,
        constraints: Dict[str, Any],
    ) -> float:
        """
        Data quality: nulls in required cols, type correctness, string formatting.
        Returns 0.0 (not 1.0) when no checks apply — no free points.
        """
        if len(df) == 0:
            return 0.0

        checks = []

        # 1. No nulls in required columns
        for col in constraints.get("no_null_cols", []):
            if col in df.columns:
                null_ratio = df[col].isnull().sum() / len(df)
                checks.append(1.0 - null_ratio)
            else:
                checks.append(0.0)

        # 2. Type correctness for numeric columns
        for col in expected_df.columns:
            if col in df.columns and expected_df[col].dtype in ("int64", "float64"):
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    valid_ratio = numeric.notna().sum() / len(df)
                    checks.append(valid_ratio)
                except Exception:
                    checks.append(0.0)

        # 3. String formatting — values must be stripped and lowercased
        for col in expected_df.columns:
            if col in df.columns and expected_df[col].dtype == "object":
                try:
                    vals = df[col].astype(str)
                    clean = vals.str.strip().str.lower()
                    clean_ratio = (vals == clean).sum() / len(df)
                    checks.append(clean_ratio)
                except Exception:
                    checks.append(0.0)

        # No checks applicable -> 0.0, not a free 1.0
        return sum(checks) / len(checks) if checks else 0.0

    # -- Constraint Score --------------------------------------------

    def _constraint_score(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> float:
        """
        Rule satisfaction: uniqueness, numeric ranges, required columns.
        Returns 0.0 (not 1.0) when no checks apply — no free points.
        """
        if len(df) == 0:
            return 0.0

        checks = []

        # 1. Unique columns
        for col in constraints.get("unique_cols", []):
            if col in df.columns:
                n_total = len(df)
                uniqueness = df[col].nunique() / n_total if n_total > 0 else 0
                checks.append(uniqueness)
            else:
                checks.append(0.0)

        # 2. Range constraints
        for col, (lo, hi) in constraints.get("range_constraints", {}).items():
            if col in df.columns:
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(numeric) > 0:
                        in_range = ((numeric >= lo) & (numeric <= hi)).sum() / len(numeric)
                        checks.append(in_range)
                    else:
                        checks.append(0.0)
                except Exception:
                    checks.append(0.0)
            else:
                checks.append(0.0)

        # 3. Required columns present
        expected_cols = constraints.get("expected_cols", [])
        if expected_cols:
            present = sum(1 for c in expected_cols if c in df.columns)
            checks.append(present / len(expected_cols))

        # No checks applicable -> 0.0, not a free 1.0
        return sum(checks) / len(checks) if checks else 0.0

    # -- Extra Row Penalty -------------------------------------------

    def _extra_row_penalty(self, df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        """Penalize extra rows beyond expected. Capped at 0.3."""
        if len(expected_df) == 0:
            return 0.1 if len(df) > 0 else 0.0

        extra = max(0, len(df) - len(expected_df))
        penalty = 0.1 * (extra / len(expected_df))
        return min(0.3, penalty)
