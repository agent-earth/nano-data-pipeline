from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from nano_data_pipeline.skill_evolution import (
    score_skill_file,
    select_skill_candidate,
    validate_skill_scorecard,
)


class SkillEvolutionTests(unittest.TestCase):
    def test_promotes_best_non_regressing_candidate(self):
        parent = self._scorecard("parent", 0.70, safety=1.0, tool=0.60)
        regressing = self._scorecard(
            "regressing",
            0.90,
            safety=0.9,
            tool=0.90,
        )
        passing = self._scorecard("passing", 0.80, safety=1.0, tool=0.75)

        receipt = select_skill_candidate(
            parent,
            [regressing, passing],
            protected_families=["safety"],
        )

        self.assertTrue(receipt["promoted"])
        self.assertEqual(receipt["selected_skill_id"], "passing")
        self.assertEqual(
            receipt["rejection_reasons"]["regressing"],
            ["protected_family_regression:safety"],
        )

    def test_retains_parent_when_candidate_does_not_improve(self):
        parent = self._scorecard("parent", 0.80, safety=1.0)
        tied = self._scorecard("tied", 0.80, safety=1.0)

        receipt = select_skill_candidate(
            parent,
            [tied],
            protected_families=["safety"],
        )

        self.assertFalse(receipt["promoted"])
        self.assertEqual(receipt["selected_skill_id"], "parent")

    def test_rejects_different_development_cases(self):
        parent = self._scorecard("parent", 0.70, safety=1.0)
        candidate = self._scorecard("candidate", 0.90, safety=1.0)
        candidate["case_ids_sha256"] = "c" * 64

        receipt = select_skill_candidate(
            parent,
            [candidate],
            protected_families=["safety"],
        )

        self.assertFalse(receipt["promoted"])
        self.assertEqual(
            receipt["rejection_reasons"]["candidate"],
            ["case_identity_mismatch"],
        )

    def test_scores_skill_file_with_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            skill = Path(directory) / "SKILL.md"
            skill.write_text("frozen skill\n", encoding="utf-8")

            scorecard = score_skill_file(
                skill,
                skill_id="candidate",
                case_ids_sha256="a" * 64,
                aggregate_score=0.75,
                family_scores={"safety": 1.0},
            )

            validate_skill_scorecard(scorecard)
            self.assertEqual(len(scorecard["skill_sha256"]), 64)

    def _scorecard(
        self,
        skill_id: str,
        aggregate: float,
        **families: float,
    ):
        scorecard = {
            "schema_version": "nano_skill_scorecard_v1",
            "skill_id": skill_id,
            "skill_sha256": (skill_id[0] * 64),
            "case_ids_sha256": "a" * 64,
            "aggregate_score": aggregate,
            "family_scores": {
                ("tool-use" if key == "tool" else key): value
                for key, value in families.items()
            },
        }
        validate_skill_scorecard(copy.deepcopy(scorecard))
        return scorecard


if __name__ == "__main__":
    unittest.main()
