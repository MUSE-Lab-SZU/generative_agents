"""Dependency-free regression tests; also collected by pytest."""
import copy
import json
import unittest
from datetime import datetime

from modules.depression.complaint_context import normalize_units, stage_units
from modules.depression.engine import DepressionSimulationEngine
from modules.depression.memory_system import DynamicMemorySystem
from modules.depression.state_machine import ComplaintGraphManager


NOW = datetime(2026, 9, 15)


def stage(identifier="a"):
    return {"id": identifier, "label": "hidden label", "summary": "hidden summary",
            "core_belief": "hidden belief", "speaking_style": {"repair_pattern": "hidden repair"},
            "disclosure_units": [
                {"id": "surface", "content": "surface concern", "disclosure_threshold": 0.25},
                {"id": "belief", "content": "sensitive belief", "disclosure_threshold": 0.75}]}


def engine():
    return DepressionSimulationEngine({"enabled": True, "agent_name": "patient",
        "memory": {"enabled": True}, "emotion": {"llm_enabled": False},
        "complaint_graph": {"initial_stage_id": "a", "planner": {"llm_enabled": False},
                            "stages": [stage(), stage("future")]}}, clock_provider=lambda: NOW)


class ComplaintContextTests(unittest.TestCase):
    def setUp(self):
        self.mem = DynamicMemorySystem({"enabled": True, "runtime_threshold": 0.2},
                                       "patient", lambda: NOW)

    def decision(self, trust=0.75, other="doctor", turn="t1", node=None):
        self.mem.trust[other] = trust
        decision = self.mem.prepare("", other, turn)
        self.mem.prepare_complaint(decision, node or stage())
        return decision

    def test_threshold_equality_and_partner_isolation(self):
        self.assertEqual(len(self.decision(0.749)["allowed_complaint"]), 1)
        self.assertEqual(len(self.decision()["allowed_complaint"]), 2)
        self.assertEqual(len(self.decision(0.3, "neighbor")["allowed_complaint"]), 1)
        self.assertEqual(self.decision(1, "")["allowed_complaint"], [])

    def test_preview_read_only_and_snapshot(self):
        obj = engine()
        before = copy.deepcopy(obj.to_dict())
        first = obj.preview_interaction_prompt("home", "morning", "doctor", turn_id="t1")
        self.assertEqual(obj.to_dict(), before)
        self.assertIn("surface concern", first)
        for secret in ("hidden label", "hidden summary", "hidden belief", "hidden repair", "sensitive belief"):
            self.assertNotIn(secret, first)
        obj.memory_system.trust["doctor"] = 1
        second = obj.preview_interaction_prompt("home", "morning", "doctor", turn_id="t1")
        self.assertEqual(first, second)
        self.assertIn("sensitive belief", obj.preview_interaction_prompt(
            "home", "morning", "doctor", turn_id="t2"))
        self.assertNotIn("surface concern", obj.get_simple_prompt())

    def test_node_switch_and_content_version(self):
        old = self.decision()["allowed_complaint"]
        new = self.decision(node=stage("b"))["allowed_complaint"]
        self.assertTrue(set(u["memory_id"] for u in old).isdisjoint(u["memory_id"] for u in new))
        revised = stage()
        revised["disclosure_units"][0]["content"] = "new feeling"
        self.assertNotEqual(old[0]["memory_id"], stage_units(revised)[0]["memory_id"])
        self.assertEqual(self.mem.records, {})

    def test_combined_budget_whole_units_and_deduplication(self):
        self.mem.config.update(max_context_chars=30, complaint_max_context_chars=20)
        decision = self.mem.prepare("", "doctor", "t")
        decision["allowed_memories"] = [{"memory_id": "m", "content": "surface concern"},
                                         {"memory_id": "n", "content": "other memory"}]
        self.mem.prepare_complaint(decision, stage())
        self.assertEqual([m["memory_id"] for m in decision["allowed_memories"]], ["n"])
        self.assertLessEqual(sum(len(u["content"]) for u in
            decision["allowed_memories"] + decision["allowed_complaint"]), 30)
        self.mem.config["complaint_max_context_chars"] = 2
        self.assertEqual(self.decision()["allowed_complaint"], [])

    def test_commit_disclosure_inheritance_idempotency_and_restore(self):
        decision = self.decision()
        identifier = decision["allowed_complaint"][1]["memory_id"]
        judge = lambda _: json.dumps({"direction": "unchanged", "disclosed_ids": [identifier, "fake"]})
        self.assertTrue(self.mem.commit_turn(decision, "sensitive belief", "hello", completion_func=judge))
        self.assertIn("doctor", self.mem.complaint_sources[identifier]["disclosed_to"])
        self.assertNotIn("fake", self.mem.complaint_sources)
        saved = self.mem.to_dict()
        self.assertFalse(self.mem.commit_turn(decision, "sensitive belief", "hello", completion_func=judge))
        self.assertEqual(saved, self.mem.to_dict())
        utterance = next(iter(self.mem.records.values()))
        self.assertEqual(utterance["disclosure_threshold"], 0.75)
        self.mem.load_state(saved)
        self.mem.add_memory({"memory_id": "derived", "content": "reflection",
                             "source_ids": [identifier], "disclosure_threshold": 0.1})
        self.assertEqual(self.mem.records["derived"]["disclosure_threshold"], 0.75)
        self.assertEqual(len(self.decision(0.3, turn="t2")["allowed_complaint"]), 1)

    def test_actual_words_remain_recallable_after_trust_drop(self):
        decision = self.decision()
        self.mem.commit_turn(decision, "actual words", "hello")
        class Embeddings:
            def get_text_embedding(self, text): return [1, 0]
            def get_query_embedding(self, text): return [1, 0]
        self.mem.embedder = Embeddings()
        self.mem.trust["doctor"] = 0
        self.assertEqual(self.mem.prepare("words", "doctor", "t2")["allowed_memories"][0]["content"], "actual words")
        self.assertEqual(self.mem.prepare("words", "neighbor", "t3")["allowed_memories"], [])

    def test_no_discomfort_merely_because_hidden_units_exist(self):
        self.assertFalse(self.decision(0)["blocked_signal"]["present"])

    def test_legacy_and_explicit_empty_units(self):
        legacy = stage()
        del legacy["disclosure_units"]
        self.assertEqual(self.decision(0.99, node=legacy)["allowed_complaint"], [])
        self.assertEqual(len(self.decision(1, node=legacy)["allowed_complaint"]), 2)
        legacy["disclosure_units"] = []
        self.assertEqual(stage_units(legacy), [])
        self.mem.load_state({"owner_id": "patient", "records": []})
        self.assertEqual(self.mem.complaint_sources, {})

    def test_generated_nodes_cannot_lower_threshold_or_spoof_source(self):
        manager = engine().graph_manager
        raw = stage("generated")
        raw["source"] = "config"
        normalized = manager._normalize_branch_detail({"stage": raw}, {"id": "generated"})
        self.assertEqual(normalized["source"], "llm")
        self.assertTrue(all(u["disclosure_threshold"] == 1 for u in stage_units(normalized)))

    def test_invalid_generated_units_fail_closed(self):
        manager = engine().graph_manager
        raw = stage("generated")
        raw["disclosure_units"][0]["disclosure_threshold"] = -1
        result = manager._normalize_branch_detail({"stage": raw}, {"id": "generated"})
        self.assertEqual(stage_units(result), [])

    def test_retrieval_failure_does_not_disable_independent_stage_context(self):
        self.mem.add_memory({"content": "memory requiring embeddings"})
        decision = self.mem.prepare("hello", "doctor", "t")
        self.assertTrue(decision["retrieval_error"])
        self.mem.prepare_complaint(decision, stage())
        self.assertEqual(len(decision["allowed_complaint"]), 1)
        self.assertEqual(decision["allowed_memories"], [])

    def test_engine_commits_frozen_units_and_next_turn_trust(self):
        obj = engine()
        obj.memory_system.trust["doctor"] = 0.74
        obj.preview_interaction_prompt("home", "morning", "doctor", turn_id="t1")
        judge = lambda _: json.dumps({"direction": "increased_slightly", "evidence": "take your time",
                                      "disclosed_ids": []})
        obj.commit_event("chat", "home", "morning", "chat", "surface concern", "doctor",
                         metadata={"turn_id": "t1"}, counterpart_utterance="take your time",
                         emotion_completion_func=judge)
        self.assertAlmostEqual(obj.memory_system.trust_for("doctor"), 0.78)
        self.assertEqual(len(obj.memory_system.complaint_sources), 1)
        self.assertIn("sensitive belief", obj.preview_interaction_prompt(
            "home", "morning", "doctor", turn_id="t2"))

    def test_future_node_never_enters_prompt(self):
        obj = engine()
        obj.memory_system.trust["doctor"] = 1
        obj.graph_manager.stage_catalog["future"]["disclosure_units"][0]["content"] = "future secret"
        prompt = obj.preview_interaction_prompt("home", "morning", "doctor", turn_id="t")
        self.assertNotIn("future secret", prompt)

    def test_invalid_units_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_units([{"id": "a", "content": "x", "disclosure_threshold": value}])
        with self.assertRaises(ValueError):
            normalize_units([{"id": "a", "content": "x"}] * 2)

    def test_checkpoint_preserves_authored_policy(self):
        obj = engine()
        saved = obj.to_dict()
        changed = copy.deepcopy(obj.raw_config["complaint_graph"])
        changed["stages"][0]["disclosure_units"][0]["disclosure_threshold"] = 0
        restored = ComplaintGraphManager.from_dict(saved["complaint_graph_manager"],
                                                   base_config=changed, now_provider=lambda: NOW)
        self.assertEqual(restored.get_current_stage()["disclosure_units"][0]["disclosure_threshold"], 0.25)

    def test_disabled_memory_keeps_original_stage_prompt(self):
        obj = engine()
        obj.memory_system.enabled = False
        self.assertIn("hidden belief", obj.preview_interaction_prompt("home", "morning", "doctor"))


if __name__ == "__main__":
    unittest.main()
