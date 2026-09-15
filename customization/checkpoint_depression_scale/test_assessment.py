import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from customization.checkpoint_depression_scale import assessment as scale


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "sim-test"
        self.archive.mkdir()
        self.names = [f"simulate-20260805-{time}.json" for time in (1530, 1540, 1550)]
        for name in self.names:
            (self.archive / name).write_text(json.dumps({
                "time": "20260805-15:30", "agents": {"测试角色": {}}
            }), encoding="utf-8")

    def test_strict_score_parsing(self):
        for reply, expected in [("0", 0), ("评分：4。理由：过去一周一直如此。", 4),
                                ("评分: 2", 2), ("评分：0。", 0)]:
            self.assertEqual(scale.parse_score(reply), expected)
        for reply in ["5", "-1", "2.5", "过去一周有2天不开心", "评分：2或3", "评分：20",
                      "评分：2。理由：评分：3。", "嗯", "评分：true", "评分：2.5"]:
            self.assertIsNone(scale.parse_score(reply), reply)

    def test_batch_retries_failures_and_persistence(self):
        originals = {p.name: p.read_bytes() for p in self.archive.iterdir()}
        calls = []

        @contextmanager
        def factory(archive, checkpoint, agent, **kwargs):
            calls.append(checkpoint)
            if checkpoint == self.names[0]:
                replies = iter(["嗯", "评分：0。理由：没有。", "1", "2", "3", "4"])
                yield SimpleNamespace(chat=lambda prompt: next(replies))
            elif checkpoint == self.names[1]:
                replies = iter(["嗯"] * 10)
                yield SimpleNamespace(chat=lambda prompt: next(replies))
            else:
                raise RuntimeError("模拟加载失败")

        results = list(scale.run_batch("sim-test", "测试角色", list(reversed(self.names)) + [self.names[0]],
                                      root=self.root, session_factory=factory))
        self.assertEqual(calls, self.names)
        records = [record for record, _ in results]
        self.assertEqual([r["status"] for r in records], ["completed", "incomplete", "error"])
        self.assertEqual([r["total_score"] for r in records], [10, None, None])
        self.assertEqual(len(records[0]["items"][0]["attempts"]), 2)
        self.assertIn("20260805-15:30", records[0]["items"][0]["attempts"][0]["prompt"])
        self.assertEqual(records[1]["valid_items"], 0)
        output = results[0][1]
        self.assertEqual(output.parent, self.archive)
        self.assertEqual([json.loads(line) for line in output.read_text().splitlines()], records)
        for name, content in originals.items():
            self.assertEqual((self.archive / name).read_bytes(), content)

    def test_error_preserves_partial_answers_and_continues(self):
        @contextmanager
        def factory(archive, checkpoint, agent, **kwargs):
            count = 0

            def chat(prompt):
                nonlocal count
                count += 1
                if count == 2 and checkpoint == self.names[0]:
                    raise RuntimeError("连接中断")
                return "4"

            yield SimpleNamespace(chat=chat)

        results = list(scale.run_batch("sim-test", "测试角色", self.names[:2], root=self.root, session_factory=factory))
        failed, completed = [record for record, _ in results]
        self.assertEqual(failed["item_scores"], {"1": 4, "2": None})
        self.assertIsNone(failed["total_score"])
        self.assertEqual(completed["total_score"], 20)

    def test_invalid_selection_does_not_create_output(self):
        for archive, agent, checkpoints in [("../outside", "测试角色", self.names),
                                             ("sim-test", "不存在", self.names),
                                             ("sim-test", "测试角色", []),
                                             ("sim-test", "测试角色", ["../escape.json"])]:
            with self.assertRaises(ValueError):
                list(scale.run_batch(archive, agent, checkpoints, root=self.root))
        self.assertEqual(list(self.archive.glob("*.jsonl")), [])

    def test_runtime_success_uses_agent_chat_and_cleans_up(self):
        from customization.checkpoint_depression_scale import runtime

        calls = []
        agent = SimpleNamespace(
            name="测试角色", associate=SimpleNamespace(index=SimpleNamespace(has_node=lambda node: True)),
            reset=lambda: calls.append("reset"), llm_available=lambda: True,
            get_tile=lambda: SimpleNamespace(get_address=lambda: ["地点"]),
            completion=lambda kind, *args: calls.append(kind) or ("1" if kind == "generate_chat" else "关系"),
        )
        storage_paths = []

        def create_game(*args, **kwargs):
            storage_paths.append(Path(kwargs["storage_root"]))
            return SimpleNamespace(get_agent=lambda name: agent)

        with patch.object(runtime, "create_game", side_effect=create_game):
            results = list(scale.run_batch("sim-test", "测试角色", self.names[:2], root=self.root))
        self.assertEqual([record["total_score"] for record, _ in results], [5, 5])
        self.assertEqual(calls.count("reset"), 2)
        self.assertEqual(calls.count("generate_chat"), 10)
        self.assertEqual(len(set(storage_paths)), 2)
        self.assertTrue(all(not path.exists() for path in storage_paths))

    def test_runtime_copies_storage_and_restores_globals_on_failure(self):
        from customization.checkpoint_depression_scale import runtime

        storage = self.archive / "storage" / "测试角色"
        storage.mkdir(parents=True)
        (storage / "marker").write_text("original")
        old_game = object()
        old_timer = object()
        namespace = runtime.utils.GenerativeAgentsMap
        game_key, timer_key = runtime.utils.GenerativeAgentsKey.GAME, runtime.utils.GenerativeAgentsKey.TIMER
        original_map = namespace.MAP.copy()
        self.addCleanup(setattr, namespace, "MAP", original_map)
        namespace.set(game_key, old_game)
        namespace.set(timer_key, old_timer)
        copied_paths = []

        def create_game(name, static_root, config, conversation, **kwargs):
            copied = Path(kwargs["storage_root"]) / "测试角色" / "marker"
            copied_paths.append(copied)
            self.assertEqual(copied.read_text(), "original")
            copied.write_text("modified")
            self.assertEqual(config["time"], {"start": "20260805-15:30"})
            self.assertEqual(conversation, {})
            namespace.set(game_key, "temporary-game")
            namespace.set(timer_key, "temporary-timer")
            raise RuntimeError("模拟构造失败")

        with patch.object(runtime, "create_game", side_effect=create_game):
            with self.assertRaisesRegex(RuntimeError, "模拟构造失败"):
                with runtime.checkpoint_session("sim-test", self.names[0], "测试角色", root=self.root):
                    self.fail("构造失败不应产生会话")
        self.assertEqual((storage / "marker").read_text(), "original")
        self.assertFalse(copied_paths[0].exists())
        self.assertIs(namespace.get(game_key), old_game)
        self.assertIs(namespace.get(timer_key), old_timer)


if __name__ == "__main__":
    unittest.main()
