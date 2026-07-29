"""Unit tests for variant-aware replay resume.

Covers:
  - TestSuite._runnable_name()
  - Replay._load_completed_test_names()
  - The filtering integration (runnable_name vs completed set)
"""

import json
import os
import tempfile
import unittest


class FakeRunnable:
    """Minimal stand-in for avocado.core.nrunner.runnable.Runnable."""

    def __init__(self, identifier, variant=None):
        self.identifier = identifier
        self.variant = variant


class TestRunnableName(unittest.TestCase):
    """Tests for TestSuite._runnable_name()."""

    def setUp(self):
        from avocado.core.suite import TestSuite

        self.fn = TestSuite._runnable_name

    def test_no_variant(self):
        r = FakeRunnable("examples/tests/passtest.py:PassTest.test", variant=None)
        self.assertEqual(
            self.fn(r),
            "examples/tests/passtest.py:PassTest.test",
        )

    def test_with_variant_id(self):
        r = FakeRunnable(
            "examples/tests/passtest.py:PassTest.test",
            variant={"variant_id": "fast-abc1", "variant": [], "paths": ["/"]},
        )
        self.assertEqual(
            self.fn(r),
            "examples/tests/passtest.py:PassTest.test;fast-abc1",
        )

    def test_variant_id_none(self):
        """variant dict present but variant_id is None → no suffix."""
        r = FakeRunnable(
            "examples/tests/passtest.py:PassTest.test",
            variant={"variant_id": None, "variant": [], "paths": ["/"]},
        )
        self.assertEqual(
            self.fn(r),
            "examples/tests/passtest.py:PassTest.test",
        )


def _write_job_dir(parent, name, job_id, results, source_job_id=None):
    """Helper: create a fake avocado job dir under *parent/name*.

    Writes ``id``, ``results.json``, and optionally ``jobdata/args.json``
    (with ``job.replay.source_job_id``) so the chain-walk tests work.
    """
    job_path = os.path.join(parent, name)
    os.makedirs(job_path, exist_ok=True)
    with open(os.path.join(job_path, "id"), "w", encoding="utf-8") as f:
        f.write(job_id)
    with open(os.path.join(job_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"tests": results}, f)
    if source_job_id is not None:
        jobdata_dir = os.path.join(job_path, "jobdata")
        os.makedirs(jobdata_dir, exist_ok=True)
        with open(os.path.join(jobdata_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump({"job.replay.source_job_id": source_job_id}, f)
    return job_path


class TestLoadCompletedTestNames(unittest.TestCase):
    """Tests for Replay._load_completed_test_names()."""

    def setUp(self):
        from avocado.plugins.replay import Replay

        self.load = Replay._load_completed_test_names

    def test_pass_and_skip_collected(self):
        with tempfile.TemporaryDirectory() as logs_dir:
            src = _write_job_dir(
                logs_dir,
                "job-orig",
                "aaa111",
                [
                    {"name": "mytest.py:T.test;v1", "status": "PASS"},
                    {"name": "mytest.py:T.test;v2", "status": "FAIL"},
                    {"name": "mytest.py:T.test;v3", "status": "SKIP"},
                    {"name": "mytest.py:T.test;v4", "status": "ERROR"},
                ],
            )
            completed = self.load(src)
        self.assertEqual(
            completed,
            {"mytest.py:T.test;v1", "mytest.py:T.test;v3"},
        )

    def test_missing_results_json(self):
        """Missing results.json returns empty set without raising."""
        self.assertEqual(self.load("/nonexistent/path/xyz"), set())

    def test_empty_tests_list(self):
        with tempfile.TemporaryDirectory() as logs_dir:
            src = _write_job_dir(logs_dir, "job-orig", "aaa111", [])
            self.assertEqual(self.load(src), set())

    def test_corrupt_json_returns_empty(self):
        with tempfile.TemporaryDirectory() as logs_dir:
            src = _write_job_dir(logs_dir, "job-orig", "aaa111", [])
            # overwrite results.json with garbage
            with open(os.path.join(src, "results.json"), "w", encoding="utf-8") as f:
                f.write("not valid json {{")
            self.assertEqual(self.load(src), set())

    def test_chain_walk_accumulates_across_replay_dirs(self):
        """PASS/SKIP names are collected from all replay dirs in the chain."""
        with tempfile.TemporaryDirectory() as logs_dir:
            root_id = "aaa111bbb222ccc333"
            src = _write_job_dir(
                logs_dir,
                "job-orig",
                root_id,
                [
                    {"name": "t.py:T.test_a", "status": "PASS"},
                    {"name": "t.py:T.test_b", "status": "FAIL"},
                ],
            )
            # First replay: interrupted after test_c
            _write_job_dir(
                logs_dir,
                "job-replay1",
                "replay1id",
                [
                    {"name": "t.py:T.test_c", "status": "PASS"},
                    {"name": "t.py:T.test_d", "status": "INTERRUPT"},
                ],
                source_job_id=src,
            )
            # Second replay: picks up from test_d
            _write_job_dir(
                logs_dir,
                "job-replay2",
                "replay2id",
                [
                    {"name": "t.py:T.test_d", "status": "PASS"},
                    {"name": "t.py:T.test_e", "status": "INTERRUPT"},
                ],
                source_job_id=src,
            )

            completed = self.load(src)

        self.assertEqual(
            completed,
            {"t.py:T.test_a", "t.py:T.test_c", "t.py:T.test_d"},
        )

    def test_chain_walk_ignores_unrelated_dirs(self):
        """Job dirs with a different source_job_id are not included."""
        with tempfile.TemporaryDirectory() as logs_dir:
            src = _write_job_dir(
                logs_dir,
                "job-orig",
                "aaa111",
                [
                    {"name": "t.py:T.test_a", "status": "PASS"},
                ],
            )
            # Unrelated replay pointing to a different source
            _write_job_dir(
                logs_dir,
                "job-other",
                "otherid",
                [
                    {"name": "t.py:T.test_z", "status": "PASS"},
                ],
                source_job_id="/some/other/path",
            )

            completed = self.load(src)
        self.assertEqual(completed, {"t.py:T.test_a"})


class TestResumeFiltering(unittest.TestCase):
    """Integration: _runnable_name matches what _load_completed_test_names returns."""

    def test_filters_out_completed_variants_only(self):
        from avocado.core.suite import TestSuite

        completed = {"mytest.py:T.test;v1", "mytest.py:T.test;v3"}
        runnables = [
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v1", "variant": [], "paths": ["/"]}
            ),
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v2", "variant": [], "paths": ["/"]}
            ),
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v3", "variant": [], "paths": ["/"]}
            ),
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v4", "variant": [], "paths": ["/"]}
            ),
        ]
        remaining = [
            r for r in runnables if TestSuite._runnable_name(r) not in completed
        ]
        self.assertEqual(len(remaining), 2)
        self.assertEqual(TestSuite._runnable_name(remaining[0]), "mytest.py:T.test;v2")
        self.assertEqual(TestSuite._runnable_name(remaining[1]), "mytest.py:T.test;v4")

    def test_no_completed_keeps_all(self):
        from avocado.core.suite import TestSuite

        runnables = [
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v1", "variant": [], "paths": ["/"]}
            ),
            FakeRunnable(
                "mytest.py:T.test", {"variant_id": "v2", "variant": [], "paths": ["/"]}
            ),
        ]
        # empty completed set → nothing filtered
        remaining = [r for r in runnables if TestSuite._runnable_name(r) not in set()]
        self.assertEqual(len(remaining), 2)


if __name__ == "__main__":
    unittest.main()
