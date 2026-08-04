# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright: Red Hat, Inc. 2020
# Authors: Cleber Rosa <crosa@redhat.com>

"""Replay Job Plugin"""

import json
import os
import sys

from avocado.core import exit_codes, job, output
from avocado.core.data_dir import get_job_results_dir
from avocado.core.jobdata import retrieve_job_config
from avocado.core.plugin_interfaces import CLICmd
from avocado.core.settings import settings


class Replay(CLICmd):
    """Implements the avocado 'replay' subcommand."""

    name = "replay"
    description = "Runs a new job using a previous job as its configuration"

    def configure(self, parser):
        parser = super().configure(parser)
        help_msg = (
            "Replays a job, identified by: complete or partial Job "
            'ID, "latest" for the latest job, the job results path.'
        )
        settings.register_option(
            section="job.replay",
            key="source_job_id",
            help_msg=help_msg,
            metavar="SOURCE_JOB_ID",
            default="latest",
            nargs="?",
            positional_arg=True,
            parser=parser,
        )
        help_msg = (
            "Resume the job, skipping tests (and their variants) that "
            "already passed or were skipped in the source job."
        )
        settings.register_option(
            section="job.replay",
            key="resume",
            help_msg=help_msg,
            default=False,
            key_type=bool,
            action="store_true",
            parser=parser,
            long_arg="--resume",
        )

    @staticmethod
    def _exit_fail(message):
        output.LOG_UI.error(message)
        sys.exit(exit_codes.AVOCADO_FAIL)

    @staticmethod
    def _retrieve_source_job_config(source_job_id, results_dir):
        try:
            return retrieve_job_config(results_dir)
        except OSError:
            msg = f"Could not open the {source_job_id} " f"Job configuration"
            Replay._exit_fail(msg)
        except json.decoder.JSONDecodeError:
            msg = f"Could not read a valid configuration " f'of Job "{source_job_id}"'
            Replay._exit_fail(msg)

    @staticmethod
    def _collect_from_results_json(results_path):
        """Return PASS/SKIP test names from a single results.json file.

        :param results_path: absolute path to a results.json file
        :type results_path: str
        :returns: set of completed test name strings
        :rtype: set
        """
        if not os.path.exists(results_path):
            return set()
        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.decoder.JSONDecodeError):
            return set()
        completed = set()
        for t in data.get("tests", []):
            name = t.get("name")
            status = t.get("status")
            if name and isinstance(status, str) and status.upper() in ("PASS", "SKIP"):
                completed.add(name)
        return completed

    @staticmethod
    def _load_completed_test_names(results_dir):
        """Return the set of test names already passed or skipped across the
        full replay chain rooted at *results_dir*.

        When a job is interrupted and replayed multiple times, each replay
        job dir stores only the results of *that* run.  To find the complete
        set of tests that must be skipped on the next resume we must walk the
        chain:

          results_dir  →  its replay job dir  →  that job's replay dir  →  …

        Each link is found via the ``job.replay.source_job_id`` key stored
        in ``jobdata/args.json`` of the *replaying* job.  We follow the chain
        forward (newer → older is already covered by the root dir itself;
        we follow forward through all replay dirs that pointed *back* to
        this root) by scanning all job dirs in the same logs directory.

        Practically, we collect PASS/SKIP names from every job dir whose
        source is the same root job id, plus the root dir itself.

        :param results_dir: path to the *original* (root) source job results
                            directory
        :type results_dir: str
        :returns: cumulative set of completed test name strings
        :rtype: set
        """
        completed = Replay._collect_from_results_json(
            os.path.join(results_dir, "results.json")
        )

        # Read the root job id so we can find all replay dirs that used it
        # as their source.
        root_id_path = os.path.join(results_dir, "id")
        if not os.path.isfile(root_id_path):
            return completed
        try:
            with open(root_id_path, "r", encoding="utf-8") as fh:
                root_id = fh.read().strip()
        except OSError:
            return completed
        if not root_id:
            return completed

        # Walk all sibling job dirs looking for replay dirs whose source_job_id
        # matches the root job id (full or prefix match, as avocado stores it).
        logs_dir = os.path.dirname(results_dir)
        try:
            entries = sorted(os.listdir(logs_dir))
        except OSError:
            return completed

        visited = {os.path.abspath(results_dir)}
        for entry in entries:
            job_path = os.path.join(logs_dir, entry)
            if not entry.startswith("job-") or not os.path.isdir(job_path):
                continue
            abs_path = os.path.abspath(job_path)
            if abs_path in visited:
                continue
            try:
                cfg = retrieve_job_config(job_path)
            except Exception:  # noqa: BLE001
                continue
            if cfg is None:
                continue
            src = cfg.get("job.replay.source_job_id", "")
            # src may be a full path (when wrapper passes absolute path) or
            # a job id string; either way we match against the root dir path
            # and the root job id.
            if (
                src == results_dir
                or src == os.path.abspath(results_dir)
                or (isinstance(src, str) and root_id.startswith(src))
            ):
                visited.add(abs_path)
                completed |= Replay._collect_from_results_json(
                    os.path.join(job_path, "results.json")
                )

        return completed

    def run(self, config):
        namespace = "job.replay.source_job_id"
        source_job_id = config.get(namespace)
        results_dir = get_job_results_dir(source_job_id)
        if not results_dir:
            msg = f"Could not find the results directory " f'for Job "{source_job_id}"'
            self._exit_fail(msg)
        source_job_config = self._retrieve_source_job_config(source_job_id, results_dir)
        if hasattr(source_job_config, namespace):
            del source_job_config[namespace]
        # Flag that this is indeed a replayed job, which is impossible to
        # tell solely based on the job.replay.source_job_id given that it
        # has a default value of 'latest' for convenience reasons
        source_job_config["job.replay.enabled"] = True
        if config.get("job.replay.resume"):
            completed = self._load_completed_test_names(results_dir)
            if completed:
                output.LOG_UI.info(
                    "Resume mode: skipping %d previously completed test(s).",
                    len(completed),
                )
            source_job_config["job.replay.resume.completed_tests"] = list(completed)
        with job.Job.from_config(source_job_config) as job_instance:
            job_run = job_instance.run()
        return job_run
