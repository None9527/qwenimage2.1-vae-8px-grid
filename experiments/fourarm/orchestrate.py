"""Run arms C, A, B, AB one after another. Never two jobs on the GPU.

A one-step smoke of all four arms runs first. It checks that every arm
started from the same official weights, and that the two unconditional
critics match each other, as do the two conditional critics.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vae8px.io import atomic_json
from vae8px.paths import PATHS

ROOT = PATHS.fourarm_run
SCRIPT = Path(__file__).resolve().with_name("train.py")
CHILD = None
STOP = False


def request_stop(sig, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def main():
    global CHILD
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    ROOT.mkdir(parents=True, exist_ok=True)
    for smoke in (True, False):
        for arm in ("C", "A", "B", "AB"):
            folder = ROOT / ("_smoke" if smoke else "arms") / arm
            folder.mkdir(parents=True, exist_ok=True)
            if STOP:
                atomic_json(ROOT / "queue_status.json", dict(state="stopped"))
                return
            status_path = folder / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete":
                continue
            command = [sys.executable, "-u", str(SCRIPT), "--arm", arm, "--steps", "1" if smoke else "5000"]
            if smoke:
                command.append("--smoke")
            with (folder / "train.log").open("a") as log:
                CHILD = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(
                    ROOT / "queue_status.json",
                    dict(state="smoke" if smoke else "running", arm=arm, child_pid=CHILD.pid, target_steps=1 if smoke else 5000),
                )
                code = CHILD.wait()
            if STOP:
                atomic_json(ROOT / "queue_status.json", dict(state="stopped", arm=arm))
                return
            if code:
                atomic_json(ROOT / "queue_status.json", dict(state="failed", arm=arm, smoke=smoke, exit_code=code))
                raise SystemExit(code)
            done = json.loads(status_path.read_text())
            if done["state"] != "complete":
                atomic_json(ROOT / "queue_status.json", dict(state="stopped", arm=arm, child_status=done))
                raise SystemExit(1)
        if smoke:
            configs = {
                arm: json.loads((ROOT / "_smoke" / arm / "config.json").read_text()) for arm in ("C", "A", "B", "AB")
            }
            hashes = {item["official_trainable_sha256"] for item in configs.values()}
            assert len(hashes) == 1
            assert configs["C"]["critic_initial_sha256"] == configs["A"]["critic_initial_sha256"]
            assert configs["B"]["critic_initial_sha256"] == configs["AB"]["critic_initial_sha256"]
            (ROOT / "smoke_passed.json").write_text(
                json.dumps(
                    dict(arms=list(configs), official_hash=configs["C"]["official_trainable_sha256"], paired_critic_initializations_equal=True),
                    indent=2,
                )
            )
    atomic_json(ROOT / "queue_status.json", dict(state="complete", arms=["C", "A", "B", "AB"], steps_per_arm=5000))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(REPO))
    main()
