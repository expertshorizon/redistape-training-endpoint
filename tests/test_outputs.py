import base64
import json
import math
import os
import sqlite3
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

import pytest


def encode_redis_tape(label: int, features: List[float]) -> bytes:
    blob = b"RT"
    blob += struct.pack("<H", len(features))
    blob += bytes([label])
    blob += struct.pack("<" + "f" * len(features), *features)
    return blob


@pytest.fixture(scope="module")
def server():
    port = 8080
    proc = subprocess.Popen(["python3", "/app/main.py"], cwd="/app")
    time.sleep(1)
    yield f"http://localhost:{port}"
    proc.terminate()
    proc.wait()


def test_encode_decode_roundtrip_logic():
    """Verify blob encoder format structures precisely."""
    blob = encode_redis_tape(1, [0.5, -1.2])
    assert blob[:2] == b"RT"
    assert struct.unpack("<H", blob[2:4])[0] == 2
    assert blob[4] == 1


def populate_db(db_path: str, records: List[Tuple[int, List[float]]]):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, blob BLOB)")
    for i, (label, feats) in enumerate(records):
        blob = encode_redis_tape(label, feats)
        c.execute("INSERT INTO records (id, blob) VALUES (?, ?)", (i, blob))
    conn.commit()
    conn.close()


def test_fit_endpoint_loss_monotonicity(server):
    """Verify training loss decreases strictly over time."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    records = [
        (1, [1.0, 1.0]),
        (1, [2.0, 2.0]),
        (0, [-1.0, -1.0]),
        (0, [-2.0, -2.0])
    ]
    populate_db(db_path, records)

    req = urllib.request.Request(
        f"{server}/fit",
        data=json.dumps({"db_path": db_path, "epochs": 20, "lr": 0.1}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read())
        losses = res["losses"]
        assert len(losses) == 20
        for i in range(1, 20):
            assert losses[i] < losses[i - 1]

    os.remove(db_path)


def test_checkpoint_resume_identical_predictions(server):
    """Confirm checkpoint recovery yields exactly matching predictions."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    records = [
        (1, [0.5, 0.5]),
        (0, [-0.5, -0.5])
    ]
    populate_db(db_path, records)

    urllib.request.urlopen(urllib.request.Request(
        f"{server}/fit",
        data=json.dumps({"db_path": db_path, "epochs": 5, "lr": 0.1}).encode(),
        headers={"Content-Type": "application/json"}
    ))

    blob = encode_redis_tape(1, [1.0, 1.0])
    b64_blob = base64.b64encode(blob).decode()

    req_score = urllib.request.Request(
        f"{server}/score",
        data=json.dumps({"blob": b64_blob}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_score) as resp:
        score_before = json.loads(resp.read())["score"]

    urllib.request.urlopen(urllib.request.Request(
        f"{server}/checkpoint",
        data=json.dumps({"key": "ckpt1"}).encode(),
        headers={"Content-Type": "application/json"}
    ))

    urllib.request.urlopen(urllib.request.Request(
        f"{server}/fit",
        data=json.dumps({"db_path": db_path, "epochs": 50, "lr": 0.1}).encode(),
        headers={"Content-Type": "application/json"}
    ))

    with urllib.request.urlopen(req_score) as resp:
        score_after = json.loads(resp.read())["score"]

    assert score_before != score_after

    urllib.request.urlopen(urllib.request.Request(
        f"{server}/resume",
        data=json.dumps({"key": "ckpt1"}).encode(),
        headers={"Content-Type": "application/json"}
    ))

    with urllib.request.urlopen(req_score) as resp:
        score_resumed = json.loads(resp.read())["score"]

    assert score_before == score_resumed
    os.remove(db_path)


def test_score_endpoint_outputs_correct_probabilities(server):
    """Verify scoring logic outputs correct probability floats."""
    blob = encode_redis_tape(1, [0.0, 0.0])
    b64_blob = base64.b64encode(blob).decode()

    req = urllib.request.Request(
        f"{server}/score",
        data=json.dumps({"blob": b64_blob}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read())
        assert 0.0 <= res["score"] <= 1.0


def test_server_file_separation_requirements():
    """Verify application code spans multiple distinct files."""
    app_dir = Path("/app")
    py_files = list(app_dir.glob("*.py"))
    assert len(py_files) >= 3
