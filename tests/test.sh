#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
chmod 700 /logs/verifier

# pytest runs as root so it can read /tests. Candidate code runs as appuser via preexec_fn.
pip install --break-system-packages pytest pytest-json-ctrf && python -m pytest -p no:cacheprovider --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
rc=$?

# Reap any stray processes left by appuser
pkill -9 -u appuser >/dev/null 2>&1 || true

if [ "$rc" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
