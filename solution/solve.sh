#!/bin/bash
# Oracle solution.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$HERE"/*.py /app/
