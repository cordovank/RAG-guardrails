#!/usr/bin/env bash
set -euo pipefail
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -d '{"q":"What are some entertainment options offered?","k":3}' | jq .
curl -X POST http://localhost:8000/answer -H "Content-Type: application/json" -d '{"q":"What are some entertainment options offered?","k":3}' | jq .
