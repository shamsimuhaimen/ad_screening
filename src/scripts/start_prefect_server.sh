#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POSTGRES_ENV_FILE="${ROOT_DIR}/docker/.env.postgres"

if [[ ! -f "${POSTGRES_ENV_FILE}" ]]; then
  echo "Missing ${POSTGRES_ENV_FILE}. Copy docker/.env.postgres.example first." >&2
  exit 1
fi

set -a
source "${POSTGRES_ENV_FILE}"
set +a

: "${POSTGRES_DB:?POSTGRES_DB must be set in docker/.env.postgres}"
: "${POSTGRES_USER:?POSTGRES_USER must be set in docker/.env.postgres}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in docker/.env.postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT}/${POSTGRES_DB}"

exec prefect server start "$@"
