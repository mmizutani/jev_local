#!/usr/bin/env bash
# Serve the local Jev API and visual demo against a structured DiffusionGemma bridge.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
config="$root/.local/diffusiongemma.env"
if [[ ! -f "$config" ]]; then
    echo 'Missing DiffusionGemma configuration. Run ./setup_diffusiongemma.sh first.' >&2
    exit 1
fi
# Only setup_diffusiongemma.sh writes this ignored local file; it holds no credentials.
source "$config"
cd "$root"
exec python3 -m diffusiongemma.server --bridge-url "$JEV_DIFFUSION_BRIDGE_URL" --port "$JEV_DIFFUSION_PORT" "$@"
