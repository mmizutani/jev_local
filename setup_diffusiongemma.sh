#!/usr/bin/env bash
# Configure this repository's adapter for an existing structured DiffusionGemma bridge.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
bridge_url="http://127.0.0.1:8011"
port="8080"
while (($#)); do
    case "$1" in
        --bridge-url) bridge_url="${2:?Expected URL after --bridge-url}"; shift 2 ;;
        --port) port="${2:?Expected port after --port}"; shift 2 ;;
        -h|--help) echo 'Usage: ./setup_diffusiongemma.sh [--bridge-url http://127.0.0.1:8011] [--port 8080]'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
python3 - "$bridge_url" "$port" <<'PY'
import sys
from urllib.parse import urlparse
if sys.version_info < (3, 12):
    raise SystemExit('Python 3.12 or newer is required')
url = urlparse(sys.argv[1])
if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
    raise SystemExit('Bridge URL must be an http(s) origin, for example http://127.0.0.1:8011')
try:
    bridge_port = url.port
    port = int(sys.argv[2])
except ValueError as exc:
    raise SystemExit(f'Invalid port: {exc}') from exc
if (bridge_port is not None and not 1 <= bridge_port <= 65535) or not 1 <= port <= 65535:
    raise SystemExit('Ports must be between 1 and 65535')
PY
mkdir -p "$root/.local"
config="$root/.local/diffusiongemma.env"
umask 077
printf 'JEV_DIFFUSION_BRIDGE_URL=%q\nJEV_DIFFUSION_PORT=%q\n' "$bridge_url" "$port" > "$config"
echo "Saved adapter settings to $config"
if python3 - "$bridge_url" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip('/') + '/health', timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
then
    echo "Structured bridge is responding at $bridge_url"
else
    echo "Structured bridge is not responding yet at $bridge_url; start vLLM and its bridge before inference."
fi
echo 'Run ./run_diffusiongemma.sh to serve the API and visual demo.'
