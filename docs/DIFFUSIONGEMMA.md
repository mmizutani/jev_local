# DiffusionGemma backend and visual demo

This backend uses the structured decision bridge merged in [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250). vLLM serves `google/diffusiongemma-26B-A4B-it` on port 8000. Its `structured_server.py` exposes `POST /v1/systemone` on port 8011. `jev_local` forwards Jev requests to that bridge and serves a browser demo and API on port 8080. The bridge owns tokenization, canvas construction, denoise reads, and probabilities.

The model is large and needs a supported GPU and a vLLM build containing that PR. GPU memory, latency, and accuracy depend on your hardware and questions; this repository does not provide measurements for this backend.

## Start the model and bridge

In the vLLM checkout containing PR #57250, start two terminals:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --diffusion-config '{"canvas_length": 64}' \
  --max-logprobs 32 --enable-prefix-caching \
  --served-model-name diffusiongemma \
  --host 127.0.0.1 --port 8000
```

```bash
python examples/features/structured_diffusion/structured_server.py \
  --upstream http://127.0.0.1:8000 \
  --tokenizer google/diffusiongemma-26B-A4B-it \
  --model diffusiongemma --canvas 64 --host 127.0.0.1 --port 8011
```

The `--served-model-name` value must match the bridge's `--model`. The exact vLLM install and model download follow the upstream project. The model is retrieved from Hugging Face on first use; accept its access conditions there if prompted. The model server and bridge remain separate processes so the same bridge can serve other clients. The upstream implementation chooses FlashAttention 4 when available and Triton otherwise; on a Spark where automatic selection fails, set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` before `vllm serve`. The diffusion async scheduler is selected automatically, so no `--scheduler-cls` flag is needed.

## Start Jev Local

From this repository:

```bash
./setup_diffusiongemma.sh
./run_diffusiongemma.sh
```

`setup_diffusiongemma.sh` requires Python 3.12+ and writes the bridge URL and local port to ignored `.local/diffusiongemma.env`. It checks `/health` if the bridge is already running. No Python package installation or model download occurs in this adapter. Use `--bridge-url http://127.0.0.1:8011 --port 8080` to change either endpoint. `run_diffusiongemma.sh --port 8081` overrides the saved local port for one run. The adapter binds to `127.0.0.1`.

Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/) to edit State and the question map and run a **live** read. The page shows P(true) for Noul, every Choice or Score option probability, browser latency, bridge timing, and bridge errors. Its initial text is an editable example; it does not display canned results. The browser calls the adapter on the same origin, so it never receives a bridge key.

The usual client and HTTP API also work:

```bash
python3 systemone_client.py --url http://127.0.0.1:8080 --input examples/systemone.json
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-dev' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

The adapter accepts `jev-latest`, `jev-preview`, or `diffusiongemma` in `model`, then sends `diffusiongemma` to the bridge. Top-level `samples` and `steps` are forwarded; the demo defaults to one draw and one denoise step. The bridge also accepts `instructions`, `auto_max`, `auto_threshold`, `think`, `ask`, `chunk_rows`, `chunk_prompt`, and `sequential`. Question-level `depends_on`, `ask_if`, and `alone` are bridge extensions. For Choice, `criteria` is an option-name to description map; for Score it is an ordered list; for Noul it can describe `true` and `false`.

Unlike the other local adapters, this response includes bridge `diagnostics` alongside `model`, `answers`, and `usage`. Noul is P(true). Score is the expected zero-indexed level, with `legend` and option probabilities. `confidence` for Choice and Score describes distribution concentration, not calibrated correctness. The bridge requires at least two Choice alternatives and supports up to 26; its canvas size also limits how many questions fit in one read. Validation and inference errors from the bridge are returned with their status and message. `GET /health` checks that the bridge responds; it does not prove that the model is ready for inference.

For a private bridge that sets `API_KEY`, export `JEV_DIFFUSION_BRIDGE_KEY` before `run_diffusiongemma.sh`; the browser never sees it. `JEV_API_KEY` changes the local `/v1/systemone` bearer key (default `local-dev`). The visual demo route requires an Origin matching the local page and is available only on loopback; do not expose the adapter through a public port forward.
