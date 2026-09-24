"""
server/server.py — long-lived HTTP server around ExpressionWrapper.

Loads ONE model (per --model/--language, fixed at startup) and serves many
/synthesize requests without reloading. Built to replace repeated `main.py`
invocations for the dual-model x dual-language x 8-emotion test sweep.

Usage:
    python server/server.py --model finetuned --language urdu
    python server/server.py --model base --language english --port 8001

Endpoints:
    GET  /health      liveness only — no model access
    GET  /status       model/language/device/cache/ready info
    POST /synthesize   {"text": "<happy>...</happy>", "output_filename": "test_01.wav"}
                        -> {"output_path": ..., "emotion": ..., "duration_s": ...}

Concurrency: single-worker, fully serialized. FastAPI runs sync endpoints in a
threadpool (default ~40 threads), which would let /synthesize calls actually
overlap despite "one worker" — a Lock around the synthesize call is what
actually enforces "second request blocks silently until the first finishes."
"""

import argparse
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

# repo root on sys.path — this file lives in server/, wrapper lives in src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.wrapper import ExpressionWrapper

# ---------------------------------------------------------------------------
# CLI args — parsed at import time so --model/--language are available to the
# lifespan startup hook below. (argparse runs once, at process start.)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="XTTS expression server.")
_parser.add_argument("--model", choices=["base", "finetuned"], default="finetuned")
_parser.add_argument("--language", choices=["english", "urdu"], default="urdu")
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=8000)
ARGS = _parser.parse_args()

# ---------------------------------------------------------------------------
# Shared state — one wrapper instance, one lock, one readiness flag.
# ---------------------------------------------------------------------------
class ServerState:
    wrapper: ExpressionWrapper | None = None
    lock: Lock = Lock()
    ready: bool = False


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[SERVER] Loading model={ARGS.model} language={ARGS.language} ...")
    state.wrapper = ExpressionWrapper(model=ARGS.model, language=ARGS.language)

    print("[SERVER] Warmup synthesis (priming first-call overhead)...")
    t0 = time.time()
    state.wrapper.synthesize("<neutral>warmup</neutral>", "_warmup.wav")
    print(f"[SERVER] Warmup done in {time.time()-t0:.1f}s")

    state.ready = True
    print(f"[SERVER] Ready — listening on {ARGS.host}:{ARGS.port}")
    yield
    print("[SERVER] Shutting down.")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str
    output_filename: str = "output.wav"


class SynthesizeResponse(BaseModel):
    output_path: str
    emotion: str
    duration_s: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness only. No model access — always returns 200 once the process is up."""
    return {"status": "ok"}


@app.get("/status")
def status():
    """Richer readiness/info snapshot. ready=False until warmup finishes."""
    if state.wrapper is None:
        return {"ready": False}
    return {
        "ready": state.ready,
        "model": state.wrapper.model_key,
        "language": state.wrapper.language,
        "device": state.wrapper.device,
        "cached_emotions": sorted(state.wrapper._cache.keys()),
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if not state.ready or state.wrapper is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    t0 = time.time()
    with state.lock:  # enforce true serialization — see module docstring
        try:
            output_path = state.wrapper.synthesize(req.text, req.output_filename)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    # emotion isn't returned by synthesize() — re-derive cheaply from the same
    # parser it already used internally, rather than changing synthesize()'s
    # return contract (kept stable per the confirmed design).
    from src.tag_parser import parse_input
    emotion, _ = parse_input(req.text)

    return SynthesizeResponse(
        output_path=str(output_path),
        emotion=emotion,
        duration_s=round(time.time() - t0, 2),
    )


if __name__ == "__main__":
    # single worker, no reload — matches the "no concurrency" design
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, workers=1)
