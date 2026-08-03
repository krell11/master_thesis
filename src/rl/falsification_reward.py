

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_STATE: dict | None = None


def _project_root() -> Path:
    env = os.environ.get("MASTERS_ROOT")
    if env:
        return Path(env).resolve()
    # src/rl/falsification_reward.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def _ensure_path():
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _http_generate(prompts: list[str], *, base_url: str, model: str, max_tokens: int, temperature: float) -> list[str]:
    import urllib.error
    import urllib.request

    texts = []
    for prompt in prompts:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            texts.append(data["choices"][0]["text"])
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Reward HTTP call failed ({base_url}): {exc}. "
                "Start reward server or set REWARD_BACKEND=hf"
            ) from exc
    return texts


def _init_state() -> dict:
    _ensure_path()
    from transformers import AutoTokenizer

    from src.core.retriever import CorpusRetriever
    from src.utils import load_jsonl

    root = _project_root()
    rag_path = Path(os.environ.get("RAG_PATH", root / "data" / "train_data" / "rag.jsonl"))
    model_path = os.environ.get("REWARD_MODEL", str(root / "models"))
    backend = os.environ.get("REWARD_BACKEND", "http").lower().strip()
    base_url = os.environ.get("REWARD_BASE_URL", "http://127.0.0.1:8001/v1")

    print(f"[falsification_reward] loading BM25 from {rag_path} ...")
    retriever = CorpusRetriever(load_jsonl(str(rag_path)))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    state = {
        "retriever": retriever,
        "tokenizer": tokenizer,
        "backend": backend,
        "base_url": base_url,
        "model": model_path,
        "hf_model": None,
        "audit_dir": os.environ.get("REWARD_AUDIT_DIR"),
    }

    if backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM

        # cpu: share one GPU with verl trainer; cuda/auto: dedicated reward GPU
        device = os.environ.get("REWARD_DEVICE", "cpu").lower().strip()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        print(
            f"[falsification_reward] loading HF reward model {model_path} "
            f"device={device} ..."
        )
        kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
        kwargs["device_map"] = "cpu" if device == "cpu" else "auto"
        state["hf_model"] = AutoModelForCausalLM.from_pretrained(
            model_path, **kwargs
        )
        state["hf_model"].eval()

    print(f"[falsification_reward] backend={backend} chunks={len(retriever.chunks)}")
    return state


def _get_state() -> dict:
    global _STATE
    with _LOCK:
        if _STATE is None:
            _STATE = _init_state()
        return _STATE


def _hf_generate(prompts: list[str], *, max_tokens: int, temperature: float) -> list[str]:
    import torch

    state = _get_state()
    model = state["hf_model"]
    tokenizer = state["tokenizer"]
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    texts = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = 0.9
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen = out[0][inputs["input_ids"].shape[1] :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    return texts


def _make_generators(state: dict):
    backend = state["backend"]

    def falsifier_generate(prompts: list[str]) -> list[str]:
        if backend == "hf":
            return _hf_generate(prompts, max_tokens=512, temperature=0.3)
        return _http_generate(
            prompts,
            base_url=state["base_url"],
            model=state["model"],
            max_tokens=512,
            temperature=0.3,
        )

    def adjudicator_generate(prompts: list[str]) -> list[str]:
        if backend == "hf":
            return _hf_generate(prompts, max_tokens=128, temperature=0.0)
        return _http_generate(
            prompts,
            base_url=state["base_url"],
            model=state["model"],
            max_tokens=128,
            temperature=0.0,
        )

    return falsifier_generate, adjudicator_generate


def _maybe_audit(extra_info: dict | None, scored: dict, solution_str: str) -> None:
    state = _get_state()
    audit_dir = state.get("audit_dir")
    if not audit_dir:
        return
    path = Path(audit_dir)
    path.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.time(),
        "extra_info": extra_info or {},
        "solution_str": solution_str[:4000],
        "score": scored.get("score"),
        "parse_ok": scored.get("parse_ok"),
        "y": scored.get("y"),
        "reward_breakdown": scored.get("reward_breakdown"),
        "claims": [
            {
                "claim_id": c.get("claim_id"),
                "verdict": c.get("verdict"),
                "text": c.get("text"),
                "anti_queries": c.get("anti_queries"),
                "rationale": c.get("rationale"),
            }
            for c in (scored.get("claims") or [])
        ],
    }
    with open(path / "grpo_audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
) -> float:
    """verl entrypoint: return scalar falsification reward."""
    _ensure_path()
    from src.core.falsification_score import score_policy_response

    state = _get_state()
    info = extra_info or {}
    question = info.get("question") or info.get("question_original") or ""
    paper_name = info.get("paper_name") or ""
    paper_id = str(info.get("id") or "")

    falsifier_generate, adjudicator_generate = _make_generators(state)
    scored = score_policy_response(
        question=question,
        paper_name=paper_name,
        paper_id=paper_id,
        solution_str=solution_str,
        retriever=state["retriever"],
        tokenizer=state["tokenizer"],
        falsifier_generate=falsifier_generate,
        adjudicator_generate=adjudicator_generate,
    )
    try:
        _maybe_audit(info, scored, solution_str)
    except Exception:
        pass
    return float(scored["score"])
