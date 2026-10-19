"""Start vLLM (OpenAI-compatible) for the LLM with flags from config.services.llm.
Run: scripts/svc.sh start llm   (the server talks to it through server/llm.py)"""
import json
import os
import sys
from pathlib import Path

from common import config


def main():
    s = config.load()["services"]["llm"]
    vllm = str(Path(sys.executable).parent / "vllm")
    args = [vllm, "serve", s["model"],
            "--served-model-name", s["served_name"],
            "--hf-overrides", json.dumps(s["hf_overrides"]),
            "--tensor-parallel-size", str(s["tensor_parallel_size"]),
            "--max-num-seqs", str(s["max_num_seqs"]),
            "--kv-cache-dtype", s["kv_cache_dtype"],
            "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
            "--max-model-len", str(s["max_model_len"]),
            "--host", s["host"], "--port", str(s["port"])]
    os.environ.update({k: str(v) for k, v in s.get("env", {}).items()})
    print(" ".join(args), flush=True)
    os.execv(vllm, args)


if __name__ == "__main__":
    main()
