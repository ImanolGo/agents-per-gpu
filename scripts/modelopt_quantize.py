"""Quantize Qwen3-8B with TensorRT Model Optimizer (PTQ) and export an HF-format checkpoint that
trtllm-serve can load. Run it in the TensorRT-LLM venv (modelopt 0.37.0, transformers 4.57.3).

    python scripts/modelopt_quantize.py OUT_DIR INT4_AWQ_CFG         # served by TensorRT-LLM 1.2.1
    python scripts/modelopt_quantize.py OUT_DIR INT8_SMOOTHQUANT_CFG  # server rejects it

Serve the result with `MODEL_OVERRIDE=OUT_DIR PREFIX_CACHE=1 ENGINE_VENV=... servers/trtllm.sh
qwen3-8b-int4-awq`, and pass --quant-path to `apg sweep` and `apg quality` with a text that contains
"own PTQ", so the report marks the results (see docs/progress.md, 2026-10-22).
"""

import sys
import time

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME, OUT, CFG = "Qwen/Qwen3-8B", sys.argv[1], sys.argv[2]
N_CALIB, MAXLEN = 128, 512
t0 = time.time()
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"train[:{N_CALIB}]")
texts = [x["article"] for x in ds]


def loop(m):
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids.cuda()
            m(ids)


model = mtq.quantize(model, getattr(mtq, CFG), loop)
print("quantized in", round(time.time() - t0), "s", flush=True)
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=OUT)
tok.save_pretrained(OUT)
print("exported to", OUT, "in", round(time.time() - t0), "s", flush=True)
