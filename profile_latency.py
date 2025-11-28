import torch
import time
import os
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

def prefill(path):
    # Benchmark actual model
    llm_engine = LLMEngine(path, max_num_batched_tokens=16384)
    sampling_params = SamplingParams(max_tokens=30)
    for _ in range(1):
        llm_engine.add_request(list(range(1024*16)), sampling_params)
    
    scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    end_time = time.time()

    profiled_latency = (end_time - start_time) * 1000 / 10

    print(f"Profiled latency: {profiled_latency} ms")

def decode(path):
    # Benchmark actual model
    llm_engine = LLMEngine(path, max_num_batched_tokens=16384, max_model_len=16384)
    sampling_params = SamplingParams(max_tokens=30)
    for _ in range(8):
        llm_engine.add_request(list(range(16000)), sampling_params)
        scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)
        llm_engine.scheduler.update(scheduled_reqs, token_ids)

    scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
    print([(r.num_scheduled_tokens, r.num_computed_tokens) for r in scheduled_reqs])

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    end_time = time.time()

    profiled_latency = (end_time - start_time) * 1000 / 10

    print(f"Profiled latency: {profiled_latency} ms")


if __name__ == "__main__":
    # path = os.path.expanduser("/data/lei/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/")
    path = os.path.expanduser("/data/lei/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/")
    # path = os.path.expanduser("/data/lei/huggingface/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29")
    prefill(path)