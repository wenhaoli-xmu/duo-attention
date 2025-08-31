# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from args import parse_args
from compute_scores import compute_scores
from datasets import load_dataset
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    GreedySearch_InfLLM,
    GreedySearch_Mamba2,
    GreedySearch_RetrAttn,
    GreedySearch_RetrAttn_Legacy,
    GreedySearch_vLLM,
    check_benchmark_availability,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    dump_jsonl_append,
    get_compressed_examples,
    get_ground_truth,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    MambaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.import_utils import _is_package_available

if _is_package_available("vllm"):
    from vllm import LLM, SamplingParams
if _is_package_available("lmcache_vllm"):
    from lmcache_vllm.vllm import LLM as LMCacheLLM
    import lmcache_vllm

import random


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    tok=None,
    use_chat_template=False,
    scdq_mode=False,
    disable_golden_context=False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if scdq_mode:
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=False,
        )
    else:
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=False,
            disable_golden_context=disable_golden_context,
        )
    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context
    if scdq_mode:
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def load_model(
    model_name: str,
    method: str,
    attn_load_dir: str,
    sparsity: float,
    sink: int,
    recent: int
):
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    
    if method == 'dense':
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2")
        
    elif method == 'duo-attn':
        from duo_attn.patch import enable_duo_attention_eval
        from duo_attn.utils import (
            load_attn_pattern,
            sparsify_attention_heads)

        assert attn_load_dir is not None, "--attn_load_dir must be provided"
        print(f"Loading attention pattern from {attn_load_dir} with sparsity {sparsity}")

        full_attention_heads, sink_size, recent_size = load_attn_pattern(attn_load_dir)

        if sink is not None:
            sink_size = sink
        if recent is not None:
            recent_size = recent

        full_attention_heads, sparsity = sparsify_attention_heads(full_attention_heads, None, sparsity)
        
        print(f"True sparsity: {sparsity}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2")

        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size)
        
        llm = model
    elif method == 'streaming':
        from duo_attn.patch.streaming_attn import streaming_attention_forward
        import types
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(streaming_attention_forward, layer.self_attn)
    else:
        raise NotImplementedError
    
    llm = llm.cuda()

    llm = GreedySearch(
        llm,
        tok,
    )

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()

    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

    # check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    scdq_mode = args.same_context_different_query

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    if max_seq_length == -1:
        max_seq_length = 160_000

    # Model
    model, tok = load_model(
        model_name,
        args.method,
        args.attn_load_dir,
        args.sparsity,
        args.sink,
        args.recent)

    disable_golden_context = (
        "_disable_golden_context" if args.disable_golden_context else ""
    )
    verbalize_hyper_param = (
        f"_{'-'.join([f'{k}={v}' for k, v in args.hyper_param.items() if k != 'best_pattern'])}"
        if args.hyper_param
        else ""
    )
    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_{args.method}{disable_golden_context}_{verbalize_hyper_param}",
    )
    result_dir.mkdir(exist_ok=True, parents=True)
    use_scdq = "_scdq" if scdq_mode else "_multi_turn"
    use_llmlingua = "_lingua" if args.use_llmlingua else ""
    real_model_name = f"{real_model_name}_{args.method}{use_scdq}{disable_golden_context}_{verbalize_hyper_param}"  # add all the args to the real_model_name, for easy identification


    results = {}
    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if isinstance(max_new_tokens, dict):
            assert (
                max(max_new_tokens.values()) <= max_seq_length
            ), "max_new_tokens must be less than max_seq_length"
        elif max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        # Data
        output_path = (
            result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
        )
        examples = load_dataset("microsoft/SCBench", data_name, split="test")

        if args.use_llmlingua:
            # do prompt compression here
            compression_ratio = (
                args.hyper_param.get("llmlingua_ratio", 3) if args.hyper_param else 3
            )
            examples = get_compressed_examples(
                examples, data_name, args.data_dir, rate=1 / compression_ratio
            )
        max_turn_size = len(examples[0]["multi_turns"])
        if args.max_turns > 0 and args.max_turns < max_turn_size:
            examples = [
                {**eg, "multi_turns": eg["multi_turns"][: args.max_turns]}
                for eg in examples
            ]
            max_turn_size = args.max_turns

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        local_preds = []
        print(f"==== Evaluation {data_name}====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Num of turns: {max_turn_size}")

        done = set()
        local_done = set()
        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    tmp = json.loads(line)
                    local_done.add(int(tmp["id"]))
                    local_preds.append(tmp)
            compute_scores(
                output_path, data_name, real_model_name, max_seq_length, scdq_mode
            )

        # =================================================================
        # NOTE: add multi-gpu evaluation
        def cdiv(x, y):
            if x // y == 0:
                return x // y
            else:
                return int(x // y) + 1
            
        global_result = []
        num_examples_per_rank = cdiv(len(examples), dist.get_world_size())
        assert num_examples_per_rank > 0

        start_idx = num_examples_per_rank * dist.get_rank()
        end_idx = min(start_idx + num_examples_per_rank, len(examples))
        examples_local = examples.select(range(start_idx, end_idx))
        dist.barrier()

        from filelock import FileLock
        lock_path = f"{output_path}.lock"
        lock = FileLock(lock_path)
        # =================================================================

        for i, eg in tqdm(enumerate(examples_local)):
            if (i + start_idx < args.start_example_id) or (i + start_idx in done):
                continue
            if data_name in [
                "scbench_summary_with_needles",
                "scbench_repoqa_and_kv",
            ]:
                max_input_length = max_seq_length - (
                    sum(list(max_new_tokens.values())) * max_turn_size // 2
                )
            else:
                max_input_length = max_seq_length - max_new_tokens * max_turn_size
            if scdq_mode:
                max_input_length -= 1000

            pred = get_pred(
                model,
                eg,
                data_name,
                max_new_tokens,
                max_input_length=max_input_length,
                tok=tok,
                use_chat_template=args.use_chat_template,
                scdq_mode=scdq_mode,
                disable_golden_context=args.disable_golden_context,
            )
            # a list of ground truth answers for each turn
            gts = get_ground_truth(eg, data_name)
            for turn_idx, (ans, gt, turn) in enumerate(
                zip(pred["answers"], gts, eg["multi_turns"])
            ):
                case = {
                    "id": i + start_idx,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                if data_name == "scbench_repoqa":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    case["func_name"] = turn["name"]
                if data_name == "scbench_repoqa_and_kv":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    if turn["task"] == "scbench_repoqa":
                        case["func_name"] = turn["name"]
                if data_name == "scbench_kv_compressible":
                    case["task"] = eg["task"]
                local_preds.append(case)

            with lock:
                dump_jsonl_append(case, output_path)

            torch.cuda.empty_cache()
            local_done.add(i + start_idx)

        dist.barrier()

        if dist.get_rank() == 0:
            score = compute_scores(
                output_path,
                data_name,
                real_model_name,
                max_seq_length=max_seq_length,
                scdq_mode=scdq_mode,
            )
            results[data_name] = score
        
        dist.barrier()

    if dist.get_rank() == 0:
        print("==== Results ====")
        print(json.dumps(results, indent=2))

    dist.barrier()
    dist.destroy_process_group()
