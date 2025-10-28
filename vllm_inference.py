# Load model directly
import os
import json
import math
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

POLICY_PATH = "/apdcephfs_gy2/share_302625456/model/pretrain/Qwen/Qwen2.5-7B"

POLICY_DATA_PATH = "/apdcephfs/share_300000800/user/yudian/magic_exp/datasets/Multi-subject-RLVR/multi-rlvr_train.json"

RM_PATH = "/apdcephfs_gy2/share_302625456/user/lfsong/models/Qwen2.5-7B-Instruct_sft_mixed40k_en"

RM_PROMPT= '''
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
'''

TOPK = 200

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--task", choices=["policy", "rm"], default="rm")
    parser.add_argument("--query_path", default=POLICY_DATA_PATH)
    parser.add_argument("--out_dir", default="/apdcephfs_gy2/share_302625456/user/lfsong/rm_calib")
    parser.add_argument("--out_file", default="policy_samples")
    parser.add_argument("--sample_file", default="policy_samples_every10_temp0.0.json")
    args = parser.parse_args()
    return args

def last_non_empty_line(text):
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if line.strip():  # Check if the line is not empty after stripping
            return line
    return None  # Return None if no non-empty line is found

def prepare_data_rm(inpath, tokenizer, args):
    with open(inpath, "r") as fin:
        inputs = json.load(fin)
    raw_data, data = [], []
    for i, inst in enumerate(inputs):
        response = last_non_empty_line(inst["text"])
        if not response:
            continue
        raw_data.append(inst)

        if "query" in inst:
            question = inst["query"][-1]["content"]
        elif "problem" in inst:
            question = inst["problem"]

        if "label" in inst:
            reference = inst["label"]
        elif "solution" in inst:
            reference = inst["solution"]

        prompt_question = RM_PROMPT.format(question=question, 
                reference=reference,
                response=response)
        messages=[
                  {"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt_question},
                ]
        data.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return raw_data, data

def prepare_data_policy(inpath, tokenizer, args):
    try:
        with open(inpath, "r") as fin:
            inputs = json.load(fin)
    except:
        with open(inpath, "r") as fin:
            inputs = [json.loads(line.strip()) for line in fin]

    data = []
    for i, inst in enumerate(inputs):
        if i % args.every == 0:
            if "query" in inst:
                messages = inst["query"]
            elif "problem" in inst:
                messages = inst["problem"]
            if isinstance(messages, str):
                messages = [{
                    "content": "You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Before giving the final result, you should output \"Therefore, the answer is\", and then give your final answer.",
                    "role": "system"},{
                    "content": messages,
                    "role": "user"
                }]
            data.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return inputs, data

def to_serializable(obj):
    # Base JSON types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # dict / list / tuple
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]

    # vLLM's Logprob-like objects
    cls = obj.__class__.__name__
    if cls == "Logprob":
        # vLLM typically stores the numeric value in .value (or sometimes .logprob)
        val = getattr(obj, "logprob", None)
        tok = getattr(obj, "decoded_token", None)
        return (float(val), tok)

    # Numpy scalars (if any)
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    # Pydantic-style models (some vLLM objects support these)
    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())

    # Fallback: string
    return str(obj)

def get_rm_score(logprobs):
    # print(type(logprobs), logprobs)
    token_probs = {token.decoded_token: math.exp(token.logprob) for token in logprobs.values()}

    # Combine probabilities of YES/NO (case-insensitive)
    yes_prob = sum(prob for token, prob in token_probs.items() if token.lower().strip() == "yes")
    no_prob = sum(prob for token, prob in token_probs.items() if token.lower().strip() == "no")
    total = yes_prob + no_prob
    if total == 0:
        print(f"!!!!! Alert, no yes or no found: {token_probs}")
        return 0.0  # Return baseline value when no valid judgment

    base = 1.0 if yes_prob > no_prob else 0.0
    soft = yes_prob / total
    return soft

args = get_args()
if args.task == "policy":
    model_path = POLICY_PATH
    tokenizer = AutoTokenizer.from_pretrained(POLICY_PATH)
    raw_data, data = prepare_data_policy(args.query_path, tokenizer, args)
else:
    model_path = RM_PATH
    tokenizer = AutoTokenizer.from_pretrained(RM_PATH)
    data_path = os.path.join(args.out_dir, args.sample_file)
    raw_data, data = prepare_data_rm(data_path, tokenizer, args)
print(f"!!!!! read {len(data)} data, example: \n{data[0]}")

sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=8096, 
                                 logprobs=0 if args.task == "policy" else TOPK)
llm = LLM(
    model=model_path,
    tensor_parallel_size=args.tp,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,  # safe for well-known repos; needed by some models
    max_logprobs=TOPK,
)
output = llm.generate(data, sampling_params=sampling_params, use_tqdm=True)

if args.task == "policy":
    out_path = os.path.join(args.out_dir, f"{args.out_file}_every{args.every}_temp{args.temperature}.json")
    output_dump = [{"text": inst.outputs[0].text} for inst in output]
    for i, inst in enumerate(raw_data):
        if i % args.every == 0:
            j = i // args.every
            output_dump[j].update(inst)
else:
    filename = args.sample_file.replace(".json", "_rm_scores.json")
    out_path = os.path.join(args.out_dir, filename)
    output_dump = []
    for i, inst in enumerate(output):
        output_dump.append({"rm_prompt": data[i]})
        output_dump[-1].update(raw_data[i])
        output_dump[-1]["rm_score"] = get_rm_score(inst.outputs[0].logprobs[-1])
with open(out_path, "w") as fout:
    json.dump(output_dump, fout, ensure_ascii=False, indent=2)
