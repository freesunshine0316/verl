
import os
import json
import argparse
import pyarrow.parquet as pq
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel
import vertexai
import traceback

import contextlib
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='/apdcephfs_gy2/share_302625456/user/lfsong/tencent-gemini-omd01-c2eab17392a5.json'


verify_prompt = '''
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gemini", "rule"], default="rule")
    parser.add_argument("--base_dir", default="/apdcephfs_gy2/share_302625456/user/lfsong/rm_calib")
    parser.add_argument("--in_file", default="policy_samples_math_every1_temp0.0_rm_scores_with_label.json")
    parser.add_argument("--out_file", default="policy_samples_math_every1_temp0.0_rm_scores_with_label.json")
    parser.add_argument("--portion", default="")
    parser.add_argument("--only_stat", action="store_true")
    args = parser.parse_args()
    return args

def verify_math_solution(solution, reference):
    verify_func = math_metric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    with contextlib.suppress(Exception, TimeoutException):
        ret_score, _ = verify_func([reference], [solution])

    return ret_score

# https://github.com/huggingface/Math-Verify/blob/main/src/math_verify/parser.py#L651
def verify_solution(question, solution, reference, verbose=True):

    p2 = verify_prompt.format(question=question, response=solution, reference=reference)

    out = generate(p2)

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4), flush=True)

    return out

def generate(prompt):
    vertexai.init(project="tencent-gemini-omd01", location="us-central1")
    model = GenerativeModel("gemini-2.5-pro")
    generation_config = { "max_output_tokens": 65535, "temperature": 0.8, "top_p": 0.95}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    for i in range(4):
        try:
            responses = model.generate_content(
                [prompt],
                generation_config=generation_config, safety_settings=safety_settings)
            return responses.candidates[0].content.parts[0].text
        except:
            print("!!! Exception occurred:")
            traceback.print_exc()

def update_stat(rm_score, actual_score, confuse_matrix, bins):
    if isinstance(actual_score, str):
        actual_score = 1 if actual_score.lower() == "yes" else 0
    if rm_score == 0.0:
        if actual_score == 0:
            confuse_matrix["tn"] += 1
        else:
            confuse_matrix["fn"] += 1
    elif rm_score == 1.0:
        if actual_score == 1:
            confuse_matrix["tp"] += 1
        else:
            confuse_matrix["fp"] += 1
    else:
        bins[int(rm_score // 0.1)].append(actual_score)

def main():
    args = get_args()
    inpath = os.path.join(args.base_dir, args.in_file)
    with open(inpath, "r") as fin:
        data = json.load(fin)
    if args.portion != "":
        a, b = args.portion.split("of")
        a, b = int(a), int(b)
        full_size = len(data)
        if a == b:
            data = data[full_size * (a-1) // b:]
        else:
            data = data[full_size * (a-1) // b: full_size * a // b]

    bins = [[] for _ in range(10)]
    confuse_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for inst in data:
        question = inst["query"][-1]["content"] if "query" in inst else inst["problem"]
        solution = inst["text"]
        reference = inst["label"] if "label" in inst else inst["solution"]
        if f"{args.method}_label" in inst:
            actual_score = inst[f"{args.method}_label"]
        elif args.only_stat:
            continue
        elif args.method == "gemini":
            actual_score = verify_solution(question, solution, reference)
        elif args.method == "rule":
            actual_score = verify_math_solution(solution, reference)
        if not f"{args.method}_label" in inst and not actual_score:
            inst[f"{args.method}_label"] = actual_score
        update_stat(inst["rm_score"], actual_score, confuse_matrix, bins)

    bins = [[f"{i/10.0}~{(i+1)/10.0}", round(1.0 * sum(item) / len(item), 4), len(item)] \
            for i, item in enumerate(bins)]
    print(f"Bin stat for uncertain cases: {bins}")
    print(f"Confusion matrix for certain cases: {confuse_matrix}")
    if args.only_stat:
        return

    outpath = os.path.join(args.base_dir, args.out_file)
    if args.portion != "":
        outpath = outpath.replace(".json", f"_{args.portion}.json")
    if "_with_label" not in outpath:
        with open(outpath, "w") as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)

main()
