"""Reduced-scale reproduction of the knowing-using gap (arXiv:2607.08393).

The released archive omits its configs and generated STaRK data. This runner
therefore constructs opaque, deterministic two-hop facts while retaining the
paper's Qwen model, LoRA parameters, training objective, exact-match scoring,
and residual-stream self-patching intervention.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CFG = json.loads(Path("config.json").read_text())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_atomic_words(tokenizer, count: int, seed: int) -> list[str]:
    """Select lowercase words represented by one token after a space."""
    candidates: list[str] = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        word = text.strip()
        if (
            text.startswith(" ")
            and word.isalpha()
            and word.islower()
            and 4 <= len(word) <= 11
            and tokenizer.encode(" " + word, add_special_tokens=False) == [token_id]
        ):
            candidates.append(word)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"Need {count} atomic words; found {len(candidates)}")
    return candidates[:count]


def build_records(tokenizer) -> list[dict]:
    n = int(CFG["chains"])
    words = choose_atomic_words(tokenizer, n * 2, int(CFG["seed"]))
    records = []
    for index in range(n):
        code = f"zqv{index:04d}x"
        bridge, answer = words[index], words[n + index]
        records.append(
            {
                "code": code,
                "bridge": bridge,
                "answer": answer,
                "mem1": f"Which relay is assigned to code {code}? Answer:",
                "mem2": f"Which output belongs to relay {bridge}? Answer:",
                "gen": f"Which output belongs to the relay assigned to code {code}? Answer:",
            }
        )
    return records


def answer_token(tokenizer, answer: str) -> int:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError((answer, ids))
    return ids[0]


class MemorizationDataset(Dataset):
    def __init__(self, tokenizer, records: list[dict]):
        self.items = []
        for row in records:
            for prompt, answer in ((row["mem1"], row["bridge"]), (row["mem2"], row["answer"])):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                target = [answer_token(tokenizer, answer)]
                self.items.append((prompt_ids + target, [-100] * len(prompt_ids) + target))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad_id: int):
    width = max(len(item[0]) for item in batch)
    ids, labels, masks = [], [], []
    for tokens, targets in batch:
        padding = width - len(tokens)
        ids.append(tokens + [pad_id] * padding)
        labels.append(targets + [-100] * padding)
        masks.append([1] * len(tokens) + [0] * padding)
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(masks)


@torch.inference_mode()
def predict_next(model, tokenizer, prompts: list[str], device, batch_size: int = 64) -> list[int]:
    predictions: list[int] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to(device)
        logits = model(**encoded).logits
        last = encoded.attention_mask.sum(dim=1) - 1
        predictions.extend(logits[torch.arange(len(last), device=device), last].argmax(-1).cpu().tolist())
    return predictions


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], device) -> dict:
    mem_prompts, mem_targets = [], []
    for row in records:
        mem_prompts.extend([row["mem1"], row["mem2"]])
        mem_targets.extend([answer_token(tokenizer, row["bridge"]), answer_token(tokenizer, row["answer"])])
    gen_prompts = [row["gen"] for row in records]
    gen_targets = [answer_token(tokenizer, row["answer"]) for row in records]
    mem_predictions = predict_next(model, tokenizer, mem_prompts, device)
    gen_predictions = predict_next(model, tokenizer, gen_prompts, device)
    mem_correct = [int(a == b) for a, b in zip(mem_predictions, mem_targets)]
    gen_correct = [int(a == b) for a, b in zip(gen_predictions, gen_targets)]
    return {
        "memorization": sum(mem_correct) / len(mem_correct),
        "generalization": sum(gen_correct) / len(gen_correct),
        "mem_correct": mem_correct,
        "gen_correct": gen_correct,
        "sample_mem_prediction": tokenizer.decode([mem_predictions[0]]),
        "sample_mem_target": tokenizer.decode([mem_targets[0]]),
        "sample_gen_prediction": tokenizer.decode([gen_predictions[0]]),
        "sample_gen_target": tokenizer.decode([gen_targets[0]]),
    }


def locate_positions(tokenizer, prompts: list[str], needles: list[str]) -> list[int]:
    positions = []
    for prompt, needle in zip(prompts, needles):
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target = tokenizer.encode(needle, add_special_tokens=False)
        matches = [i for i in range(len(tokens) - len(target) + 1) if tokens[i : i + len(target)] == target]
        if not matches:
            raise RuntimeError(f"Could not locate {needle!r} in prompt")
        positions.append(matches[-1] + len(target) - 1)
    return positions


@torch.inference_mode()
def patch_scan(model, tokenizer, records: list[dict], device) -> dict:
    """Copy an entity state from source boundary to target boundary."""
    model.eval()
    prompts = [row["gen"] for row in records]
    targets = torch.tensor([answer_token(tokenizer, row["answer"]) for row in records], device=device)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    positions = torch.tensor(locate_positions(tokenizer, prompts, [row["code"] for row in records]), device=device)
    rows = torch.arange(len(records), device=device)
    layers = model.base_model.model.model.layers
    cached: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def capture(index):
        def hook(_module, args):
            cached[index] = args[0][rows, positions].detach().clone()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(capture(index)))
    clean_logits = model(**encoded).logits
    for handle in handles:
        handle.remove()
    last = encoded.attention_mask.sum(dim=1) - 1
    clean_next = clean_logits[rows, last]
    clean_correct = clean_next.argmax(-1).eq(targets)
    clean_prob = clean_next.softmax(-1)[rows, targets]
    oracle = clean_correct.clone()
    fixed = clean_correct.clone()
    layer_count = len(layers)
    fixed_pairs = [
        (min(round(0.82 * layer_count), layer_count - 1), min(round(0.45 * layer_count), layer_count - 1)),
        (min(round(0.10 * layer_count), layer_count - 1), min(round(0.45 * layer_count), layer_count - 1)),
    ]
    accuracy_matrix, gain_matrix = [], []
    for source_index, source in enumerate(cached):
        assert source is not None
        accuracy_row, gain_row = [], []
        for target_index, layer in enumerate(layers):
            def inject(_module, args, replacement=source):
                hidden = args[0].clone()
                hidden[rows, positions] = replacement
                return (hidden,) + args[1:]

            handle = layer.register_forward_pre_hook(inject)
            logits = model(**encoded).logits[rows, last]
            handle.remove()
            probabilities = logits.softmax(-1)[rows, targets]
            correct = logits.argmax(-1).eq(targets)
            oracle |= correct
            if (source_index, target_index) in fixed_pairs:
                fixed |= correct
            accuracy_row.append(float(correct.float().mean()))
            gain_row.append(float((probabilities - clean_prob).mean()))
        accuracy_matrix.append(accuracy_row)
        gain_matrix.append(gain_row)
    return {
        "n": len(records),
        "layers": layer_count,
        "clean_accuracy": float(clean_correct.float().mean()),
        "oracle_accuracy": float(oracle.float().mean()),
        "fixed_accuracy": float(fixed.float().mean()),
        "fixed_pairs": fixed_pairs,
        "accuracy_matrix": accuracy_matrix,
        "probability_gain_matrix": gain_matrix,
    }


def saturation_epoch(curve: list[dict], key: str, threshold: float = 0.95) -> int | None:
    for point in curve:
        if point[key] >= threshold:
            return int(point["epoch"])
    return None


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("Formal reproduction requires CUDA")
    device = torch.device("cuda")
    seed_everything(int(CFG["seed"]))
    properties = torch.cuda.get_device_properties(device)
    print(f"DEVICE={properties.name}; VRAM_GIB={properties.total_memory / 2**30:.1f}", flush=True)
    print("CONFIG=" + json.dumps(CFG, sort_keys=True), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records = build_records(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        CFG["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    lora = LoraConfig(
        r=int(CFG["lora_rank"]),
        lora_alpha=int(CFG["lora_alpha"]),
        lora_dropout=float(CFG["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"PARAMETERS=trainable:{trainable};total:{total};fraction:{trainable / total:.6f}", flush=True)
    curve = []

    def run_eval(epoch: int) -> None:
        metrics = evaluate(model, tokenizer, records, device)
        point = {
            "epoch": epoch,
            "memorization": metrics["memorization"],
            "generalization": metrics["generalization"],
            "sample_mem_prediction": metrics["sample_mem_prediction"],
            "sample_mem_target": metrics["sample_mem_target"],
            "sample_gen_prediction": metrics["sample_gen_prediction"],
            "sample_gen_target": metrics["sample_gen_target"],
        }
        curve.append(point)
        print("EPOCH_METRICS=" + json.dumps(point), flush=True)

    run_eval(0)
    if int(CFG["epochs"]) > 0:
        dataset = MemorizationDataset(tokenizer, records)
        generator = torch.Generator().manual_seed(int(CFG["seed"]))
        loader = DataLoader(
            dataset,
            batch_size=int(CFG["batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(CFG["learning_rate"]),
            weight_decay=float(CFG["weight_decay"]),
        )
        accumulation = int(CFG["gradient_accumulation"])
        eval_epochs = set(int(x) for x in CFG["evaluation_epochs"])
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, int(CFG["epochs"]) + 1):
            model.train()
            epoch_loss = 0.0
            for step, (ids, labels, masks) in enumerate(loader, start=1):
                loss = model(input_ids=ids.to(device), labels=labels.to(device), attention_mask=masks.to(device)).loss / accumulation
                epoch_loss += float(loss.detach()) * accumulation
                loss.backward()
                if step % accumulation == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            print(f"EPOCH_TRAIN_LOSS={epoch}:{epoch_loss / len(loader):.6f}", flush=True)
            if epoch in eval_epochs:
                run_eval(epoch)

    final = evaluate(model, tokenizer, records, device)
    patch = None
    patch_count = int(CFG["patch_examples"])
    if patch_count:
        failures = [row for row, correct in zip(records, final["gen_correct"]) if not correct]
        patch_records = (failures + records)[:patch_count]
        patch = patch_scan(model, tokenizer, patch_records, device)
    elapsed = time.time() - started
    result = {
        "paper": "arXiv:2607.08393",
        "scope": "reduced-scale opaque synthetic chaining reconstruction",
        "config": CFG,
        "device": properties.name,
        "runtime_seconds": elapsed,
        "curve": curve,
        "final": {"memorization": final["memorization"], "generalization": final["generalization"], "gap": final["memorization"] - final["generalization"]},
        "saturation_epoch_95": {"memorization": saturation_epoch(curve, "memorization"), "generalization": saturation_epoch(curve, "generalization")},
        "patch": patch,
    }
    print("ORX_RESULT_JSON=" + json.dumps(result, separators=(",", ":")), flush=True)
    print(f"RUNTIME_SECONDS={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
