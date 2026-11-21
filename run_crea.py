import argparse
import json

import numpy as np
import torch

from admodels.CREA_ICA import CREAICA
from admodels.CREA_Routing import CREARouting
from admodels.CREA_Explanation import CREAExplanation
from admodels.CREA_Adaptation import selective_adaptation


def make_contexts(records, length):
    if records.ndim != 2 or len(records) <= length:
        raise ValueError("Each split must contain more rows than the context length.")

    history = records.unfold(0, length, 1)[:-1].transpose(1, 2)
    return history, records[length:]


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        splits = [
            torch.tensor(data[name], dtype=torch.float32)
            for name in ("train", "control", "validation", "test")
        ]
        labels = torch.as_tensor(data["validation_labels"].copy())

    if any(x.ndim != 2 or not torch.isfinite(x).all() for x in splits):
        raise ValueError("Each feature matrix must be finite and two-dimensional.")
    if any(x.shape[1] != splits[0].shape[1] for x in splits):
        raise ValueError("All splits must use the same feature columns.")
    if labels.shape != (len(splits[2]),):
        raise ValueError("Provide one validation label per original row.")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Use 0 for normal and 1 for anomalous validation records.")

    train = splits[0]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-8)
    return [(x - mean) / scale for x in splits], labels


def train_models(ica, routing, history, current, epochs):
    parameters = list(ica.parameters()) + list(routing.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    ica.train()
    routing.train()

    for _ in range(epochs):
        for indices in torch.randperm(len(current)).split(32):
            x = current[indices]
            h = history[indices]
            loss = (
                ica.training_loss(x)
                + routing.lo_fi(h, x).mean()
                + routing.hi_fi(h, x).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ica.eval()
    routing.eval()


@torch.no_grad()
def calibrate_routing(ica, routing, history, current):
    risk, _ = ica(current)
    routing.fit_reference(risk, history, current)


@torch.no_grad()
def score_window(ica, routing, history, current):
    risk, _ = ica(current)
    return routing(risk, history, current)


def run(args):
    if not 7 <= args.length <= 256:
        raise ValueError("Context length must be between 7 and 256.")
    if not 0 <= args.budget <= 1 or args.epochs < 1 or args.block_size < 1:
        raise ValueError("Invalid budget, epoch count, or block size.")

    torch.manual_seed(args.seed)
    splits, labels = load_data(args.data)
    train, control, validation, test = [
        make_contexts(records, args.length) for records in splits
    ]
    train_h, train_x = train
    control_h, control_x = control
    val_h, val_x = validation
    test_h, test_x = test

    if len(control_x) < 10:
        raise ValueError("At least 10 eligible control records are required.")
    if len(control_x) % args.block_size or len(val_x) % args.block_size:
        raise ValueError("Eligible control and validation counts must fit whole blocks.")

    ica = CREAICA(input_dim=train_x.shape[1])
    routing = CREARouting(input_dim=train_x.shape[1])
    explanation = CREAExplanation()

    train_models(ica, routing, train_h, train_x, args.epochs)
    ica.fit_reference(control_x)
    calibrate_routing(ica, routing, control_h, control_x)

    control_scores, _, _ = score_window(ica, routing, control_h, control_x)
    val_scores, gates, drift = score_window(ica, routing, val_h, val_x)
    p_value, control_mask, current_mask = explanation(
        control_scores, val_scores, block_size=args.block_size
    )

    queried = torch.empty(0, dtype=torch.long)
    normal = torch.empty(0, dtype=torch.long)
    updated = False
   if current_mask is not None:
        limit = int(args.budget * len(val_x))
        queried = torch.argsort(current_mask, descending=True)[:limit]

        # Map eligible targets back to their original validation rows.
        queried_labels = labels[queried + args.length]
        normal = queried[queried_labels == 0]

        keep_threshold = 0.5
        reference, updated = selective_adaptation(
            ica, control_x, control_mask, val_x[normal],
            keep_threshold=keep_threshold,
        )
        if updated:
            kept = control_mask >= keep_threshold
            reference_h = torch.cat((control_h[kept], val_h[normal]), dim=0)
            calibrate_routing(ica, routing, reference_h, reference)

    test_scores, test_gates, _ = score_window(ica, routing, test_h, test_x)

    return {
        "validation_drift": float(drift),
        "permutation_p": p_value,
        "validation_hi_fi_count": int(gates.sum()),
        "eligible_validation_count": len(val_x),
        "queried_validation_rows": (queried + args.length).tolist(),
        "confirmed_normal_count": len(normal),
        "updated": updated,
        "test_first_row": args.length,
        "test_hi_fi_count": int(test_gates.sum()),
        "test_scores": test_scores.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="crea_results.json")
    args = parser.parse_args()

    result = run(args)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)