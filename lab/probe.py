"""RAM probe: CNN reading game RAM labels from a single Pokemon Red frame."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.data import load
from lab.eval.metrics import majority_class_accuracy, manhattan


class FrameProbe(nn.Module):
    """Small CNN predicting RAM labels from a 72x80 greyscale Game Boy frame."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 10, 256),
            nn.SiLU(),
        )
        self.head_map = nn.Linear(256, 256)
        self.head_x = nn.Linear(256, 64)
        self.head_y = nn.Linear(256, 80)
        self.head_battle = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.fc(self.conv(x))
        return {
            "map": self.head_map(feats),
            "x": self.head_x(feats),
            "y": self.head_y(feats),
            "battle": self.head_battle(feats)[:, 0],
        }


def load_probe(path: str = "work/probe/probe.pt", device: str = "cpu") -> FrameProbe:
    """Load FrameProbe checkpoint in eval mode using weights_only=True."""
    probe = FrameProbe()
    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()
    return probe


def predict(probe: FrameProbe, frames: np.ndarray, device: str = "cpu", batch: int = 1024) -> dict:
    """Predict map_id, x, y, and battle from a numpy array (N, 72, 80) uint8 or float32."""
    probe.eval()
    probe.to(device)
    n = len(frames)
    if n == 0:
        return {
            "map_id": np.empty((0,), dtype=int),
            "x": np.empty((0,), dtype=int),
            "y": np.empty((0,), dtype=int),
            "battle": np.empty((0,), dtype=bool),
        }

    is_uint8 = np.issubdtype(frames.dtype, np.integer)
    map_list = []
    x_list = []
    y_list = []
    battle_list = []

    with torch.no_grad():
        for start_idx in range(0, n, batch):
            end_idx = min(start_idx + batch, n)
            chunk = np.ascontiguousarray(frames[start_idx:end_idx])
            if is_uint8:
                chunk_tensor = torch.from_numpy(chunk).float().div_(255.0)
            else:
                chunk_tensor = torch.from_numpy(chunk).float()

            chunk_tensor = chunk_tensor.unsqueeze(1).to(device)
            out = probe(chunk_tensor)

            map_list.append(out["map"].argmax(dim=-1).cpu().numpy().astype(int))
            x_list.append(out["x"].argmax(dim=-1).cpu().numpy().astype(int))
            y_list.append(out["y"].argmax(dim=-1).cpu().numpy().astype(int))
            battle_list.append((out["battle"] > 0).cpu().numpy().astype(bool))

    return {
        "map_id": np.concatenate(map_list),
        "x": np.concatenate(x_list),
        "y": np.concatenate(y_list),
        "battle": np.concatenate(battle_list),
    }


def evaluate_split(probe: FrameProbe, split_frames: np.ndarray, split_map: np.ndarray,
                   split_x: np.ndarray, split_y: np.ndarray, split_bat: np.ndarray,
                   train_map_non_battle: np.ndarray, device: str = "cpu") -> dict:
    """Evaluate FrameProbe on a split reporting map, x, y, manhattan, and battle accuracy."""
    preds = predict(probe, split_frames, device=device, batch=1024)
    true_bat = split_bat.astype(bool)
    battle_acc = float(np.mean(preds["battle"] == true_bat))

    non_battle = ~true_bat
    if np.any(non_battle):
        map_acc = float(np.mean(preds["map_id"][non_battle] == split_map[non_battle]))
        x_acc = float(np.mean(preds["x"][non_battle] == split_x[non_battle]))
        y_acc = float(np.mean(preds["y"][non_battle] == split_y[non_battle]))
        pred_xy = np.stack([preds["x"][non_battle], preds["y"][non_battle]], axis=-1)
        true_xy = np.stack([split_x[non_battle], split_y[non_battle]], axis=-1)
        mean_manhattan = float(np.mean(manhattan(pred_xy, true_xy)))
        majority_map_acc = float(majority_class_accuracy(train_map_non_battle, split_map[non_battle]))
    else:
        map_acc = float("nan")
        x_acc = float("nan")
        y_acc = float("nan")
        mean_manhattan = float("nan")
        majority_map_acc = float("nan")

    return {
        "map_accuracy": map_acc,
        "x_accuracy": x_acc,
        "y_accuracy": y_acc,
        "mean_manhattan": mean_manhattan,
        "battle_accuracy": battle_acc,
        "majority_class_map_accuracy": majority_map_acc,
    }


def train_probe():
    """Train FrameProbe on CPU with 8 threads and evaluate on test splits."""
    torch.set_num_threads(8)
    work_dir = Path("work/probe")
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "train.log"

    print("Loading dataset...")
    data, splits = load()

    train_mask = np.isin(data["episode_id"], splits["train"])
    train_indices = np.where(train_mask)[0]
    print(f"Train frames: {len(train_indices)}")

    frames = data["frames"]
    map_ids = data["map_ids"]
    xs = data["xs"]
    ys = data["ys"]
    in_battles = data["in_battles"]

    # Pre-extract non-battle train map labels for majority baseline
    train_non_battle_mask = train_mask & (~in_battles)
    train_map_nb = map_ids[train_non_battle_mask]

    probe = FrameProbe()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    batch_size = 256
    rng = np.random.default_rng(0)

    # Speed measurement on CPU
    print("Measuring training speed with 50 warmup steps...")
    t_start_measure = time.time()
    for _ in range(50):
        idx = rng.choice(train_indices, size=batch_size, replace=False)
        bx = torch.from_numpy(frames[idx]).float().div_(255.0).unsqueeze(1)
        b_map = torch.from_numpy(map_ids[idx].astype(np.int64))
        b_x = torch.from_numpy(xs[idx].astype(np.int64))
        b_y = torch.from_numpy(ys[idx].astype(np.int64))
        b_bat = torch.from_numpy(in_battles[idx].astype(np.float32))

        optimizer.zero_grad()
        out = probe(bx)
        loss_bat = F.binary_cross_entropy_with_logits(out["battle"], b_bat)
        nb_mask = (b_bat == 0)
        loss = loss_bat
        if nb_mask.any():
            loss = (
                loss
                + F.cross_entropy(out["map"][nb_mask], b_map[nb_mask])
                + F.cross_entropy(out["x"][nb_mask], b_x[nb_mask])
                + F.cross_entropy(out["y"][nb_mask], b_y[nb_mask])
            )
        loss.backward()
        optimizer.step()

    t_measure = time.time() - t_start_measure
    speed = 50.0 / t_measure
    print(f"Measured training speed: {speed:.2f} steps/s ({t_measure:.2f}s for 50 steps)")

    # Size steps to keep training comfortably <= 20 minutes (targeting ~8-10 minutes)
    target_time_sec = 480.0
    total_steps = int(target_time_sec * speed)
    total_steps = int(round(total_steps / 100.0) * 100)
    # Bound between 3000 and 8000 steps to ensure both good convergence and <= 20 min
    total_steps = max(3000, min(8000, total_steps))
    est_time = total_steps / speed
    print(f"Selected total steps: {total_steps} (estimated wall time: {est_time:.1f}s / {est_time/60:.1f}m)")

    log_lines = [
        f"Training FrameProbe on CPU (8 threads)",
        f"Batch size: {batch_size}, LR: 1e-3, Total steps: {total_steps}",
        f"Measured speed: {speed:.2f} steps/s",
        "---",
    ]
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    t_train_start = time.time()
    t_chunk_start = time.time()

    for step in range(1, total_steps + 1):
        idx = rng.choice(train_indices, size=batch_size, replace=False)
        bx = torch.from_numpy(frames[idx]).float().div_(255.0).unsqueeze(1)
        b_map = torch.from_numpy(map_ids[idx].astype(np.int64))
        b_x = torch.from_numpy(xs[idx].astype(np.int64))
        b_y = torch.from_numpy(ys[idx].astype(np.int64))
        b_bat = torch.from_numpy(in_battles[idx].astype(np.float32))

        optimizer.zero_grad()
        out = probe(bx)
        loss_bat = F.binary_cross_entropy_with_logits(out["battle"], b_bat)
        nb_mask = (b_bat == 0)
        loss = loss_bat
        loss_map_val = 0.0
        loss_x_val = 0.0
        loss_y_val = 0.0
        if nb_mask.any():
            l_map = F.cross_entropy(out["map"][nb_mask], b_map[nb_mask])
            l_x = F.cross_entropy(out["x"][nb_mask], b_x[nb_mask])
            l_y = F.cross_entropy(out["y"][nb_mask], b_y[nb_mask])
            loss = loss + l_map + l_x + l_y
            loss_map_val = l_map.item()
            loss_x_val = l_x.item()
            loss_y_val = l_y.item()

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            elapsed_chunk = time.time() - t_chunk_start
            cur_speed = 100.0 / elapsed_chunk if step % 100 == 0 else (step % 100) / elapsed_chunk
            t_chunk_start = time.time()
            log_str = (
                f"Step {step:5d}/{total_steps} | "
                f"Loss: {loss.item():.4f} (bce: {loss_bat.item():.4f}, map: {loss_map_val:.4f}, "
                f"x: {loss_x_val:.4f}, y: {loss_y_val:.4f}) | {cur_speed:.2f} steps/s"
            )
            print(log_str, flush=True)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
                f.flush()

    total_train_time = time.time() - t_train_start
    final_speed = total_steps / total_train_time
    print(f"Training completed in {total_train_time:.2f}s ({final_speed:.2f} steps/s).", flush=True)

    # Save model checkpoint
    model_path = work_dir / "probe.pt"
    torch.save(probe.state_dict(), model_path)
    print(f"Saved weights to {model_path}", flush=True)

    # Verify load_probe works
    loaded_probe = load_probe(str(model_path), device="cpu")
    print("Successfully verified load_probe checkpoint loading.", flush=True)

    # Evaluation
    print("Evaluating probe...", flush=True)
    # 5000-frame random sample of train
    sample_rng = np.random.default_rng(42)
    sample_5k_idx = sample_rng.choice(train_indices, size=5000, replace=False)

    test_id_mask = np.isin(data["episode_id"], splits["test_id"])
    test_id_indices = np.where(test_id_mask)[0]

    test_ood_mask = np.isin(data["episode_id"], splits["test_ood"])
    test_ood_indices = np.where(test_ood_mask)[0]

    eval_splits = {
        "train_5k": sample_5k_idx,
        "test_id": test_id_indices,
        "test_ood": test_ood_indices,
    }

    metrics = {
        "cpu_training_time_seconds": round(total_train_time, 2),
        "total_training_steps": total_steps,
        "training_steps_per_second": round(final_speed, 2),
    }

    print("\n--- Final Evaluation Metrics ---", flush=True)
    print(f"{'Split':<10} | {'Map Acc':<8} | {'Maj Map':<8} | {'X Acc':<8} | {'Y Acc':<8} | {'Manhattan':<9} | {'Battle Acc':<10}", flush=True)
    print("-" * 75, flush=True)

    for name, indices in eval_splits.items():
        res = evaluate_split(
            loaded_probe,
            frames[indices],
            map_ids[indices],
            xs[indices],
            ys[indices],
            in_battles[indices],
            train_map_nb,
            device="cpu",
        )
        metrics[name] = res
        print(
            f"{name:<10} | {res['map_accuracy']*100:6.2f}% | {res['majority_class_map_accuracy']*100:6.2f}% | "
            f"{res['x_accuracy']*100:6.2f}% | {res['y_accuracy']*100:6.2f}% | "
            f"{res['mean_manhattan']:9.3f} | {res['battle_accuracy']*100:9.2f}%",
            flush=True,
        )

    metrics_path = work_dir / "probe_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    train_probe()
