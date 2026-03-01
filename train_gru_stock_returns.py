#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_COLS = [
    "mktcap_z",
    "log_mktcap_z",
    "book_to_market_z",
    "momentum_z",
    "rev_1m_z",
    "volatility_z",
    "beta_z",
    "roa_z",
    "ni_over_at_z",
    "investment_z",
    "asset_growth_z",
    "leverage_z",
]
REQUIRED_COLS = ["permno", "month", "ret_fwd", "split", *FEATURE_COLS]
VALID_SPLITS = ("train", "val", "test")

WINDOW_COUNTS_OUT = Path(r"d:/ML project/gru_window_counts_by_month.csv")
ALIGNMENT_SAMPLES_OUT = Path(r"d:/ML project/gru_alignment_samples.txt")
BASELINES_SUMMARY_OUT = Path(r"d:/ML project/gru_constant_baselines_summary.csv")
RIDGE_SUMMARY_OUT = Path(r"d:/ML project/gru_laststep_ridge_summary.csv")
MONTHLY_DIAGNOSTICS_OUT = Path(r"d:/ML project/gru_monthly_diagnostics.csv")
DEBUG_PREDICTIONS_OUT = Path(r"d:/ML project/gru_test_predictions_debug.csv")


@dataclass
class SplitArrays:
    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    permno: np.ndarray
    month_ns: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, arrays: SplitArrays) -> None:
        self.x = torch.from_numpy(arrays.x)
        self.y = torch.from_numpy(arrays.y)
        self.y_raw = torch.from_numpy(arrays.y_raw)
        self.permno = torch.from_numpy(arrays.permno)
        self.month_ns = torch.from_numpy(arrays.month_ns)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.y_raw[idx], self.permno[idx], self.month_ns[idx]


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU for monthly cross-sectional stock return prediction.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(r"d:/ML project/model_input_features_12f_2001_2024.parquet"),
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path(r"d:/ML project/rnn_test_predictions.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(r"d:/ML project/rnn_results_summary.csv"),
    )
    parser.add_argument("--seq-len", type=int, default=12, help="History length in months, e.g. 6, 12, 24.")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=("raw", "xs"), default="raw")
    parser.add_argument("--eval-on-raw", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Run diagnostics and save debug outputs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_input(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")

    df = pd.read_parquet(path, columns=REQUIRED_COLS)
    require_columns(df, REQUIRED_COLS)

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        raise ValueError("Column 'month' contains invalid datetimes.")

    df = df.sort_values(["permno", "month"], kind="mergesort").reset_index(drop=True)

    dupes = df.duplicated(["permno", "month"], keep=False)
    if dupes.any():
        count = int(dupes.sum())
        raise ValueError(f"Found {count} duplicate (permno, month) rows. Sequence building requires unique keys.")

    bad_split = ~df["split"].isin(VALID_SPLITS)
    if bad_split.any():
        invalid = sorted(df.loc[bad_split, "split"].astype(str).unique().tolist())
        raise ValueError(f"Invalid split labels found: {invalid}")

    df["permno"] = df["permno"].astype(np.int64)
    df["ret_fwd"] = df["ret_fwd"].astype(np.float32)
    for col in FEATURE_COLS:
        df[col] = df[col].astype(np.float32)

    return df


def add_cross_sectional_target(df: pd.DataFrame) -> pd.DataFrame:
    month_mean = df.groupby("month", sort=False)["ret_fwd"].transform("mean").astype(np.float32)
    df = df.copy()
    df["ret_fwd_xs"] = (df["ret_fwd"] - month_mean).astype(np.float32)
    return df


def empty_split_arrays(seq_len: int, n_features: int) -> SplitArrays:
    return SplitArrays(
        x=np.empty((0, seq_len, n_features), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        y_raw=np.empty((0,), dtype=np.float32),
        permno=np.empty((0,), dtype=np.int64),
        month_ns=np.empty((0,), dtype=np.int64),
    )


def concat_split_chunks(chunks: list[SplitArrays], seq_len: int, n_features: int) -> SplitArrays:
    if not chunks:
        return empty_split_arrays(seq_len, n_features)

    return SplitArrays(
        x=np.concatenate([chunk.x for chunk in chunks], axis=0).astype(np.float32, copy=False),
        y=np.concatenate([chunk.y for chunk in chunks], axis=0).astype(np.float32, copy=False),
        y_raw=np.concatenate([chunk.y_raw for chunk in chunks], axis=0).astype(np.float32, copy=False),
        permno=np.concatenate([chunk.permno for chunk in chunks], axis=0).astype(np.int64, copy=False),
        month_ns=np.concatenate([chunk.month_ns for chunk in chunks], axis=0).astype(np.int64, copy=False),
    )


def build_sequence_splits(
    df: pd.DataFrame,
    seq_len: int,
    target_col: str = "ret_fwd",
) -> dict[str, SplitArrays]:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    n_features = len(FEATURE_COLS)
    split_chunks: dict[str, list[SplitArrays]] = {split: [] for split in VALID_SPLITS}

    grouped = df.groupby("permno", sort=False, observed=True)
    for permno, group in grouped:
        n_rows = len(group)
        if n_rows < seq_len:
            continue

        features = group[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
        targets = group[target_col].to_numpy(dtype=np.float32, copy=False)
        targets_raw = group["ret_fwd"].to_numpy(dtype=np.float32, copy=False)
        splits = group["split"].to_numpy(copy=False)
        month_ns = group["month"].astype("int64").to_numpy(copy=False)
        month_ord = group["month"].dt.to_period("M").astype("int64").to_numpy()

        if seq_len == 1:
            valid_window_mask = np.ones(n_rows, dtype=bool)
        else:
            gap_flags = (month_ord[1:] - month_ord[:-1]) != 1
            prefix = np.concatenate(([0], np.cumsum(gap_flags.astype(np.int32))))
            starts = np.arange(0, n_rows - seq_len + 1, dtype=np.int64)
            ends = starts + seq_len - 1
            invalid_counts = prefix[ends] - prefix[starts]
            valid_window_mask = invalid_counts == 0

        if not valid_window_mask.any():
            continue

        windows = sliding_window_view(features, window_shape=seq_len, axis=0)
        windows = np.swapaxes(windows, 1, 2)

        starts = np.arange(0, n_rows - seq_len + 1, dtype=np.int64)
        ends = starts + seq_len - 1

        windows = windows[valid_window_mask]
        end_idx = ends[valid_window_mask]
        end_splits = splits[end_idx]
        end_y = targets[end_idx]
        end_y_raw = targets_raw[end_idx]
        end_month_ns = month_ns[end_idx]

        for split in VALID_SPLITS:
            split_mask = end_splits == split
            if not split_mask.any():
                continue

            split_chunks[split].append(
                SplitArrays(
                    x=windows[split_mask].astype(np.float32, copy=False),
                    y=end_y[split_mask].astype(np.float32, copy=False),
                    y_raw=end_y_raw[split_mask].astype(np.float32, copy=False),
                    permno=np.full(int(split_mask.sum()), permno, dtype=np.int64),
                    month_ns=end_month_ns[split_mask].astype(np.int64, copy=False),
                )
            )

    split_arrays = {
        split: concat_split_chunks(chunks, seq_len=seq_len, n_features=n_features)
        for split, chunks in split_chunks.items()
    }

    for split, arrays in split_arrays.items():
        if len(arrays.y) == 0:
            raise ValueError(f"No valid windows found for split '{split}'.")

    return split_arrays


def make_loader(arrays: SplitArrays, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(arrays)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch, _, _, _ in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            batch_size = y_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    max_epochs: int,
    patience: int,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x_batch, y_batch, _, _, _ in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = y_batch.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_loss(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch:02d} | train_mse={train_loss:.8f} | val_mse={val_loss:.8f}")

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    model.load_state_dict(best_state)
    return model


def predictions_from_arrays(
    arrays: SplitArrays,
    y_pred: np.ndarray,
    use_raw_for_eval: bool,
) -> pd.DataFrame:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_pred.shape[0] != arrays.y.shape[0]:
        raise ValueError(f"Prediction length mismatch: got {y_pred.shape[0]}, expected {arrays.y.shape[0]}.")

    y_true_eval = arrays.y_raw if use_raw_for_eval else arrays.y
    pred_df = pd.DataFrame(
        {
            "permno": arrays.permno.astype(np.int64, copy=False),
            "end_month": pd.to_datetime(arrays.month_ns),
            "y_true_train_target": arrays.y.astype(np.float32, copy=False),
            "y_true_raw": arrays.y_raw.astype(np.float32, copy=False),
            "y_true": y_true_eval.astype(np.float32, copy=False),
            "y_pred": y_pred.astype(np.float32, copy=False),
        }
    )
    return pred_df.sort_values(["end_month", "permno"], kind="mergesort").reset_index(drop=True)


def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_raw_for_eval: bool) -> pd.DataFrame:
    model.eval()
    y_pred_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []
    y_raw_parts: list[np.ndarray] = []
    permno_parts: list[np.ndarray] = []
    month_parts: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch, y_raw_batch, permno_batch, month_batch in loader:
            preds = model(x_batch.to(device, non_blocking=True)).cpu().numpy().astype(np.float32, copy=False)
            y_pred_parts.append(preds)
            y_true_parts.append(y_batch.numpy().astype(np.float32, copy=False))
            y_raw_parts.append(y_raw_batch.numpy().astype(np.float32, copy=False))
            permno_parts.append(permno_batch.numpy().astype(np.int64, copy=False))
            month_parts.append(month_batch.numpy().astype(np.int64, copy=False))

    arrays = SplitArrays(
        x=np.empty((0, 0, 0), dtype=np.float32),
        y=np.concatenate(y_true_parts).astype(np.float32, copy=False),
        y_raw=np.concatenate(y_raw_parts).astype(np.float32, copy=False),
        permno=np.concatenate(permno_parts).astype(np.int64, copy=False),
        month_ns=np.concatenate(month_parts).astype(np.int64, copy=False),
    )
    y_pred = np.concatenate(y_pred_parts).astype(np.float32, copy=False)
    return predictions_from_arrays(arrays, y_pred, use_raw_for_eval=use_raw_for_eval)


def compute_oos_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    numerator = float(np.square(y_true - y_pred).sum())
    denominator = float(np.square(y_true - train_mean).sum())
    if denominator == 0.0:
        return float("nan")
    return 1.0 - numerator / denominator


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def compute_monthly_long_short(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for end_month, month_df in pred_df.groupby("end_month", sort=True):
        n = len(month_df)
        if n < 2:
            continue

        if month_df["y_pred"].nunique(dropna=False) < 2:
            ls = 0.0
        else:
            month_df = month_df.sort_values(["y_pred", "permno"], kind="mergesort")
            bucket = max(1, int(math.ceil(n * 0.10)))
            bottom = float(month_df["y_true"].iloc[:bucket].mean())
            top = float(month_df["y_true"].iloc[-bucket:].mean())
            ls = top - bottom

        rows.append({"end_month": end_month, "ls_return": ls, "ls_n_obs": n})

    return pd.DataFrame(rows)


def compute_monthly_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for end_month, month_df in pred_df.groupby("end_month", sort=True):
        n = len(month_df)
        if n < 2:
            continue

        if month_df["y_true"].nunique(dropna=False) < 2 or month_df["y_pred"].nunique(dropna=False) < 2:
            ic = 0.0
        else:
            raw_ic = month_df["y_true"].corr(month_df["y_pred"], method="spearman")
            ic = float(raw_ic) if pd.notna(raw_ic) else 0.0

        rows.append({"end_month": end_month, "ic": ic, "ic_n_obs": n})

    return pd.DataFrame(rows)


def summarize_predictions(pred_df: pd.DataFrame, train_mean: float, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = pred_df["y_true"].to_numpy(dtype=np.float32, copy=False)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float32, copy=False)

    oos_r2 = compute_oos_r2(y_true, y_pred, train_mean=train_mean)
    rmse = compute_rmse(y_true, y_pred)

    ic_df = compute_monthly_ic(pred_df)
    ls_df = compute_monthly_long_short(pred_df)

    monthly_df = pd.merge(ic_df, ls_df, on=["end_month"], how="outer").sort_values("end_month", kind="mergesort")
    if "ic_n_obs" not in monthly_df.columns:
        monthly_df["ic_n_obs"] = np.nan
    if "ls_n_obs" not in monthly_df.columns:
        monthly_df["ls_n_obs"] = np.nan
    monthly_df["n_obs"] = monthly_df["ic_n_obs"].combine_first(monthly_df["ls_n_obs"])
    both_mask = monthly_df["ic_n_obs"].notna() & monthly_df["ls_n_obs"].notna()
    monthly_df.loc[both_mask, "n_obs"] = np.maximum(
        monthly_df.loc[both_mask, "ic_n_obs"].to_numpy(dtype=np.float64, copy=False),
        monthly_df.loc[both_mask, "ls_n_obs"].to_numpy(dtype=np.float64, copy=False),
    )

    mean_ic = float(monthly_df["ic"].dropna().mean()) if "ic" in monthly_df.columns else float("nan")
    mean_ls = float(monthly_df["ls_return"].dropna().mean()) if "ls_return" in monthly_df.columns else float("nan")
    ls_non_na = monthly_df["ls_return"].dropna() if "ls_return" in monthly_df.columns else pd.Series(dtype=float)
    std_ls = float(ls_non_na.std(ddof=1)) if len(ls_non_na) > 1 else float("nan")
    annual_ret = mean_ls * 12.0 if pd.notna(mean_ls) else float("nan")
    sharpe = (mean_ls / std_ls) * math.sqrt(12.0) if pd.notna(std_ls) and std_ls > 0 else float("nan")

    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "oos_r2": oos_r2,
                "rmse": rmse,
                "mean_monthly_ls": mean_ls,
                "annual_ret": annual_ret,
                "sharpe": sharpe,
                "mean_ic": mean_ic,
            }
        ]
    )
    return summary_df, monthly_df


def build_window_counts_by_month(split_arrays: dict[str, SplitArrays]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in VALID_SPLITS:
        arrays = split_arrays[split]
        month_series = pd.Series(pd.to_datetime(arrays.month_ns), name="end_month")
        counts = month_series.value_counts(sort=False).rename_axis("end_month").reset_index(name="n_windows")
        counts["split"] = split
        frames.append(counts[["end_month", "split", "n_windows"]].sort_values("end_month", kind="mergesort"))
    return pd.concat(frames, ignore_index=True)


def print_window_metadata(split_arrays: dict[str, SplitArrays], window_counts_df: pd.DataFrame) -> None:
    print("\nWindow metadata")
    for split in VALID_SPLITS:
        arrays = split_arrays[split]
        end_months = pd.to_datetime(arrays.month_ns)
        print(f"{split}_windows: {len(arrays.y):,}")
        print(f"{split}_end_month_min: {end_months.min().date()}")
        print(f"{split}_end_month_max: {end_months.max().date()}")
        print(f"{split}_unique_end_months: {pd.Series(end_months).nunique()}")

    test_counts = window_counts_df.loc[window_counts_df["split"] == "test", "n_windows"]
    avg_test = float(test_counts.mean()) if len(test_counts) else float("nan")
    print(f"test_avg_windows_per_month: {avg_test:.2f}")


def run_alignment_check(df_raw: pd.DataFrame, test_arrays: SplitArrays, seq_len: int) -> None:
    n_samples = min(5, len(test_arrays.y))
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(len(test_arrays.y), size=n_samples, replace=False)
    raw_lookup = df_raw.set_index(["permno", "month"])["ret_fwd"]
    grouped = {permno: group.reset_index(drop=True) for permno, group in df_raw.groupby("permno", sort=False)}

    lines: list[str] = []
    for idx in sample_idx:
        permno = int(test_arrays.permno[idx])
        end_month = pd.Timestamp(test_arrays.month_ns[idx])
        y_true_raw = float(test_arrays.y_raw[idx])

        raw_value = float(raw_lookup.loc[(permno, end_month)])
        if not np.isclose(raw_value, y_true_raw, atol=1e-7, rtol=1e-6):
            raise Exception(
                "Alignment mismatch: "
                f"permno={permno}, end_month={end_month.date()}, window_y_raw={y_true_raw:.10f}, "
                f"raw_ret_fwd={raw_value:.10f}"
            )

        perm_df = grouped[permno]
        locs = np.flatnonzero(perm_df["month"].to_numpy() == end_month.to_datetime64())
        if locs.size != 1:
            raise Exception(
                f"Alignment reconstruction failed for permno={permno}, end_month={end_month.date()}: "
                f"expected exactly one match, found {locs.size}."
            )

        end_pos = int(locs[0])
        start_pos = end_pos - seq_len + 1
        if start_pos < 0:
            raise Exception(
                f"Alignment reconstruction failed for permno={permno}, end_month={end_month.date()}: "
                "insufficient history for sampled test window."
            )

        window_months = perm_df["month"].iloc[start_pos : end_pos + 1].reset_index(drop=True)
        if len(window_months) != seq_len:
            raise Exception(
                f"Alignment reconstruction failed for permno={permno}, end_month={end_month.date()}: "
                f"expected {seq_len} months, got {len(window_months)}."
            )

        month_ord = window_months.dt.to_period("M").astype("int64").to_numpy()
        if seq_len > 1 and not np.all((month_ord[1:] - month_ord[:-1]) == 1):
            raise Exception(
                f"Alignment reconstruction failed for permno={permno}, end_month={end_month.date()}: "
                "sampled window months are not consecutive."
            )

        month_list = [ts.strftime("%Y-%m-%d") for ts in window_months]
        lines.extend(
            [
                f"permno: {permno}",
                f"window_months: {month_list}",
                f"end_month: {end_month.strftime('%Y-%m-%d')}",
                f"y_true_train_target: {float(test_arrays.y[idx]):.10f}",
                f"y_true_raw: {y_true_raw:.10f}",
                f"raw_ret_fwd_lookup: {raw_value:.10f}",
                "",
            ]
        )

    text = "\n".join(lines).strip() + "\n"
    ALIGNMENT_SAMPLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_SAMPLES_OUT.write_text(text, encoding="utf-8")
    print("\nAlignment samples")
    print(text.rstrip())
    print(f"Saved alignment samples: {ALIGNMENT_SAMPLES_OUT}")


def print_extreme_months(monthly_df: pd.DataFrame) -> None:
    print("\n10 most negative IC months")
    print(monthly_df.nsmallest(10, "ic")[["end_month", "ic", "n_obs"]].to_string(index=False))

    print("\n10 most positive IC months")
    print(monthly_df.nlargest(10, "ic")[["end_month", "ic", "n_obs"]].to_string(index=False))

    print("\n10 most negative long-short months")
    print(monthly_df.nsmallest(10, "ls_return")[["end_month", "ls_return", "n_obs"]].to_string(index=False))


def build_model_name(args: argparse.Namespace) -> str:
    model_name = f"GRU_L{args.seq_len}_H{args.hidden_size}_T{args.target}"
    if args.target == "xs" and args.eval_on_raw:
        model_name += "_Eraw"
    return model_name


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Debug mode: {args.debug}")
    print(f"Target mode: {args.target}")
    print(f"Evaluate on raw: {args.eval_on_raw}")

    df = load_input(args.input_path)
    if args.target == "xs":
        df = add_cross_sectional_target(df)
        target_col = "ret_fwd_xs"
    else:
        target_col = "ret_fwd"

    print(f"Loaded rows: {len(df):,}")
    print(df["split"].value_counts().to_string())

    split_arrays = build_sequence_splits(df, seq_len=args.seq_len, target_col=target_col)

    window_counts_df = build_window_counts_by_month(split_arrays)
    print_window_metadata(split_arrays, window_counts_df)
    if args.debug:
        WINDOW_COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        window_counts_df.to_csv(WINDOW_COUNTS_OUT, index=False)
        print(f"Saved window counts by month: {WINDOW_COUNTS_OUT}")
        run_alignment_check(df_raw=df, test_arrays=split_arrays["test"], seq_len=args.seq_len)

    train_loader = make_loader(split_arrays["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(split_arrays["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(split_arrays["test"], batch_size=args.batch_size, shuffle=False)

    model = GRURegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    use_raw_for_eval = args.eval_on_raw
    train_eval_mean = float(split_arrays["train"].y_raw.mean()) if use_raw_for_eval else float(split_arrays["train"].y.mean())

    if args.debug:
        baseline_rows: list[pd.DataFrame] = []

        const_train_mean_pred = np.full(len(split_arrays["test"].y), train_eval_mean, dtype=np.float32)
        const_train_mean_df = predictions_from_arrays(
            split_arrays["test"], const_train_mean_pred, use_raw_for_eval=use_raw_for_eval
        )
        const_train_mean_summary, _ = summarize_predictions(
            const_train_mean_df, train_mean=train_eval_mean, model_name="ConstTrainMean"
        )
        baseline_rows.append(const_train_mean_summary)

        const_zero_pred = np.zeros(len(split_arrays["test"].y), dtype=np.float32)
        const_zero_df = predictions_from_arrays(
            split_arrays["test"], const_zero_pred, use_raw_for_eval=use_raw_for_eval
        )
        const_zero_summary, _ = summarize_predictions(
            const_zero_df, train_mean=train_eval_mean, model_name="ConstZero"
        )
        baseline_rows.append(const_zero_summary)

        x_train_last = split_arrays["train"].x[:, -1, :]
        x_test_last = split_arrays["test"].x[:, -1, :]
        ridge = Ridge(alpha=1.0)
        ridge.fit(x_train_last, split_arrays["train"].y)
        ridge_pred = ridge.predict(x_test_last).astype(np.float32, copy=False)
        ridge_pred_df = predictions_from_arrays(split_arrays["test"], ridge_pred, use_raw_for_eval=use_raw_for_eval)
        ridge_summary_df, _ = summarize_predictions(
            ridge_pred_df, train_mean=train_eval_mean, model_name="LastStepRidge"
        )

        baseline_summary_df = pd.concat(baseline_rows, ignore_index=True)
        BASELINES_SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
        RIDGE_SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
        baseline_summary_df.to_csv(BASELINES_SUMMARY_OUT, index=False)
        ridge_summary_df.to_csv(RIDGE_SUMMARY_OUT, index=False)

        print("\nConstant baselines")
        print(baseline_summary_df.to_string(index=False))
        print("\nLast-step ridge baseline")
        print(ridge_summary_df.to_string(index=False))
        print(f"Saved constant baselines summary: {BASELINES_SUMMARY_OUT}")
        print(f"Saved last-step ridge summary: {RIDGE_SUMMARY_OUT}")

    test_pred_df = predict(model, test_loader, device, use_raw_for_eval=use_raw_for_eval)
    summary_df, monthly_diag_df = summarize_predictions(
        test_pred_df,
        train_mean=train_eval_mean,
        model_name=build_model_name(args),
    )

    output_pred_df = test_pred_df[["permno", "end_month", "y_true", "y_pred"]].copy()

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    output_pred_df.to_csv(args.predictions_out, index=False)
    summary_df.to_csv(args.summary_out, index=False)

    if args.debug:
        DEBUG_PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
        MONTHLY_DIAGNOSTICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        output_pred_df.to_csv(DEBUG_PREDICTIONS_OUT, index=False)
        monthly_diag_df.to_csv(MONTHLY_DIAGNOSTICS_OUT, index=False)
        print(f"Saved debug test predictions: {DEBUG_PREDICTIONS_OUT}")
        print(f"Saved monthly diagnostics: {MONTHLY_DIAGNOSTICS_OUT}")
        print_extreme_months(monthly_diag_df)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    summary_row = summary_df.iloc[0]
    print(
        "Long-short stats | "
        f"mean_monthly_ls={summary_row['mean_monthly_ls']:.8f} | "
        f"annual_ret={summary_row['annual_ret']:.8f} | "
        f"sharpe={summary_row['sharpe']:.8f}"
    )
    print(f"Saved test predictions: {args.predictions_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
