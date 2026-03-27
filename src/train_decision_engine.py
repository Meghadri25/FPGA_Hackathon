"""
SMART ENERGY DECISION ENGINE v3
- Multi-horizon forecasting (15-min + 1-hour)
- Risk-based alert states (NORMAL/WATCH/CRITICAL)
- Peak cost saver with Indian Rupee pricing
- Multi-client batch processing
- FPGA-ready decision outputs
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


np.random.seed(42)
torch.manual_seed(42)


CONFIG = {
    "n_lags": 16,
    "data_size": 100000,
    "epochs": 140,
    "batch_size": 128,
    "learning_rate": 0.001,
    "train_ratio": 0.7,
    "early_stopping_patience": 18,
    "loss_weights": {"h1": 1.0, "h4": 1.6},
    "num_clients": 8,
    # Cost Saver Config (India)
    "peak_charge_per_kw_monthly_inr": 500,  # ₹/kW/month
    "action_load_reduction_percent": 0.04,  # assume 4% load reduction from action
}


class StrongShortHorizonModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.head_h1 = nn.Linear(32, 1)
        self.head_h4 = nn.Linear(32, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_h1(feat), self.head_h4(feat)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["is_weekend"] = (out["dow"] >= 5).astype(float)

    return out


def create_features(df: pd.DataFrame, client: str, n_lags: int) -> pd.DataFrame:
    d = df[["timestamp", client]].copy()

    for i in range(1, n_lags + 1):
        d[f"lag_{i}"] = d[client].shift(i)

    d["roll_mean_3"] = d[client].shift(1).rolling(3).mean()
    d["roll_mean_12"] = d[client].shift(1).rolling(12).mean()
    d["roll_std_12"] = d[client].shift(1).rolling(12).std()

    d = add_time_features(d)

    d["target_h1"] = d[client].shift(-1)
    d["target_h4"] = d[client].shift(-4)

    d = d.dropna().reset_index(drop=True)
    return d


def calc_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        r2 = 0.0
    else:
        r2 = float(1.0 - np.sum(err ** 2) / denom)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def classify_alert_state(
    pred_1h: float,
    threshold_p90: float,
    pred_15m: float,
    anomaly_score: float,
) -> tuple:
    """
    Classify alert state: NORMAL, WATCH, or CRITICAL
    
    Returns: (state, risk_score)
    """
    level_ratio = pred_1h / threshold_p90
    trend_ratio = max(0, (pred_1h - pred_15m) / threshold_p90)

    risk = 0.6 * level_ratio + 0.25 * trend_ratio + 0.15 * anomaly_score

    if risk >= 1.0:
        state = "CRITICAL"
    elif risk >= 0.85:
        state = "WATCH"
    else:
        state = "NORMAL"

    return state, risk


def get_action_recommendation(state: str) -> str:
    """Map alert state to action recommendation."""
    actions = {
        "NORMAL": "Continue normal operations. Monitor next forecast window.",
        "WATCH": "Soft mitigation: defer non-critical loads, prepare demand-response capacity.",
        "CRITICAL": "URGENT: activate demand-response now. Defer HVAC, reduce non-critical loads by 4%.",
    }
    return actions.get(state, "Unknown state")


def calculate_cost_impact(
    pred_1h: float,
    monthly_peak: float,
    peak_charge_per_kw: float,
    action_reduction_percent: float,
) -> dict:
    """Calculate cost saver metrics in INR."""
    
    if pred_1h > monthly_peak:
        peak_overage_kw = pred_1h - monthly_peak
        potential_monthly_charge_inr = peak_overage_kw * peak_charge_per_kw
        potent_annual_charge_inr = potential_monthly_charge_inr * 12

        reduction_kw = pred_1h * action_reduction_percent
        savings_per_action_inr = min(reduction_kw * peak_charge_per_kw, potential_monthly_charge_inr)
        savings_annual_inr = savings_per_action_inr * 12
    else:
        potential_monthly_charge_inr = 0.0
        potent_annual_charge_inr = 0.0
        reduction_kw = 0.0
        savings_per_action_inr = 0.0
        savings_annual_inr = 0.0

    return {
        "peak_risk_charge_monthly_inr": float(potential_monthly_charge_inr),
        "peak_risk_charge_annual_inr": float(potent_annual_charge_inr),
        "action_load_reduction_kw": float(reduction_kw),
        "savings_if_action_taken_monthly_inr": float(savings_per_action_inr),
        "savings_if_action_taken_annual_inr": float(savings_annual_inr),
    }


def train_one_client(df: pd.DataFrame, client: str, cfg: dict) -> dict:
    """Train model for one client and produce decision outputs."""
    
    series = df[client].values
    rolling_std = pd.Series(series).rolling(window=10000).std()
    start_idx = int(rolling_std.idxmax())

    local = df.iloc[start_idx : start_idx + cfg["data_size"]].copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"])

    feat_df = create_features(local, client, cfg["n_lags"])

    lag_cols = [f"lag_{i}" for i in range(1, cfg["n_lags"] + 1)]
    feature_cols = lag_cols + [
        "roll_mean_3",
        "roll_mean_12",
        "roll_std_12",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]

    X = feat_df[feature_cols].values
    y1 = feat_df[["target_h1"]].values
    y4 = feat_df[["target_h4"]].values

    scaler_x = MinMaxScaler()
    scaler_y1 = MinMaxScaler()
    scaler_y4 = MinMaxScaler()

    Xs = scaler_x.fit_transform(X)
    y1s = scaler_y1.fit_transform(y1)
    y4s = scaler_y4.fit_transform(y4)

    split = int(len(Xs) * cfg["train_ratio"])

    X_train = torch.FloatTensor(Xs[:split])
    X_test = torch.FloatTensor(Xs[split:])

    y1_train = torch.FloatTensor(y1s[:split])
    y1_test = y1s[split:]

    y4_train = torch.FloatTensor(y4s[:split])
    y4_test = y4s[split:]

    model = StrongShortHorizonModel(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()

        p1, p4 = model(X_train)
        loss = (
            cfg["loss_weights"]["h1"] * loss_fn(p1, y1_train)
            + cfg["loss_weights"]["h4"] * loss_fn(p4, y4_train)
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vp1, vp4 = model(X_test)
            val = (
                cfg["loss_weights"]["h1"]
                * loss_fn(vp1, torch.FloatTensor(y1_test))
                + cfg["loss_weights"]["h4"]
                * loss_fn(vp4, torch.FloatTensor(y4_test))
            )

        v = float(val.item())
        scheduler.step(v)

        if v < best_val:
            best_val = v
            patience_count = 0
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= cfg["early_stopping_patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred1_s, pred4_s = model(X_test)

    pred1 = scaler_y1.inverse_transform(pred1_s.numpy())
    pred4 = scaler_y4.inverse_transform(pred4_s.numpy())

    y1_true = scaler_y1.inverse_transform(y1_test)
    y4_true = scaler_y4.inverse_transform(y4_test)

    m1 = calc_reg_metrics(y1_true, pred1)
    m4 = calc_reg_metrics(y4_true, pred4)

    resid1 = np.abs(y1_true.flatten() - pred1.flatten())
    anom_threshold = float(np.mean(resid1) + 2.5 * np.std(resid1))
    anomalies = (resid1 > anom_threshold).astype(int)
    anomaly_rate = float(np.mean(anomalies) * 100.0)

    # Compute monthly peak threshold from training data
    train_y4_true = scaler_y4.inverse_transform(y4s[:split])
    monthly_peak_threshold = float(np.percentile(train_y4_true, 90))

    # Compute alert states for test window
    alert_states = []
    for i in range(len(pred1)):
        pred_15m = float(pred1[i, 0])
        pred_1h = float(pred4[i, 0])
        anomaly_norm = float(anomalies[i] / (anom_threshold + 1e-9))

        state, risk = classify_alert_state(
            pred_1h, monthly_peak_threshold, pred_15m, anomaly_norm
        )
        alert_states.append({"state": state, "risk_score": risk})

    # Aggregated counts
    state_counts = {}
    for a in alert_states:
        s = a["state"]
        state_counts[s] = state_counts.get(s, 0) + 1

    # Cost impact example (using mean prediction)
    cost_info = calculate_cost_impact(
        float(np.mean(pred4)),
        monthly_peak_threshold,
        cfg["peak_charge_per_kw_monthly_inr"],
        cfg["action_load_reduction_percent"],
    )

    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'client': client,
        'config': cfg,
        'scaler_y1': scaler_y1,
        'scaler_y4': scaler_y4,
    }, f'model_{client}.pt')

    return {
        "client": client,
        "feature_count": int(X_train.shape[1]),
        "h1": m1,
        "h4": m4,
        "anomaly_threshold_kw": anom_threshold,
        "anomaly_rate_percent": anomaly_rate,
        "monthly_peak_threshold_kw": monthly_peak_threshold,
        "alert_state_distribution": state_counts,
        "cost_saver": cost_info,
    }


def main():
    print("=" * 90)
    print("SMART ENERGY DECISION ENGINE v3")
    print("Forecasting + Alert States + Cost Saver (INR) + Multi-Client")
    print("=" * 90)

    df = pd.read_csv("LD2011_2014.txt", sep=";", decimal=",")
    df.columns = ["timestamp"] + list(df.columns[1:])

    client_stats = {}
    for col in df.columns[1:]:
        s = float(df[col].std())
        if s > 0:
            client_stats[col] = s

    selected = sorted(client_stats.items(), key=lambda x: x[1], reverse=True)
    selected_clients = [k for k, _ in selected[: CONFIG["num_clients"]]]

    print(f"\nSelected {len(selected_clients)} clients:")
    for i, c in enumerate(selected_clients, 1):
        print(f"  {i}. {c}")

    results = []
    for idx, c in enumerate(selected_clients, start=1):
        print(f"\n[{idx}/{len(selected_clients)}] Training {c}...")
        res = train_one_client(df, c, CONFIG)
        results.append(res)

        print(
            f"  R2(15m)={res['h1']['r2']:.4f}, R2(1h)={res['h4']['r2']:.4f}"
        )
        print(f"  Alert States: {res['alert_state_distribution']}")
        print(
            f"  Cost Risk: INR {res['cost_saver']['peak_risk_charge_annual_inr']:.0f}/year"
        )
        print(
            f"  Savings Potential: INR {res['cost_saver']['savings_if_action_taken_annual_inr']:.0f}/year"
        )

    # Save summary
    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "client": r["client"],
                "r2_15m": r["h1"]["r2"],
                "r2_1h": r["h4"]["r2"],
                "rmse_1h_kw": r["h4"]["rmse"],
                "anomaly_rate_pct": r["anomaly_rate_percent"],
                "monthly_peak_threshold_kw": r["monthly_peak_threshold_kw"],
                "critical_events": r["alert_state_distribution"].get("CRITICAL", 0),
                "watch_events": r["alert_state_distribution"].get("WATCH", 0),
                "cost_risk_annual_inr": r["cost_saver"]["peak_risk_charge_annual_inr"],
                "savings_potential_annual_inr": r["cost_saver"]["savings_if_action_taken_annual_inr"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("decision_engine_summary.csv", index=False)

    with Path("decision_engine_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": CONFIG,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("TRAINING COMPLETE")
    print("=" * 90)
    print(f"Average R2 (15-min): {summary_df['r2_15m'].mean():.4f}")
    print(f"Average R2 (1-hour): {summary_df['r2_1h'].mean():.4f}")
    print(f"Total Annual Cost Risk (all clients): ₹{summary_df['cost_risk_annual_inr'].sum():.0f}")
    print(f"Total Annual Savings Potential: ₹{summary_df['savings_potential_annual_inr'].sum():.0f}")
    print()
    print("Saved: decision_engine_summary.csv, decision_engine_results.json")


if __name__ == "__main__":
    main()
