"""
04_bayesian_analysis.py
==========================
Análise completa dos resultados de Otimização Bayesiana V1 (AdamW auto)
para YOLOv8x (detecção), gerando figuras e relatório para dissertação.

Nota: Na V1, optimizer="auto" fez o Ultralytics fixar AdamW com lr=0.002
e momentum=0.9, ignorando os valores sugeridos pelo Optuna.

Autor: Gabriel A. Ferreira Gualda
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from itertools import combinations

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
EXPERIMENT_DIR = Path("F:/MESTRADO_Bilada/01_Mestrado/MODEL/01_bayezian_optimization/yolov8x_seg_bayesian_opt")

OUTPUT_DIR = Path("F:/MESTRADO_Bilada/01_Mestrado/MODEL/02_bayezian_analysis/bayezian_analysis_TESTE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_METRIC = "metrics/mAP50(B)"
TARGET_METRIC_LABEL = "mAP50 (Box)"

EXTRA_METRICS = [
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
]

V1_BEST_MAP50 = 0.7602
V2_BEST_MAP50 = 0.7849
V3_BEST_MAP50 = 0.7783

HP_DISPLAY_NAMES = {
    "freeze": "Freeze Layers",
    "lr0": "Learning Rate (lr0)",
    "lrf": "Final LR Factor (lrf)",
    "momentum": "Momentum",
    "weight_decay": "Weight Decay",
    "warmup_epochs": "Warmup Epochs",
    "box": "Box Loss Weight",
    "cls": "Class Loss Weight",
    "hsv_h": "HSV Hue Aug.",
    "hsv_s": "HSV Sat. Aug.",
    "hsv_v": "HSV Val. Aug.",
    "degrees": "Rotation (deg)",
    "translate": "Translation",
    "scale": "Scale Aug.",
    "shear": "Shear (deg)",
    "perspective": "Perspective",
    "flipud": "Flip Up-Down",
    "fliplr": "Flip Left-Right",
    "mosaic": "Mosaic",
    "mixup": "MixUp",
    "copy_paste": "Copy-Paste",
}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ============================================================================
# 1. CARREGAMENTO DOS DADOS (OPTUNA V1)
# ============================================================================
def load_experiment_data(experiment_dir: Path):
    csv_path = experiment_dir / "all_trials.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    
    df_optuna = pd.read_csv(csv_path)
    
    records = []
    trials_epochs = {}
    
    for _, row in df_optuna.iterrows():
        trial_num = int(row["number"])
        trial_dir_name = f"train_{trial_num:04d}"
        trial_dir = experiment_dir / trial_dir_name
        
        record = {
            "trial_dir": trial_dir_name,
            "trial_number": trial_num + 1,
            "trial_order": trial_num,
            TARGET_METRIC: row.get("value", np.nan),
            "time_total_s": row.get("duration", 0),
        }
        
        for col in df_optuna.columns:
            if col.startswith("params_"):
                hp_name = col.replace("params_", "")
                record[f"hp_{hp_name}"] = row[col]
        
        results_csv = trial_dir / "weights" / "results.csv"
        if results_csv.exists():
            try:
                ep_df = pd.read_csv(results_csv)
                ep_df.columns = [c.strip() for c in ep_df.columns]
                trials_epochs[trial_dir_name] = ep_df
                
                last_row = ep_df.iloc[-1]
                for metric in EXTRA_METRICS:
                    col_name = metric.strip()
                    if col_name in ep_df.columns:
                        record[metric] = last_row[col_name]
                    else:
                        record[metric] = np.nan
                
                record["num_epochs"] = len(ep_df)
                
                for loss_key in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
                    if loss_key in ep_df.columns:
                        record[loss_key] = last_row[loss_key]
                    else:
                        record[loss_key] = np.nan
                        
            except Exception as e:
                print(f"  Aviso: Erro ao ler {results_csv}: {e}")
                record["num_epochs"] = 20
                for metric in EXTRA_METRICS:
                    record[metric] = np.nan
        else:
            record["num_epochs"] = 20
            for metric in EXTRA_METRICS:
                record[metric] = np.nan
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values("trial_order").reset_index(drop=True)
    df["trial_number"] = range(1, len(df) + 1)
    
    trial_order = df["trial_dir"].tolist()
    
    return df, trials_epochs, trial_order


# ============================================================================
# 2. CURVA DE CONVERGÊNCIA
# ============================================================================
def plot_convergence_curve(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    trials = df["trial_number"].values
    metric_vals = df[TARGET_METRIC].values
    best_so_far = np.maximum.accumulate(metric_vals)
    
    ax.scatter(trials, metric_vals, c="#DD8452", s=70, zorder=5,
               edgecolors="white", linewidths=0.8, label="Valor por trial")
    ax.plot(trials, best_so_far, color="#C44E52", linewidth=2.2,
            linestyle="-", marker="D", markersize=5, zorder=4,
            label="Melhor acumulado")
    ax.fill_between(trials, 0, best_so_far, alpha=0.08, color="#C44E52")
    
    best_idx = np.argmax(metric_vals)
    ax.annotate(
        f"Melhor: {metric_vals[best_idx]:.4f}\n(Trial {trials[best_idx]})",
        xy=(trials[best_idx], metric_vals[best_idx]),
        xytext=(trials[best_idx] + 0.5, metric_vals[best_idx] - 0.06),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )
    
    ax.set_xlabel("Número do Trial (ordem cronológica)")
    ax.set_ylabel(TARGET_METRIC_LABEL)
    ax.set_title("Curva de Convergência da Otimização Bayesiana (AdamW auto)")
    ax.set_xticks(trials)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    path = output_dir / "fig01_convergence_curve_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 3. EVOLUÇÃO DAS MÉTRICAS
# ============================================================================
def plot_metrics_evolution(df, output_dir):
    metrics_to_plot = [
        (TARGET_METRIC, TARGET_METRIC_LABEL),
        ("metrics/mAP50-95(B)", "mAP50-95 (Box)"),
        ("metrics/precision(B)", "Precision (Box)"),
    ]
    
    available = [(m, l) for m, l in metrics_to_plot if m in df.columns and df[m].notna().any()]
    
    if not available:
        print("  [SKIP] Nenhuma métrica extra disponível")
        return None
    
    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 5), sharey=False)
    if len(available) == 1:
        axes = [axes]
    
    colors = ["#DD8452", "#55A868", "#4C72B0"]
    
    for ax, (metric, label), color in zip(axes, available, colors):
        trials = df["trial_number"].values
        vals = df[metric].values
        valid = ~np.isnan(vals)
        ax.scatter(trials[valid], vals[valid], c=color, s=60, edgecolors="white",
                   linewidths=0.6, zorder=5, alpha=0.9)
        if valid.sum() >= 3:
            z = np.polyfit(trials[valid], vals[valid], deg=2)
            p = np.poly1d(z)
            x_smooth = np.linspace(trials[valid].min(), trials[valid].max(), 100)
            ax.plot(x_smooth, p(x_smooth), color=color, linewidth=1.8,
                    linestyle="--", alpha=0.7, label="Tendência (poly2)")
        ax.set_xlabel("Trial")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
    
    fig.suptitle("Evolução das Métricas por Trial (AdamW auto)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "fig02_metrics_evolution_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 4. IMPORTÂNCIA DOS HIPERPARÂMETROS
# ============================================================================
def compute_hp_importance(df):
    hp_cols = [c for c in df.columns if c.startswith("hp_")]
    hp_names = [c.replace("hp_", "") for c in hp_cols]
    X = df[hp_cols].values
    y = df[TARGET_METRIC].values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "hyperparameter": hp_names, "importance": importances,
        "display_name": [HP_DISPLAY_NAMES.get(n, n) for n in hp_names],
    }).sort_values("importance", ascending=False)
    return importance_df, rf, hp_cols, hp_names


def plot_hp_importance(importance_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    imp = importance_df.sort_values("importance", ascending=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(imp)))
    bars = ax.barh(imp["display_name"], imp["importance"], color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, imp["importance"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importância Relativa (Random Forest)")
    ax.set_title("Importância dos Hiperparâmetros — V1 (AdamW auto)\n(lr0 e momentum ignorados pelo Ultralytics)")
    ax.set_xlim(0, imp["importance"].max() * 1.2)
    fig.tight_layout()
    path = output_dir / "fig03_hp_importance_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 5. PARTIAL DEPENDENCE PLOTS
# ============================================================================
def plot_partial_dependence_plots(df, rf, hp_cols, hp_names, importance_df, output_dir, top_n=6):
    top_hps = importance_df.head(top_n)["hyperparameter"].tolist()
    top_indices = [hp_names.index(hp) for hp in top_hps]
    top_display = [HP_DISPLAY_NAMES.get(hp, hp) for hp in top_hps]
    X = df[hp_cols].values
    mask = ~np.isnan(df[TARGET_METRIC].values)
    X = X[mask]
    ncols = 3
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = axes.flatten()
    for i, (idx, hp_name, display_name) in enumerate(zip(top_indices, top_hps, top_display)):
        ax = axes[i]
        pdp_result = partial_dependence(rf, X, features=[idx], kind="average", grid_resolution=50)
        grid_values = pdp_result["grid_values"][0]
        avg_pred = pdp_result["average"][0]
        ax.plot(grid_values, avg_pred, color="#DD8452", linewidth=2)
        ax.fill_between(grid_values, avg_pred.min(), avg_pred, alpha=0.1, color="#DD8452")
        ax.scatter(X[:, idx], [avg_pred.min()] * len(X[:, idx]), marker="|", color="black", s=50, alpha=0.5)
        ax.set_xlabel(display_name)
        ax.set_ylabel(f"Efeito parcial em\n{TARGET_METRIC_LABEL}")
        ax.set_title(display_name, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Partial Dependence Plots — V1 (AdamW auto)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "fig04_partial_dependence_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 6. HEATMAPS BIDIMENSIONAIS
# ============================================================================
def plot_heatmaps(df, rf, hp_cols, hp_names, importance_df, output_dir, top_n=3):
    top_hps = importance_df.head(top_n)["hyperparameter"].tolist()
    pairs = list(combinations(range(len(top_hps)), 2))
    X = df[hp_cols].values
    mask = ~np.isnan(df[TARGET_METRIC].values)
    X = X[mask]
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        hp_i, hp_j = top_hps[i], top_hps[j]
        idx_i, idx_j = hp_names.index(hp_i), hp_names.index(hp_j)
        display_i, display_j = HP_DISPLAY_NAMES.get(hp_i, hp_i), HP_DISPLAY_NAMES.get(hp_j, hp_j)
        pdp_result = partial_dependence(rf, X, features=[(idx_i, idx_j)], kind="average", grid_resolution=25)
        grid_i, grid_j = pdp_result["grid_values"][0], pdp_result["grid_values"][1]
        pdp_values = pdp_result["average"][0]
        im = ax.contourf(grid_i, grid_j, pdp_values.T, levels=20, cmap="RdYlBu_r")
        fig.colorbar(im, ax=ax, label=TARGET_METRIC_LABEL, shrink=0.8)
        ax.scatter(X[:, idx_i], X[:, idx_j], c="black", s=40, edgecolors="white", linewidths=0.5, zorder=5, alpha=0.8)
        ax.set_xlabel(display_i)
        ax.set_ylabel(display_j)
        ax.set_title(f"{display_i} vs. {display_j}")
    fig.suptitle("Interação entre Hiperparâmetros — V1 (AdamW auto)", fontsize=14, y=1.03)
    fig.tight_layout()
    path = output_dir / "fig05_heatmaps_2d_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 7. CURVAS DE TREINAMENTO POR ÉPOCA
# ============================================================================
def plot_training_curves(df, trials_epochs, output_dir, top_k=3):
    top_trials = df.nlargest(top_k, TARGET_METRIC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, top_k))
    for idx, (_, row) in enumerate(top_trials.iterrows()):
        trial_dir = row["trial_dir"]
        trial_num = row["trial_number"]
        if trial_dir not in trials_epochs:
            continue
        ep_df = trials_epochs[trial_dir]
        epoch_col = "epoch" if "epoch" in ep_df.columns else ep_df.columns[0]
        epochs = ep_df[epoch_col].values
        map50_col = TARGET_METRIC.strip()
        if map50_col in ep_df.columns:
            axes[0].plot(epochs, ep_df[map50_col].values, color=colors[idx], linewidth=1.8, marker="o", markersize=4,
                         label=f"Trial {int(trial_num)} (final: {row[TARGET_METRIC]:.4f})")
        loss_col = "val/box_loss"
        if loss_col in ep_df.columns:
            axes[1].plot(epochs, ep_df[loss_col].values, color=colors[idx], linewidth=1.8, marker="s", markersize=4,
                         label=f"Trial {int(trial_num)}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel(TARGET_METRIC_LABEL)
    axes[0].set_title(f"Evolução de {TARGET_METRIC_LABEL} por Época")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Box Loss (val)")
    axes[1].set_title("Evolução da Box Loss (val)")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Curvas de Treinamento — Top {top_k} Trials (AdamW auto)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "fig06_training_curves_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 8. SCATTER MATRIX
# ============================================================================
def plot_hp_scatter_matrix(df, importance_df, output_dir, top_n=6):
    top_hps = importance_df.head(top_n)["hyperparameter"].tolist()
    ncols = 3
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = axes.flatten()
    metric_vals = df[TARGET_METRIC].values
    norm = plt.Normalize(vmin=metric_vals.min(), vmax=metric_vals.max())
    cmap = plt.cm.RdYlGn
    for i, hp in enumerate(top_hps):
        ax = axes[i]
        hp_col = f"hp_{hp}"
        display_name = HP_DISPLAY_NAMES.get(hp, hp)
        valid = ~np.isnan(df[hp_col]) & ~np.isnan(metric_vals)
        ax.scatter(df[hp_col][valid], metric_vals[valid], c=metric_vals[valid], cmap=cmap, norm=norm,
                   s=80, edgecolors="gray", linewidths=0.5, zorder=5)
        if valid.sum() >= 3:
            z = np.polyfit(df[hp_col][valid], metric_vals[valid], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df[hp_col][valid].min(), df[hp_col][valid].max(), 50)
            ax.plot(x_range, p(x_range), "k--", linewidth=1.2, alpha=0.5)
        ax.set_xlabel(display_name)
        ax.set_ylabel(TARGET_METRIC_LABEL)
        ax.set_title(display_name, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label=TARGET_METRIC_LABEL)
    fig.suptitle("Relação Hiperparâmetro vs. Métrica — V1 (AdamW auto)", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    path = output_dir / "fig07_hp_scatter_matrix_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 9. PAINEL RESUMO
# ============================================================================
def plot_summary_panel(df, importance_df, output_dir):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    trials = df["trial_number"].values
    metric_vals = df[TARGET_METRIC].values
    best_so_far = np.maximum.accumulate(metric_vals)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(trials, metric_vals, c="#DD8452", s=50, edgecolors="white", linewidths=0.5, zorder=5, label="Valor por trial")
    ax1.plot(trials, best_so_far, color="#C44E52", linewidth=2, marker="D", markersize=4, label="Melhor acumulado")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel(TARGET_METRIC_LABEL)
    ax1.set_title("(a) Curva de Convergência")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[0, 1])
    imp = importance_df.head(8).sort_values("importance", ascending=True)
    colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(imp)))
    ax2.barh(imp["display_name"], imp["importance"], color=colors_bar, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Importância Relativa")
    ax2.set_title("(b) Importância dos Hiperparâmetros")

    ax3 = fig.add_subplot(gs[1, 0])
    optimizers = ["AdamW\n(V1)", "SGD\n(V2)", "RMSProp\n(V3)"]
    opt_values = [V1_BEST_MAP50, V2_BEST_MAP50, V3_BEST_MAP50]
    opt_colors = ["#DD8452", "#4C72B0", "#55A868"]
    bars = ax3.bar(optimizers, opt_values, color=opt_colors, edgecolor="white", linewidth=0.5)
    ax3.set_ylabel(TARGET_METRIC_LABEL)
    ax3.set_title("(c) Comparação de Optimizers")
    ax3.set_ylim(0.7, 0.8)
    for bar, val in zip(bars, opt_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(metric_vals, bins=10, color="#DD8452", edgecolor="white", alpha=0.8)
    ax4.axvline(metric_vals.max(), color="#C44E52", linestyle="--", linewidth=2, label=f"Melhor: {metric_vals.max():.4f}")
    ax4.axvline(metric_vals.mean(), color="#55A868", linestyle="--", linewidth=2, label=f"Média: {metric_vals.mean():.4f}")
    ax4.set_xlabel(TARGET_METRIC_LABEL)
    ax4.set_ylabel("Frequência")
    ax4.set_title("(d) Distribuição do mAP50 (AdamW auto)")
    ax4.legend(fontsize=9)

    fig.suptitle("Resumo da Otimização Bayesiana V1 — YOLOv8x (AdamW auto)", fontsize=15, y=1.01)
    path = output_dir / "fig08_summary_panel_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 10. RADAR CHART
# ============================================================================
def plot_radar_chart(df, output_dir):
    metrics_radar = [
        (TARGET_METRIC, "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall"),
    ]
    available = [(m, l) for m, l in metrics_radar if m in df.columns and df[m].notna().any()]
    if len(available) < 3:
        print("  [SKIP] Métricas insuficientes para radar chart")
        return None
    best = df.loc[df[TARGET_METRIC].idxmax()]
    worst = df.loc[df[TARGET_METRIC].idxmin()]
    labels = [m[1] for m in available]
    best_vals = [best[m[0]] if not np.isnan(best[m[0]]) else 0 for m in available]
    worst_vals = [worst[m[0]] if not np.isnan(worst[m[0]]) else 0 for m in available]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    best_vals += best_vals[:1]
    worst_vals += worst_vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, best_vals, "o-", linewidth=2, color="#DD8452", label=f"Melhor (Trial #{int(best['trial_number'])})")
    ax.fill(angles, best_vals, alpha=0.15, color="#DD8452")
    ax.plot(angles, worst_vals, "s-", linewidth=2, color="#C44E52", label=f"Pior (Trial #{int(worst['trial_number'])})")
    ax.fill(angles, worst_vals, alpha=0.15, color="#C44E52")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.1, 1.1, 0.1)], fontsize=8, color="gray")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.tick_params(axis='x', pad=18)
    ax.set_title("Comparação de Métricas: Melhor vs. Pior Trial (AdamW auto)", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 0.95))
    fig.tight_layout()
    path = output_dir / "fig09_radar_chart_v1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ============================================================================
# 11. TABELAS
# ============================================================================
def generate_comparison_table(df):
    best = df.loc[df[TARGET_METRIC].idxmax()]
    worst = df.loc[df[TARGET_METRIC].idxmin()]
    median_idx = (df[TARGET_METRIC] - df[TARGET_METRIC].median()).abs().idxmin()
    median_trial = df.loc[median_idx]
    table_data = {
        "Configuração": ["Melhor Trial", "Trial Mediano", "Pior Trial"],
        "Trial": [f"#{int(best['trial_number'])}", f"#{int(median_trial['trial_number'])}", f"#{int(worst['trial_number'])}"],
        "mAP50 (Box)": [f"{best[TARGET_METRIC]:.4f}", f"{median_trial[TARGET_METRIC]:.4f}", f"{worst[TARGET_METRIC]:.4f}"],
    }
    for metric, label in [("metrics/mAP50-95(B)", "mAP50-95"), ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")]:
        if metric in df.columns and df[metric].notna().any():
            table_data[label] = [
                f"{best.get(metric, 0):.4f}" if not np.isnan(best.get(metric, np.nan)) else "N/A",
                f"{median_trial.get(metric, 0):.4f}" if not np.isnan(median_trial.get(metric, np.nan)) else "N/A",
                f"{worst.get(metric, 0):.4f}" if not np.isnan(worst.get(metric, np.nan)) else "N/A",
            ]
    table_data["Épocas"] = [int(best.get("num_epochs", 20)), int(median_trial.get("num_epochs", 20)), int(worst.get("num_epochs", 20))]
    return pd.DataFrame(table_data)


def generate_best_hp_table(df):
    search_space_bounds = {
        "freeze": (0, 21), "lr0": (1e-5, 1e-2), "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98), "weight_decay": (1e-4, 1e-3),
        "warmup_epochs": (0.0, 5.0), "box": (0.02, 0.2), "cls": (0.2, 4.0),
        "hsv_h": (0.0, 0.1), "hsv_s": (0.0, 0.9), "hsv_v": (0.0, 0.9),
        "degrees": (0.0, 45.0), "translate": (0.0, 0.9), "scale": (0.0, 0.9),
        "shear": (0.0, 10.0), "perspective": (0.0, 0.001),
        "flipud": (0.0, 1.0), "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 1.0), "mixup": (0.0, 1.0), "copy_paste": (0.0, 1.0),
    }
    best = df.loc[df[TARGET_METRIC].idxmax()]
    records = []
    for hp_name, (lo, hi) in search_space_bounds.items():
        col = f"hp_{hp_name}"
        val = best.get(col, np.nan)
        display = HP_DISPLAY_NAMES.get(hp_name, hp_name)
        if hp_name in ["lr0", "weight_decay", "perspective"]:
            val_str = f"{val:.6f}"
        elif hp_name == "freeze":
            val_str = f"{int(round(val))}"
        else:
            val_str = f"{val:.4f}"
        records.append({"Hiperparâmetro": display, "Limite Inferior": f"{lo}", "Limite Superior": f"{hi}", "Melhor Valor": val_str})
    return pd.DataFrame(records)


# ============================================================================
# 12. RELATÓRIO MARKDOWN
# ============================================================================
def generate_markdown_report(df, importance_df, comparison_table, hp_table, output_dir):
    best = df.loc[df[TARGET_METRIC].idxmax()]
    worst = df.loc[df[TARGET_METRIC].idxmin()]
    mean_metric = df[TARGET_METRIC].mean()
    std_metric = df[TARGET_METRIC].std()
    n_trials = len(df)
    metric_vals = df[TARGET_METRIC].values
    best_trial_idx = np.argmax(metric_vals) + 1
    top3_hps = importance_df.head(3)

    report = f"""# Relatório — Otimização Bayesiana V1 (AdamW auto)

## Modelo: YOLOv8x | Detecção de Cecropia

**Autor:** Gabriel A. Ferreira Gualda

---

## Observação Importante

Nesta versão (V1), o parâmetro `optimizer` não foi definido explicitamente, resultando
no comportamento padrão do Ultralytics (`optimizer="auto"`), que seleciona **AdamW** com
**lr=0.002** e **momentum=0.9** fixos, ignorando os valores sugeridos pelo Optuna para
lr0 e momentum. Os demais 19 hiperparâmetros foram otimizados normalmente.

---

## Resultados

- **Melhor mAP50:** {best[TARGET_METRIC]:.4f} (Trial #{int(best['trial_number'])})
- **Pior mAP50:** {worst[TARGET_METRIC]:.4f} (Trial #{int(worst['trial_number'])})
- **Média:** {mean_metric:.4f} (±{std_metric:.4f})
- **Melhor encontrado no Trial:** #{best_trial_idx}

### Comparação de Optimizers

| Versão | Optimizer | Melhor mAP50 |
|---|---|---|
| **V1** | **AdamW (auto)** | **{V1_BEST_MAP50:.4f}** |
| V2 | SGD | {V2_BEST_MAP50:.4f} |
| V3 | RMSProp | {V3_BEST_MAP50:.4f} |

### Tabela Comparativa

{comparison_table.to_markdown(index=False)}

### Top 3 Hiperparâmetros

| Ranking | Hiperparâmetro | Importância |
|---|---|---|
"""
    for i, (_, row) in enumerate(top3_hps.iterrows()):
        report += f"| {i+1} | {row['display_name']} | {row['importance']:.4f} |\n"

    report += f"""
### Melhores Hiperparâmetros

{hp_table.to_markdown(index=False)}

---

*Relatório gerado por `04_bayesian_analysis_v1.py`*
*Autor: Gabriel A. Ferreira Gualda*
"""
    report_path = output_dir / "relatorio_otimizacao_v1.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [OK] {report_path.name}")
    return report_path


# ============================================================================
# 13. EXPORTAR DADOS
# ============================================================================
def export_consolidated_data(df, importance_df, output_dir):
    df.to_csv(output_dir / "trials_summary_v1.csv", index=False)
    print(f"  [OK] trials_summary_v1.csv")
    importance_df.to_csv(output_dir / "hp_importance_v1.csv", index=False)
    print(f"  [OK] hp_importance_v1.csv")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("ANÁLISE DE RESULTADOS — OTIMIZAÇÃO BAYESIANA V1")
    print("YOLOv8x Detecção de Cecropia | Optimizer: AdamW (auto)")
    print("=" * 60)

    print("\n[1/10] Carregando dados...")
    df, trials_epochs, trial_order = load_experiment_data(EXPERIMENT_DIR)
    print(f"  Trials: {len(df)} | Melhor: {df[TARGET_METRIC].max():.4f} | Pior: {df[TARGET_METRIC].min():.4f} | Média: {df[TARGET_METRIC].mean():.4f}")
    print(f"  Curvas por época: {len(trials_epochs)} trials")

    print("\n[2/10] Curva de convergência...")
    plot_convergence_curve(df, OUTPUT_DIR)

    print("\n[3/10] Evolução das métricas...")
    plot_metrics_evolution(df, OUTPUT_DIR)

    print("\n[4/10] Importância dos hiperparâmetros...")
    importance_df, rf, hp_cols, hp_names = compute_hp_importance(df)
    plot_hp_importance(importance_df, OUTPUT_DIR)
    for _, row in importance_df.head(3).iterrows():
        print(f"    {row['display_name']:30s} {row['importance']:.4f}")

    print("\n[5/10] Partial Dependence Plots...")
    plot_partial_dependence_plots(df, rf, hp_cols, hp_names, importance_df, OUTPUT_DIR)

    print("\n[6/10] Heatmaps 2D...")
    plot_heatmaps(df, rf, hp_cols, hp_names, importance_df, OUTPUT_DIR)

    print("\n[7/10] Curvas de treinamento...")
    plot_training_curves(df, trials_epochs, OUTPUT_DIR)

    print("\n[8/10] Scatter matrix + painel + radar...")
    plot_hp_scatter_matrix(df, importance_df, OUTPUT_DIR)
    plot_summary_panel(df, importance_df, OUTPUT_DIR)
    plot_radar_chart(df, OUTPUT_DIR)

    print("\n[9/10] Tabelas e relatório...")
    comparison_table = generate_comparison_table(df)
    hp_table = generate_best_hp_table(df)
    generate_markdown_report(df, importance_df, comparison_table, hp_table, OUTPUT_DIR)

    print("\n[10/10] Exportando dados...")
    export_consolidated_data(df, importance_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("ANÁLISE V1 CONCLUÍDA!")
    print(f"Resultados em: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nArquivos gerados:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name:45s} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()