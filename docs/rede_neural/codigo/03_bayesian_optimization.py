import os
import yaml
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import time

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================
DATASET_YAML = "F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/model_dataset_structure.yaml"
MODEL_WEIGHTS = "yolov8x.pt"
NUM_SAMPLES = 30
MAX_EPOCHS = 20
DEVICE = 0  # 0 para GPU, "cpu" para CPU

# Diretório para salvar os resultados
RESULTS_DIR = "F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/bayezian_optimization"
EXP_NAME = "yolov8x_seg_bayesian_opt"

# ==============================================================================
# 1. FUNÇÃO OBJETIVO (equivalente ao train_yolo do Ray)
# ==============================================================================
def objective(trial):
    """
    Função objetivo do Optuna. Cada trial sugere hiperparâmetros,
    treina o YOLO e retorna o mAP50.
    """
    # --- Amostragem dos hiperparâmetros (mesmo espaço de busca anterior) ---
    
    # Hiperparâmetros de Otimizador
    freeze = trial.suggest_int("freeze", 0, 21)
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 1.0)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    warmup_epochs = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    box = trial.suggest_float("box", 0.02, 0.2)
    cls = trial.suggest_float("cls", 0.2, 4.0)
    
    # Augmentations
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
    degrees = trial.suggest_float("degrees", 0.0, 45.0)
    translate = trial.suggest_float("translate", 0.0, 0.9)
    scale = trial.suggest_float("scale", 0.0, 0.9)
    shear = trial.suggest_float("shear", 0.0, 10.0)
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    flipud = trial.suggest_float("flipud", 0.0, 1.0)
    fliplr = trial.suggest_float("fliplr", 0.0, 1.0)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 1.0)
    copy_paste = trial.suggest_float("copy_paste", 0.0, 1.0)

    # --- Diretório único para este trial ---
    trial_dir = os.path.join(RESULTS_DIR, EXP_NAME, f"train_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # --- Treinamento ---
    dataset_path = os.path.abspath(DATASET_YAML)
    model = YOLO(MODEL_WEIGHTS)

    try:
        results = model.train(
            data=dataset_path,
            epochs=MAX_EPOCHS,
            imgsz=893,
            batch=8,
            patience=10,
            workers=2,
            verbose=False,
            plots=False,
            device=DEVICE,
            exist_ok=True,
            val=True,
            project=trial_dir,
            name="weights",
            # Hiperparâmetros
            freeze=freeze,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            box=box,
            cls=cls,
            # Augmentations
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
        )

        # Extrai o mAP50 dos resultados
        mAP50 = results.results_dict.get("metrics/mAP50(M)", None)
        
        if mAP50 is None:
            # Tenta outras chaves possíveis
            mAP50 = results.results_dict.get("metrics/mAP50(B)", None)
        
        if mAP50 is None:
            print(f"  Trial {trial.number}: Métrica mAP50 não encontrada. Chaves disponíveis: {list(results.results_dict.keys())}")
            return 0.0

        print(f"  Trial {trial.number}: mAP50 = {mAP50:.4f}")
        return mAP50

    except Exception as e:
        print(f"  Trial {trial.number}: ERRO - {e}")
        traceback.print_exc()
        return 0.0


# ==============================================================================
# 2. EXECUÇÃO DA OTIMIZAÇÃO
# ==============================================================================
def main():
    print("=" * 60)
    print("Iniciando Otimização Bayesiana com Optuna")
    print("=" * 60)
    print(f"Dataset: {DATASET_YAML}")
    print(f"Modelo: {MODEL_WEIGHTS}")
    print(f"Amostras (Trials): {NUM_SAMPLES}")
    print(f"Épocas por Trial: {MAX_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Resultados: {RESULTS_DIR}/{EXP_NAME}")
    print("=" * 60)

    # Cria diretório de resultados
    exp_dir = os.path.join(os.path.abspath(RESULTS_DIR), EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # TPE Sampler (Tree-structured Parzen Estimator)
    # Equivalente ao BayesOptSearch do Ray Tune
    sampler = TPESampler(
        seed=42,
        n_startup_trials=5  # Equivalente ao random_search_steps=5
    )

    # MedianPruner: encerra trials ruins cedo (equivalente ao ASHA Scheduler)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )

    # Banco de dados SQLite para persistir o estudo (retoma se parar)
    storage_path = os.path.join(exp_dir, "optuna_study.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=EXP_NAME,
        direction="maximize",  # Maximizar mAP50
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True  # Retoma de onde parou se o estudo já existe
    )

    print(f"\n📦 Estudo salvo em: {storage_path}")
    print(f"   (Se o script parar, ele retoma de onde parou automaticamente!)\n")

    # Executa a otimização
    study.optimize(objective, n_trials=NUM_SAMPLES, show_progress_bar=True)

    # ==========================================================================
    # 3. RESULTADOS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("=" * 60)

    best_trial = study.best_trial

    print(f"\nMelhor mAP50: {best_trial.value:.4f}")
    print(f"Trial número: {best_trial.number}")
    print("\nMelhores Hiperparâmetros encontrados:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # ==========================================================================
    # 4. GRÁFICOS
    # ==========================================================================
    try:
        plt.figure(figsize=(14, 5))

        # --- Gráfico 1: Histórico da Otimização ---
        plt.subplot(1, 2, 1)
        trials_values = [t.value for t in study.trials if t.value is not None]
        trials_numbers = [t.number for t in study.trials if t.value is not None]
        best_so_far = np.maximum.accumulate(trials_values)

        plt.plot(trials_numbers, trials_values, marker='o', linestyle='-', color='b', alpha=0.7, label='mAP50 por Trial')
        plt.plot(trials_numbers, best_so_far, linestyle='--', color='r', linewidth=2, label='Melhor mAP50 (Acumulado)')
        plt.title(f'Histórico da Otimização ({len(trials_values)} trials)')
        plt.xlabel('Trial')
        plt.ylabel('mAP50')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --- Gráfico 2: Importância dos Hiperparâmetros ---
        plt.subplot(1, 2, 2)
        try:
            importances = optuna.importance.get_param_importances(study)
            top_params = dict(list(importances.items())[:10])
            plt.barh(list(top_params.keys()), list(top_params.values()), color='steelblue')
            plt.title('Top 10 - Importância dos Hiperparâmetros')
            plt.xlabel('Importância')
            plt.gca().invert_yaxis()
        except Exception as e:
            plt.text(0.5, 0.5, f'Não foi possível calcular\nimportância: {e}',
                     ha='center', va='center', transform=plt.gca().transAxes)

        plot_path = os.path.join(exp_dir, "optimization_history.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n✅ Gráfico salvo em: {plot_path}")

    except Exception as e:
        print(f"\n⚠️ Não foi possível gerar os gráficos: {e}")

    # ==========================================================================
    # 5. SALVAR MELHORES HIPERPARÂMETROS
    # ==========================================================================
    output_yaml = os.path.join(exp_dir, "best_hyperparameters.yaml")

    try:
        clean_params = {}
        for k, v in best_trial.params.items():
            if isinstance(v, (np.generic, np.ndarray)):
                clean_params[k] = v.item()
            else:
                clean_params[k] = v

        with open(output_yaml, 'w') as f:
            yaml.dump(clean_params, f, default_flow_style=False)

        print(f"✅ Melhores hiperparâmetros salvos em: {output_yaml}")
        print(f"\nPara treinar o modelo final com esses parâmetros:")
        print(f"  yolo train model={MODEL_WEIGHTS} data={DATASET_YAML} epochs=300 cfg='{output_yaml}'")

    except Exception as e:
        print(f"\n❌ Erro ao salvar YAML: {e}")
        backup_path = os.path.join(exp_dir, "best_hyperparameters_backup.json")
        with open(backup_path, 'w') as f:
            json.dump(best_trial.params, f, indent=2)
        print(f"  Backup salvo em: {backup_path}")

    # ==========================================================================
    # 6. EXPORTAR TODOS OS TRIALS PARA CSV
    # ==========================================================================
    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(exp_dir, "all_trials.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Todos os trials exportados em: {csv_path}")
    except Exception as e:
        print(f"⚠️ Não foi possível exportar CSV: {e}")


if __name__ == "__main__":
    main()