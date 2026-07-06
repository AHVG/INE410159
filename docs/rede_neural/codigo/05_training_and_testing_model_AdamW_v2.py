"""
05_training_and_testing_model_AdamW_v2.py
==========================================
Retreinamento do YOLOv8x com labels corrigidos (val + test).
Mesmos hiperparâmetros do AdamW otimizado, agora com labels
de cecropias não anotadas adicionados ao dataset.

Autor: Gabriel A. Ferreira Gualda
"""

from ultralytics import YOLO
import torch


# ==============================================================================
# TREINO COM LABELS CORRIGIDOS
# ==============================================================================
def train():
    print("=" * 60)
    print("RETREINAMENTO — YOLOv8x Detecção de Cecropia")
    print("Optimizer: AdamW | Labels: CORRIGIDOS (val + test)")
    print("=" * 60)

    model = YOLO("yolov8x.pt")

    results = model.train(
        # --- Dataset e configuração geral ---
        data="F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/model_dataset_structure.yaml",
        epochs=500,
        patience=500,
        imgsz=893,
        batch=8,
        device=0,
        workers=2,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        project="F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/model_BEST_val_test",
        name="yolov8x_det_AdamW_corrigidos_BEST_val_test",
        exist_ok=True,

        # --- Optimizer AdamW ---
        optimizer="AdamW",

        # --- Melhores hiperparâmetros — Otimização Bayesiana AdamW ---
        freeze=0,
        lr0=0.0041234635967467795,
        lrf=0.0999262262353128,
        momentum=0.8952373561904403,
        weight_decay=0.00015250503090270503,
        warmup_epochs=1.864761110373314,
        box=0.05281453646469406,
        cls=2.980785918932106,

        # --- Augmentations ---
        hsv_h=0.08688090209497161,
        hsv_s=0.7602393861815027,
        hsv_v=0.11217468982046158,
        degrees=25.926120699410163,
        translate=0.09338637893387094,
        scale=0.2780995168930698,
        shear=3.2220528673944293,
        perspective=0.000877671077694895,
        flipud=0.37663593792331107,
        fliplr=0.15614645200260535,
        mosaic=0.1457901399842012,
        mixup=0.41989705646374653,
        copy_paste=0.5658223707335815,
    )

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"Resultados salvos em: F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/model_BEST_val_test/yolov8x_det_AdamW_corrigidos_BEST_val_test")

    return results


# ==============================================================================
# TESTE NO CONJUNTO DE TEST (166 imagens — labels corrigidos)
# ==============================================================================
def test():
    print("\n" + "=" * 60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE (labels corrigidos)")
    print("=" * 60)

    model = YOLO(
        "F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/model_BEST_val_test/yolov8x_det_AdamW_corrigidos_BEST_val_test/weights/best.pt"
    )

    metrics = model.val(
        data="F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/model_dataset_structure.yaml",
        split="test",
        imgsz=893,
        batch=8,
        device=0,
        plots=True,
        project="F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/evaluation_BEST_val_test",
        name="test_results_AdamW_labels_corrigidos_val_test",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("RESULTADOS NO CONJUNTO DE TESTE")
    print("=" * 60)
    print(f"  mAP50:      {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95:   {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision:  {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall:     {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
    print("=" * 60)

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Etapa 1: Retreino com labels corrigidos
    train_results = train()

    # Etapa 2: Teste
    test_metrics = test()

    print("\n✅ Pipeline completo! Retreino + Teste finalizados.")
    print("Próximo passo: comparar com modelo anterior e inferência no ortomosaico.")