"""
09_encontrar_nao_anotadas_val.py
==================================
Encontra as detecções do YOLO no conjunto de VALIDAÇÃO que NÃO têm
label correspondente (possíveis cecropias não anotadas).

Mesmo processo do script 09 para test, agora para val.

Autor: Gabriel A. Ferreira Gualda
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

MODEL_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/model_v1/yolov8x_det_final_AdamW/weights/best.pt"
VAL_IMAGES_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/images/val"
VAL_LABELS_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/val"
OUTPUT_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/nao_anotadas_val"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMGSZ = 896


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def calcular_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersecao = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    uniao = area1 + area2 - intersecao

    if uniao <= 0:
        return 0.0
    return intersecao / uniao


def parse_label_polygon(label_line, img_w, img_h):
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None

    coords = [float(v) for v in parts[1:]]
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]

    if len(xs) == 0 or len(ys) == 0:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]


def xyxy_to_label_polygon(bbox, img_w, img_h, class_id=0):
    x1 = bbox[0] / img_w
    y1 = bbox[1] / img_h
    x2 = bbox[2] / img_w
    y2 = bbox[3] / img_h

    return (f"{class_id} "
            f"{x1:.6f} {y2:.6f} "
            f"{x1:.6f} {y1:.6f} "
            f"{x2:.6f} {y1:.6f} "
            f"{x2:.6f} {y2:.6f} "
            f"{x1:.6f} {y2:.6f}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================
def main():
    print("=" * 60)
    print("ENCONTRAR CECROPIAS NÃO ANOTADAS NO VAL")
    print("=" * 60)
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Imagens val: {VAL_IMAGES_DIR}")
    print(f"  Labels val: {VAL_LABELS_DIR}")
    print(f"  Confiança mín: {CONF_THRESHOLD}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print(f"  Saída: {OUTPUT_DIR}")
    print("=" * 60)

    img_output_dir = os.path.join(OUTPUT_DIR, "imagens_destacadas")
    labels_output_dir = os.path.join(OUTPUT_DIR, "labels_novos")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # =========================================================================
    # 1. Carregar modelo
    # =========================================================================
    print("\n[1/4] Carregando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print(f"  Modelo carregado!")

    print("\n  Verificando formato dos labels...")
    sample_labels = [f for f in os.listdir(VAL_LABELS_DIR) if f.endswith('.txt')]
    if len(sample_labels) > 0:
        with open(os.path.join(VAL_LABELS_DIR, sample_labels[0])) as f:
            sample_line = f.readline().strip()
            n_values = len(sample_line.split())
            print(f"  Formato detectado: {n_values} valores por linha (polígono com {(n_values-1)//2} pontos)")

    # =========================================================================
    # 2. Listar imagens de val
    # =========================================================================
    print("\n[2/4] Listando imagens de validação...")
    extensoes = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    imagens = sorted([
        f for f in os.listdir(VAL_IMAGES_DIR)
        if f.lower().endswith(extensoes)
    ])
    print(f"  Total de imagens: {len(imagens)}")

    # =========================================================================
    # 3. Rodar inferência e comparar com labels
    # =========================================================================
    print("\n[3/4] Rodando inferência e comparando com labels...")

    total_deteccoes = 0
    total_com_match = 0
    total_sem_match = 0
    total_labels = 0
    imagens_com_nao_anotadas = 0

    csv_path = os.path.join(OUTPUT_DIR, "relatorio_nao_anotadas_val.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["imagem", "conf", "x_min", "y_min", "x_max", "y_max",
                         "largura_px", "altura_px"])

    for img_name in tqdm(imagens, desc="  Processando"):
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(VAL_LABELS_DIR, label_name)

        labels_existentes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    bbox = parse_label_polygon(line, img_w, img_h)
                    if bbox is not None:
                        labels_existentes.append(bbox)

        total_labels += len(labels_existentes)

        results = model(
            img_path,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False,
            save=False,
        )

        deteccoes = results[0].boxes.xyxy.tolist()
        confs = results[0].boxes.conf.tolist()

        total_deteccoes += len(deteccoes)

        nao_anotadas_img = []

        for det_bbox, conf in zip(deteccoes, confs):
            melhor_iou = 0.0
            for label_bbox in labels_existentes:
                iou = calcular_iou(det_bbox, label_bbox)
                melhor_iou = max(melhor_iou, iou)

            if melhor_iou >= IOU_THRESHOLD:
                total_com_match += 1
            else:
                total_sem_match += 1
                nao_anotadas_img.append({
                    "bbox": det_bbox,
                    "conf": conf,
                })

                largura = det_bbox[2] - det_bbox[0]
                altura = det_bbox[3] - det_bbox[1]
                csv_writer.writerow([
                    img_name, round(conf, 3),
                    round(det_bbox[0], 1), round(det_bbox[1], 1),
                    round(det_bbox[2], 1), round(det_bbox[3], 1),
                    round(largura, 1), round(altura, 1),
                ])

        if len(nao_anotadas_img) > 0:
            imagens_com_nao_anotadas += 1

            draw = ImageDraw.Draw(img)

            for label_bbox in labels_existentes:
                draw.rectangle(label_bbox, outline="green", width=2)

            for det in nao_anotadas_img:
                bbox = det["bbox"]
                conf = det["conf"]
                draw.rectangle(bbox, outline="red", width=3)
                text = f"NAO ANOTADA {conf:.2f}"
                draw.text((bbox[0], max(0, bbox[1] - 15)), text, fill="red")

            save_name = os.path.splitext(img_name)[0] + ".jpg"
            img.save(os.path.join(img_output_dir, save_name), quality=95)

            label_save_path = os.path.join(labels_output_dir, label_name)
            with open(label_save_path, "w") as f:
                for det in nao_anotadas_img:
                    yolo_line = xyxy_to_label_polygon(det["bbox"], img_w, img_h)
                    f.write(yolo_line + "\n")

    csv_file.close()

    # =========================================================================
    # 4. Resumo
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTADO — VALIDAÇÃO")
    print("=" * 60)
    print(f"  Total de labels existentes: {total_labels}")
    print(f"  Total de detecções YOLO: {total_deteccoes}")
    print(f"  Com match (label existente): {total_com_match}")
    print(f"  Sem match (possíveis não anotadas): {total_sem_match}")
    print(f"  Imagens com não anotadas: {imagens_com_nao_anotadas}")
    print(f"\n  Saídas:")
    print(f"    Imagens destacadas: {img_output_dir}")
    print(f"      (verde = label existente, vermelho = sem match)")
    print(f"    Labels novos: {labels_output_dir}")
    print(f"    Relatório CSV: {csv_path}")
    print(f"\n  PRÓXIMOS PASSOS:")
    print(f"    1. Abra as imagens destacadas")
    print(f"    2. Delete os .txt que NÃO são cecropia")
    print(f"    3. Rode o merge para val")
    print("=" * 60)


if __name__ == "__main__":
    main()