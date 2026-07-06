"""
16_teste_sistematico_sam3_bbox_cropsize.py
=============================================
Teste sistemático SAM3 bbox prompt em diferentes tamanhos de recorte.
Objetivo: encontrar o crop size que maximiza o número de polígonos válidos.

Testa: 80, 100, 120, 140, 160, 200cm
Modelo: SAM3 apenas
Seleção: central (centralidade² × score)
Filtro: MIN_POLYGON_AREA_M2

Autor: Gabriel A. Ferreira Gualda
"""

import os, sys, logging, json, time, gc
sys.stdout.reconfigure(line_buffering=True)

# ══════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════
CROP_SIZES = ["300cm", "350cm"]
MIN_POLYGON_AREA_M2 = 0.30
SAM3_CONFIDENCE = 0.25

SAM3_REPO = r"F:/MESTRADO_Bilada/Backup_Mestrado_V1/02_CODE/sam3"
SAM3_CHECKPOINT = r"C:\Users\User\.cache\huggingface\hub\models--facebook--sam3\snapshots\3c879f39826c281e95690f02c7821c4de09afae7\sam3.pt"
SAM3_RESOLUTION = 1008

BASE_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/04_IMAGES"
OUTPUT_BASE = os.path.join(BASE_DIR, "teste_sistematico_sam3_bbox")

MASK_KEEP_LARGEST_BLOB = True
MASK_BREAK_TENDRIL_PX = 3
MORPH_CLOSE_KERNEL = 11
MORPH_OPEN_KERNEL = 5
MIN_COMPONENT_AREA = 50
# ══════════════════════════════════════════════════════════════

import numpy as np
import cv2
import torch
from PIL import Image
import csv

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════
#  CARREGAR SAM3 ANTES DE RASTERIO
# ══════════════════════════════════════════════════════════════
if SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)

print("  Carregando SAM3...", flush=True)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_sam3_model = build_sam3_image_model(
    checkpoint_path=SAM3_CHECKPOINT, load_from_HF=False,
    device=DEVICE, eval_mode=True, enable_segmentation=True)
_sam3_processor = Sam3Processor(
    model=_sam3_model, resolution=SAM3_RESOLUTION,
    device=DEVICE, confidence_threshold=SAM3_CONFIDENCE)
print("  SAM3 OK!", flush=True)

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
print("  Geo imports OK!\n", flush=True)


# ══════════════════════════════════════════════════════════════
#  SELEÇÃO DE MÁSCARA — CENTRAL
# ══════════════════════════════════════════════════════════════

def select_mask_central(masks, scores, img_w, img_h):
    if len(masks) == 0:
        return None, 0.0
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    max_d = ((img_w / 2)**2 + (img_h / 2)**2)**0.5
    best_idx, best_val = 0, -1
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        d = ((np.mean(xs) - cx_img)**2 + (np.mean(ys) - cy_img)**2)**0.5 / max_d
        combined = float(scores[i]) * (max(0.0, 1.0 - d) ** 2.0)
        if combined > best_val:
            best_val = combined
            best_idx = i
    return masks[best_idx], float(scores[best_idx])


# ══════════════════════════════════════════════════════════════
#  SEGMENTAÇÃO SAM3
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sam3_bbox(pil_img, bbox_xyxy):
    try:
        W, H = pil_img.width, pil_img.height
        x_min, y_min, x_max, y_max = bbox_xyxy
        cx = ((x_min + x_max) / 2.0) / W
        cy = ((y_min + y_max) / 2.0) / H
        bw = (x_max - x_min) / W
        bh = (y_max - y_min) / H

        state = _sam3_processor.set_image(pil_img)
        state = _sam3_processor.add_geometric_prompt(
            box=[cx, cy, bw, bh], label=True, state=state)

        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None or masks.shape[0] == 0:
            return None, 0.0

        masks_np = masks.squeeze(1).cpu().numpy()
        scores_np = scores.cpu().numpy()
        mask_sel, score = select_mask_central(masks_np, scores_np, W, H)
        if mask_sel is None:
            return None, 0.0
        return (mask_sel > 0.5).astype(np.uint8) * 255, score
    except:
        return None, 0.0
    finally:
        try:
            del state
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# ══════════════════════════════════════════════════════════════
#  MASK CLEANUP
# ══════════════════════════════════════════════════════════════

def keep_largest_cc(b, min_area=0):
    bin01 = (b > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if n <= 1:
        return b
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(areas >= min_area)[0]
    if len(valid) == 0:
        valid = np.arange(len(areas))
    largest = int(valid[np.argmax(areas[valid])]) + 1
    out = np.zeros_like(bin01, dtype=np.uint8)
    out[labels == largest] = 255
    return out


def fill_holes(b):
    h, w = b.shape
    ext = b.copy()
    cv2.floodFill(ext, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(b, cv2.bitwise_not(ext))


def clean_mask(b):
    out = b.copy()
    if MASK_KEEP_LARGEST_BLOB and MASK_BREAK_TENDRIL_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (MASK_BREAK_TENDRIL_PX*2+1, MASK_BREAK_TENDRIL_PX*2+1))
        e = cv2.erode(out, k, iterations=1)
        if e.max() > 0:
            e = keep_largest_cc(e, MIN_COMPONENT_AREA)
            d = cv2.dilate(e, k, iterations=1)
            out = cv2.bitwise_and(d, b)
    if MASK_KEEP_LARGEST_BLOB:
        out = keep_largest_cc(out, MIN_COMPONENT_AREA)
    kc = MORPH_CLOSE_KERNEL | 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    ko = MORPH_OPEN_KERNEL | 1
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    if MASK_KEEP_LARGEST_BLOB:
        out = keep_largest_cc(out, MIN_COMPONENT_AREA)
    return fill_holes(out)


def mask_to_polygon(mask_bool, transform):
    u8 = mask_bool.astype(np.uint8)
    polys = [shape(g) for g, v in shapes(u8, mask=(u8 == 1), transform=transform) if v == 1]
    if not polys:
        return None
    return max(polys, key=lambda p: p.area)


# ══════════════════════════════════════════════════════════════
#  VISUALIZAÇÃO — SÓ MOSTRA SE PASSOU NO FILTRO
# ══════════════════════════════════════════════════════════════

def save_visual(rgb, mask_bool, score, bbox, name, vis_dir, passed_filter):
    H, W = rgb.shape[:2]

    orig = rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(orig, "Original + bbox", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    overlay = rgb.copy()
    if passed_filter and mask_bool is not None and mask_bool.any():
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * 0.55
            + np.array([220, 50, 50], np.float32) * 0.45).astype(np.uint8)
        c, _ = cv2.findContours(mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, c, -1, (255, 255, 255), 2)
        cv2.putText(overlay, f"SAM3 | {score:.3f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    else:
        cv2.putText(overlay, "SAM3 | SEM MASCARA", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    sp = np.ones((H, 6, 3), np.uint8) * 40
    panel = np.concatenate([orig, sp, overlay], axis=1)
    cv2.imwrite(os.path.join(vis_dir, f"sam3_{name}.png"), panel[:, :, ::-1])


# ══════════════════════════════════════════════════════════════
#  PROCESSAR UM CROP SIZE
# ══════════════════════════════════════════════════════════════

def process_crop_size(crop_size):
    input_folder = os.path.join(BASE_DIR, f"recortes_bbox_orto_{crop_size}", "tif")
    metadata_path = os.path.join(BASE_DIR, f"recortes_bbox_orto_{crop_size}", "metadata.json")
    output_dir = os.path.join(OUTPUT_BASE, crop_size)
    vis_dir = os.path.join(output_dir, "visualizacao")
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.isdir(input_folder):
        print(f"  [SKIP] Pasta nao encontrada: {input_folder}", flush=True)
        return None

    # Metadados
    meta_dict = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r") as f:
            meta_dict = {m["filename"]: m for m in json.load(f)}

    tifs = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))])
    total = len(tifs)

    if total == 0:
        print(f"  [SKIP] Nenhum TIF em {input_folder}", flush=True)
        return None

    results = []
    sucesso = 0
    skip = 0
    small = 0
    falha = 0
    crs = None
    areas = []
    scores_list = []
    t0 = time.time()

    for idx, tname in enumerate(tifs):
        bname = os.path.splitext(tname)[0]
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    [{idx+1}/{total}] ok={sucesso} skip={skip} small={small}", flush=True)

        try:
            with rasterio.open(os.path.join(input_folder, tname)) as ds:
                gt = ds.transform
                crs = ds.crs
                raster = ds.read()

            rgb = np.moveaxis(raster[:3], 0, -1)
            pil = Image.fromarray(rgb).convert("RGB")
            oh, ow = rgb.shape[:2]

            meta = meta_dict.get(bname, {})
            bbox = meta.get("bbox_in_crop",
                            [int(ow*0.1), int(oh*0.1), int(ow*0.9), int(oh*0.9)])

            m_raw, sc = run_sam3_bbox(pil, bbox)
            m_bool = None
            poly = None
            passed = False

            if m_raw is not None:
                if m_raw.shape != (oh, ow):
                    m_raw = cv2.resize(m_raw, (ow, oh), interpolation=cv2.INTER_NEAREST)
                m_bool = clean_mask(m_raw) > 0
                if m_bool.sum() > 0:
                    poly = mask_to_polygon(m_bool, gt)

            if poly is not None and not poly.is_empty:
                if poly.area < MIN_POLYGON_AREA_M2:
                    small += 1
                else:
                    passed = True
                    sucesso += 1
                    areas.append(poly.area)
                    scores_list.append(sc)
                    results.append({
                        "geometry": poly,
                        "id": meta.get("id", 0),
                        "conf_yolo": meta.get("conf", 0.0),
                        "area_m2": round(poly.area, 2),
                        "sam3_score": round(sc, 4),
                        "filename": tname,
                    })
            else:
                skip += 1

            # Visualização — só mostra se passou
            save_visual(rgb, m_bool, sc, bbox, bname, vis_dir, passed)

        except Exception as e:
            falha += 1
            if falha <= 3:
                print(f"    [ERRO] {tname}: {e}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    dt = time.time() - t0

    # Exportar GPKG
    if results and crs:
        gdf = gpd.GeoDataFrame(results, crs=crs)
        gpkg_path = os.path.join(output_dir, f"cecropia_SAM3_bbox_{crop_size}.gpkg")
        gdf.to_file(gpkg_path, driver="GPKG")

    taxa = (sucesso / total * 100) if total > 0 else 0
    area_m = np.mean(areas) if areas else 0
    area_min = np.min(areas) if areas else 0
    area_max = np.max(areas) if areas else 0
    area_med = np.median(areas) if areas else 0
    score_m = np.mean(scores_list) if scores_list else 0

    return {
        "crop_size": crop_size,
        "total": total,
        "sucesso": sucesso,
        "skip": skip,
        "small": small,
        "falha": falha,
        "taxa": round(taxa, 1),
        "area_media": round(area_m, 2),
        "area_min": round(area_min, 2),
        "area_max": round(area_max, 2),
        "area_mediana": round(area_med, 2),
        "score_medio": round(score_m, 4),
        "tempo_s": round(dt, 0),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("  TESTE SISTEMATICO SAM3 BBOX — DIFERENTES CROP SIZES", flush=True)
    print(f"  Crop sizes: {', '.join(CROP_SIZES)}", flush=True)
    print(f"  Selecao: central | Area min: {MIN_POLYGON_AREA_M2} m2", flush=True)
    print(f"  SAM3 confidence: {SAM3_CONFIDENCE}", flush=True)
    print(f"  Output: {OUTPUT_BASE}", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_results = []
    t_total = time.time()

    for i, crop_size in enumerate(CROP_SIZES):
        print(f"\n{'='*50}", flush=True)
        print(f"  [{i+1}/{len(CROP_SIZES)}] Processando: {crop_size}", flush=True)
        print(f"{'='*50}", flush=True)

        result = process_crop_size(crop_size)
        if result:
            all_results.append(result)
            print(f"  -> {crop_size}: {result['sucesso']}/{result['total']} "
                  f"({result['taxa']}%) | area={result['area_media']} m2 | "
                  f"score={result['score_medio']}", flush=True)

    dt_total = time.time() - t_total

    # ══════════════════════════════════════════════════════════
    #  TABELA COMPARATIVA FINAL
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  TABELA COMPARATIVA — SAM3 BBOX POR CROP SIZE", flush=True)
    print(f"  Selecao: central | Area min: {MIN_POLYGON_AREA_M2} m2 | Conf: {SAM3_CONFIDENCE}", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Crop':<10} {'Polig.':>8} {'Skip':>6} {'Small':>6} {'Taxa':>8} "
          f"{'Min m2':>8} {'Med m2':>8} {'Avg m2':>8} {'Max m2':>8} {'Score':>8}", flush=True)
    print(f"  {'-'*80}", flush=True)

    best_result = None
    best_taxa = 0

    for r in all_results:
        print(f"  {r['crop_size']:<10} {r['sucesso']:>8} {r['skip']:>6} {r['small']:>6} "
              f"{r['taxa']:>7.1f}% {r['area_min']:>8.2f} {r['area_mediana']:>8.2f} "
              f"{r['area_media']:>8.2f} {r['area_max']:>8.2f} {r['score_medio']:>8.4f}", flush=True)
        if r['taxa'] > best_taxa:
            best_taxa = r['taxa']
            best_result = r

    print(f"  {'-'*80}", flush=True)
    if best_result:
        print(f"\n  MELHOR: {best_result['crop_size']} com {best_result['sucesso']}/273 "
              f"({best_result['taxa']}%)", flush=True)

    print(f"  Tempo total: {dt_total:.0f}s ({dt_total/60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    # Salvar CSV comparativo
    csv_path = os.path.join(OUTPUT_BASE, "comparacao_crop_sizes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop_size", "total", "poligonos_gpkg", "skip", "small", "falha",
                     "taxa_pct", "area_min_m2", "area_mediana_m2", "area_media_m2",
                     "area_max_m2", "score_medio", "tempo_s"])
        for r in all_results:
            w.writerow([r["crop_size"], r["total"], r["sucesso"], r["skip"],
                        r["small"], r["falha"], r["taxa"], r["area_min"],
                        r["area_mediana"], r["area_media"], r["area_max"],
                        r["score_medio"], r["tempo_s"]])
    print(f"  CSV: {csv_path}", flush=True)

    # Salvar resumo texto
    txt_path = os.path.join(OUTPUT_BASE, "resumo.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TESTE SISTEMATICO SAM3 BBOX — CROP SIZES\n")
        f.write(f"Selecao: central | Area min: {MIN_POLYGON_AREA_M2} m2\n")
        f.write(f"SAM3 confidence: {SAM3_CONFIDENCE}\n")
        f.write(f"Total imagens por teste: 273\n\n")
        for r in all_results:
            f.write(f"{r['crop_size']}: {r['sucesso']}/273 ({r['taxa']}%) "
                    f"area min={r['area_min']} med={r['area_mediana']} "
                    f"avg={r['area_media']} max={r['area_max']} m2 "
                    f"score={r['score_medio']}\n")
        if best_result:
            f.write(f"\nMELHOR: {best_result['crop_size']} "
                    f"({best_result['sucesso']}/273 = {best_result['taxa']}%)\n")
    print(f"  Resumo: {txt_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()