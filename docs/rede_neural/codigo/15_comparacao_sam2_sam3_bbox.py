"""
15_comparacao_sam2_sam3_bbox.py
=================================
Comparação SAM2 bbox vs SAM3 bbox nos recortes GeoTIFF.

CONFIGURAÇÕES FÁCEIS DE TROCAR:
  - CROP_SIZE: "20cm", "40cm", "60cm", "80cm"
  - MIN_POLYGON_AREA_M2: área mínima do polígono (ex: 1.7)
  - MASK_SELECTION: "largest" (maior polígono) ou "central" (centralidade × score)

Autor: Gabriel A. Ferreira Gualda
"""

import os, sys, logging, json, traceback, time, gc
sys.stdout.reconfigure(line_buffering=True)

# ══════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES — TROQUE AQUI
# ══════════════════════════════════════════════════════════════
CROP_SIZE = "200cm"              # ← "20cm", "40cm", "60cm", "80cm"
MIN_POLYGON_AREA_M2 = 4.00      # ← área mínima em m² (0 = sem filtro)
MASK_SELECTION = "central"      # ← "largest" (maior polígono) ou "central" (centralidade × score)
# ══════════════════════════════════════════════════════════════

SAM3_REPO = r"F:/MESTRADO_Bilada/Backup_Mestrado_V1/02_CODE/sam3"
SAM3_CHECKPOINT = r"C:\Users\User\.cache\huggingface\hub\models--facebook--sam3\snapshots\3c879f39826c281e95690f02c7821c4de09afae7\sam3.pt"
SAM3_RESOLUTION = 1008
SAM3_CONFIDENCE = 0.40
SAM2_MODEL = "facebook/sam2-hiera-large"

BASE_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/04_IMAGES"
INPUT_FOLDER = os.path.join(BASE_DIR, f"recortes_bbox_orto_{CROP_SIZE}", "tif")
OUTPUT_DIR = os.path.join(BASE_DIR, f"comparacao_sam2_sam3_bbox_{CROP_SIZE}")
METADATA_PATH = os.path.join(BASE_DIR, f"recortes_bbox_orto_{CROP_SIZE}", "metadata.json")

MASK_KEEP_LARGEST_BLOB = True
MASK_BREAK_TENDRIL_PX = 3
MORPH_CLOSE_KERNEL = 11
MORPH_OPEN_KERNEL = 5
MIN_COMPONENT_AREA = 50

import numpy as np
import cv2
import torch
from PIL import Image
import csv

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════
#  CARREGAR MODELOS ANTES DE RASTERIO
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

print("  Carregando SAM2...", flush=True)
from transformers import Sam2Processor, Sam2Model

_sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL)
_sam2_model = Sam2Model.from_pretrained(SAM2_MODEL).to(DEVICE)
_sam2_model.eval()
print("  SAM2 OK!", flush=True)

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
print("  Geo imports OK!\n", flush=True)


# ══════════════════════════════════════════════════════════════
#  SELEÇÃO DE MÁSCARA
# ══════════════════════════════════════════════════════════════

def select_mask_central(masks, scores, img_w, img_h):
    """Seleciona máscara por centralidade² × score."""
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


def select_mask_largest(masks, scores, img_w, img_h):
    """Seleciona máscara com maior área (mais pixels)."""
    if len(masks) == 0:
        return None, 0.0
    best_idx, best_area = 0, -1
    for i, mask in enumerate(masks):
        area = np.sum(mask > 0)
        if area > best_area:
            best_area = area
            best_idx = i
    return masks[best_idx], float(scores[best_idx])


def select_mask(masks, scores, img_w, img_h):
    """Seleciona máscara conforme MASK_SELECTION."""
    if MASK_SELECTION == "largest":
        return select_mask_largest(masks, scores, img_w, img_h)
    else:
        return select_mask_central(masks, scores, img_w, img_h)


# ══════════════════════════════════════════════════════════════
#  SEGMENTAÇÃO
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
        mask_sel, score = select_mask(masks_np, scores_np, W, H)
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


@torch.no_grad()
def run_sam2_bbox(pil_img, bbox_xyxy):
    try:
        inputs = _sam2_processor(
            images=pil_img, input_boxes=[[bbox_xyxy]],
            return_tensors="pt").to(DEVICE)
        outputs = _sam2_model(**inputs)

        pred_masks = outputs.pred_masks.cpu().numpy()[0, 0]
        iou_scores = outputs.iou_scores.cpu().numpy()[0, 0]

        oh, ow = pil_img.height, pil_img.width
        masks_resized = [
            cv2.resize((m > 0).astype(np.uint8) * 255, (ow, oh),
                       interpolation=cv2.INTER_NEAREST) for m in pred_masks]

        mask_sel, score = select_mask(masks_resized, iou_scores, ow, oh)
        if mask_sel is None:
            return None, 0.0
        return mask_sel, score
    except:
        return None, 0.0
    finally:
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
    # Retorna o maior polígono
    return max(polys, key=lambda p: p.area)


# ══════════════════════════════════════════════════════════════
#  VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════

def save_comparison(rgb, mask2, mask3, score2, score3, bbox, name, vis_dir):
    H, W = rgb.shape[:2]

    orig = rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(orig, "Original + bbox", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ov2 = rgb.copy()
    if mask2 is not None and mask2.any():
        ov2[mask2] = (ov2[mask2].astype(np.float32) * 0.55
            + np.array([50, 100, 220], np.float32) * 0.45).astype(np.uint8)
        c2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov2, c2, -1, (255, 255, 255), 2)
    cv2.putText(ov2, f"SAM2 | {score2:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    ov3 = rgb.copy()
    if mask3 is not None and mask3.any():
        ov3[mask3] = (ov3[mask3].astype(np.float32) * 0.55
            + np.array([220, 50, 50], np.float32) * 0.45).astype(np.uint8)
        c3, _ = cv2.findContours(mask3.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov3, c3, -1, (255, 255, 255), 2)
    cv2.putText(ov3, f"SAM3 | {score3:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    sp = np.ones((H, 6, 3), np.uint8) * 40
    panel = np.concatenate([orig, sp, ov2, sp, ov3], axis=1)
    cv2.imwrite(os.path.join(vis_dir, f"comp_{name}.png"), panel[:, :, ::-1])


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print(f"  COMPARACAO SAM2 vs SAM3 bbox — {CROP_SIZE}", flush=True)
    print(f"  Selecao mascara: {MASK_SELECTION}", flush=True)
    print(f"  Area minima: {MIN_POLYGON_AREA_M2} m2", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Input: {INPUT_FOLDER}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    vis_dir = os.path.join(OUTPUT_DIR, "visualizacao")
    os.makedirs(vis_dir, exist_ok=True)

    meta_dict = {}
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta_dict = {m["filename"]: m for m in json.load(f)}
        print(f"  Metadados: {len(meta_dict)}", flush=True)

    tifs = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".tif", ".tiff"))])
    total = len(tifs)
    print(f"  TIFs: {total}\n", flush=True)

    res2, res3 = [], []
    s2_ok = s2_skip = s2_small = 0
    s3_ok = s3_skip = s3_small = 0
    falha = 0
    both_ok = only_sam2 = only_sam3 = neither = 0
    crs = None
    t0 = time.time()

    for idx, tname in enumerate(tifs):
        bname = os.path.splitext(tname)[0]
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] SAM2={s2_ok} SAM3={s3_ok} err={falha}", flush=True)

        try:
            with rasterio.open(os.path.join(INPUT_FOLDER, tname)) as ds:
                gt = ds.transform
                crs = ds.crs
                raster = ds.read()

            rgb = np.moveaxis(raster[:3], 0, -1)
            pil = Image.fromarray(rgb).convert("RGB")
            oh, ow = rgb.shape[:2]

            meta = meta_dict.get(bname, {})
            bbox = meta.get("bbox_in_crop",
                            [int(ow*0.1), int(oh*0.1), int(ow*0.9), int(oh*0.9)])

            # SAM3
            m3_raw, sc3 = run_sam3_bbox(pil, bbox)
            m3_bool = None
            poly3 = None
            if m3_raw is not None:
                if m3_raw.shape != (oh, ow):
                    m3_raw = cv2.resize(m3_raw, (ow, oh), interpolation=cv2.INTER_NEAREST)
                m3_bool = clean_mask(m3_raw) > 0
                if m3_bool.sum() > 0:
                    poly3 = mask_to_polygon(m3_bool, gt)

            got3 = poly3 is not None and not poly3.is_empty
            if got3 and poly3.area < MIN_POLYGON_AREA_M2:
                s3_small += 1
                got3 = False
                poly3 = None
            if got3:
                s3_ok += 1
                res3.append({"geometry": poly3, "id": meta.get("id", 0),
                    "conf_yolo": meta.get("conf", 0.0),
                    "area_m2": round(poly3.area, 2),
                    "sam_score": round(sc3, 4), "filename": tname})
            else:
                s3_skip += 1

            # SAM2
            m2_raw, sc2 = run_sam2_bbox(pil, bbox)
            m2_bool = None
            poly2 = None
            if m2_raw is not None:
                if m2_raw.shape != (oh, ow):
                    m2_raw = cv2.resize(m2_raw, (ow, oh), interpolation=cv2.INTER_NEAREST)
                m2_bool = clean_mask(m2_raw) > 0
                if m2_bool.sum() > 0:
                    poly2 = mask_to_polygon(m2_bool, gt)

            got2 = poly2 is not None and not poly2.is_empty
            if got2 and poly2.area < MIN_POLYGON_AREA_M2:
                s2_small += 1
                got2 = False
                poly2 = None
            if got2:
                s2_ok += 1
                res2.append({"geometry": poly2, "id": meta.get("id", 0),
                    "conf_yolo": meta.get("conf", 0.0),
                    "area_m2": round(poly2.area, 2),
                    "sam_score": round(sc2, 4), "filename": tname})
            else:
                s2_skip += 1

            if got2 and got3:
                both_ok += 1
            elif got2:
                only_sam2 += 1
            elif got3:
                only_sam3 += 1
            else:
                neither += 1

            save_comparison(rgb, m2_bool, m3_bool, sc2, sc3, bbox, bname, vis_dir)

        except Exception as e:
            falha += 1
            print(f"    [ERRO] {tname}: {e}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    dt = time.time() - t0

    # Exportar
    print("\n  Exportando...", flush=True)
    if res2 and crs:
        gdf2 = gpd.GeoDataFrame(res2, crs=crs)
        p2 = os.path.join(OUTPUT_DIR, f"cecropia_SAM2_bbox_{CROP_SIZE}.gpkg")
        gdf2.to_file(p2, driver="GPKG")
        print(f"  SAM2 GPKG: {p2}", flush=True)

    if res3 and crs:
        gdf3 = gpd.GeoDataFrame(res3, crs=crs)
        p3 = os.path.join(OUTPUT_DIR, f"cecropia_SAM3_bbox_{CROP_SIZE}.gpkg")
        gdf3.to_file(p3, driver="GPKG")
        print(f"  SAM3 GPKG: {p3}", flush=True)

    # Métricas
    t2 = (s2_ok / total * 100) if total > 0 else 0
    t3 = (s3_ok / total * 100) if total > 0 else 0
    a2 = np.mean([r["area_m2"] for r in res2]) if res2 else 0
    a3 = np.mean([r["area_m2"] for r in res3]) if res3 else 0
    sc2m = np.mean([r["sam_score"] for r in res2]) if res2 else 0
    sc3m = np.mean([r["sam_score"] for r in res3]) if res3 else 0

    # Relatório
    print("\n" + "=" * 70, flush=True)
    print(f"  RESULTADOS — SAM2 vs SAM3 bbox ({CROP_SIZE})", flush=True)
    print(f"  Selecao: {MASK_SELECTION} | Area min: {MIN_POLYGON_AREA_M2} m2", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metrica':<30} {'SAM2':>15} {'SAM3':>15}", flush=True)
    print(f"  {'-'*60}", flush=True)
    print(f"  {'Total imagens':<30} {total:>15} {total:>15}", flush=True)
    print(f"  {'Sucesso':<30} {s2_ok:>15} {s3_ok:>15}", flush=True)
    print(f"  {'Skip (sem mascara)':<30} {s2_skip:>15} {s3_skip:>15}", flush=True)
    print(f"  {'Removidos (area < min)':<30} {s2_small:>15} {s3_small:>15}", flush=True)
    print(f"  {'Taxa sucesso (%)':<30} {t2:>14.1f}% {t3:>14.1f}%", flush=True)
    print(f"  {'Area media (m2)':<30} {a2:>15.2f} {a3:>15.2f}", flush=True)
    print(f"  {'Score medio':<30} {sc2m:>15.4f} {sc3m:>15.4f}", flush=True)
    print(f"  {'Falhas':<30} {falha:>15} {falha:>15}", flush=True)
    print(f"  {'-'*60}", flush=True)
    print(f"  {'Ambos OK':<30} {both_ok:>15}", flush=True)
    print(f"  {'So SAM2 OK':<30} {only_sam2:>15}", flush=True)
    print(f"  {'So SAM3 OK':<30} {only_sam3:>15}", flush=True)
    print(f"  {'Nenhum OK':<30} {neither:>15}", flush=True)
    print(f"  {'-'*60}", flush=True)
    print(f"  Tempo: {dt:.0f}s ({dt/60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    if t2 > t3:
        print(f"\n  VENCEDOR: SAM2 bbox ({t2:.1f}% vs {t3:.1f}%)", flush=True)
    elif t3 > t2:
        print(f"\n  VENCEDOR: SAM3 bbox ({t3:.1f}% vs {t2:.1f}%)", flush=True)
    else:
        print(f"\n  EMPATE ({t2:.1f}% vs {t3:.1f}%)", flush=True)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, f"comparacao_sam2_sam3_bbox_{CROP_SIZE}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "SAM2_bbox", "SAM3_bbox"])
        w.writerow(["crop_size", CROP_SIZE, CROP_SIZE])
        w.writerow(["mask_selection", MASK_SELECTION, MASK_SELECTION])
        w.writerow(["min_area_m2", MIN_POLYGON_AREA_M2, MIN_POLYGON_AREA_M2])
        w.writerow(["total", total, total])
        w.writerow(["sucesso", s2_ok, s3_ok])
        w.writerow(["skip", s2_skip, s3_skip])
        w.writerow(["removidos_area_min", s2_small, s3_small])
        w.writerow(["taxa_pct", round(t2, 1), round(t3, 1)])
        w.writerow(["area_media_m2", round(a2, 2), round(a3, 2)])
        w.writerow(["score_medio", round(sc2m, 4), round(sc3m, 4)])
        w.writerow(["ambos_ok", both_ok, both_ok])
        w.writerow(["so_este_ok", only_sam2, only_sam3])
        w.writerow(["nenhum_ok", neither, neither])
        w.writerow(["falhas", falha, falha])
        w.writerow(["tempo_s", round(dt, 0), round(dt, 0)])
    print(f"  CSV: {csv_path}", flush=True)
    print(f"  Visualizacoes: {vis_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()