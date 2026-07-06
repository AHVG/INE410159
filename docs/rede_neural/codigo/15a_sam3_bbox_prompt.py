"""
15a_sam3_bbox_prompt.py
=========================
SAM3 com BBOX prompt nos recortes GeoTIFF.
IMPORTANTE: SAM3 carregado ANTES de rasterio/geopandas (conflito de DLLs).

TROQUE CROP_SIZE para testar diferentes tamanhos.
Autor: Gabriel A. Ferreira Gualda
"""

import os, sys, logging, json, traceback, time, gc
sys.stdout.reconfigure(line_buffering=True)

# ══════════════════════════════════════════════════════════════
CROP_SIZE = "40cm"  # ← TROQUE: "20cm", "40cm", "60cm", "80cm"

SAM3_REPO = r"F:/MESTRADO_Bilada/Backup_Mestrado_V1/02_CODE/sam3"
CHECKPOINT = r"C:\Users\User\.cache\huggingface\hub\models--facebook--sam3\snapshots\3c879f39826c281e95690f02c7821c4de09afae7\sam3.pt"
BASE_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/04_IMAGES"
INPUT_FOLDER = os.path.join(BASE_DIR, f"recortes_bbox_orto_{CROP_SIZE}", "tif")
OUTPUT_DIR = os.path.join(BASE_DIR, f"resultado_sam3_bbox_{CROP_SIZE}")
METADATA_PATH = os.path.join(BASE_DIR, f"recortes_bbox_orto_{CROP_SIZE}", "metadata.json")

CONFIDENCE = 0.25
RESOLUTION = 1008
MASK_KEEP_LARGEST_BLOB = True
MASK_BREAK_TENDRIL_PX = 3
MORPH_CLOSE_KERNEL = 11
MORPH_OPEN_KERNEL = 5
MIN_COMPONENT_AREA = 50
# ══════════════════════════════════════════════════════════════

# ORDEM CRÍTICA: SAM3 antes de rasterio/geopandas
if SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)

import numpy as np
import cv2
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── CARREGAR SAM3 PRIMEIRO (antes de rasterio) ───────────────
print("  Carregando SAM3...", flush=True)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_model = build_sam3_image_model(
    checkpoint_path=CHECKPOINT, load_from_HF=False,
    device=DEVICE, eval_mode=True, enable_segmentation=True)
_processor = Sam3Processor(
    model=_model, resolution=RESOLUTION,
    device=DEVICE, confidence_threshold=CONFIDENCE)
print("  SAM3 carregado!", flush=True)

# AGORA importa rasterio/geopandas
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
print("  Geo imports ok!", flush=True)


# ── SAM3 BBOX SEGMENT ────────────────────────────────────────
@torch.no_grad()
def sam3_bbox_segment(pil_img, bbox_xyxy):
    try:
        W, H = pil_img.width, pil_img.height
        x_min, y_min, x_max, y_max = bbox_xyxy
        cx = ((x_min + x_max) / 2.0) / W
        cy = ((y_min + y_max) / 2.0) / H
        bw = (x_max - x_min) / W
        bh = (y_max - y_min) / H

        state = _processor.set_image(pil_img)
        state = _processor.add_geometric_prompt(
            box=[cx, cy, bw, bh], label=True, state=state)

        masks = state.get("masks")
        scores = state.get("scores")

        if masks is None or scores is None or masks.shape[0] == 0:
            return None, 0.0

        masks_np = masks.squeeze(1).cpu().numpy()
        scores_np = scores.cpu().numpy()

        # Seleciona máscara mais central
        center_x, center_y = W / 2.0, H / 2.0
        max_d = ((W / 2)**2 + (H / 2)**2)**0.5
        best_idx, best_val = 0, -1

        for i, mask in enumerate(masks_np):
            ys, xs = np.where(mask > 0.5)
            if len(xs) == 0:
                continue
            d = ((np.mean(xs) - center_x)**2 + (np.mean(ys) - center_y)**2)**0.5 / max_d
            combined = float(scores_np[i]) * (max(0.0, 1.0 - d) ** 2.0)
            if combined > best_val:
                best_val = combined
                best_idx = i

        return (masks_np[best_idx] > 0.5).astype(np.uint8) * 255, float(scores_np[best_idx])

    except Exception as e:
        print(f"    [SAM3 ERRO] {e}", flush=True)
        return None, 0.0
    finally:
        try:
            del state
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# ── MASK CLEANUP ──────────────────────────────────────────────
def keep_largest_cc(binary_255, min_area=0):
    bin01 = (binary_255 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num_labels <= 1:
        return binary_255
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(areas >= min_area)[0]
    if len(valid) == 0:
        valid = np.arange(len(areas))
    largest = int(valid[np.argmax(areas[valid])]) + 1
    out = np.zeros_like(bin01, dtype=np.uint8)
    out[labels == largest] = 255
    return out


def fill_holes(binary_255):
    h, w = binary_255.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    ext = binary_255.copy()
    cv2.floodFill(ext, flood, (0, 0), 255)
    return cv2.bitwise_or(binary_255, cv2.bitwise_not(ext))


def clean_mask(binary_255):
    out = binary_255.copy()
    if MASK_KEEP_LARGEST_BLOB and MASK_BREAK_TENDRIL_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (MASK_BREAK_TENDRIL_PX * 2 + 1, MASK_BREAK_TENDRIL_PX * 2 + 1))
        eroded = cv2.erode(out, k, iterations=1)
        if eroded.max() > 0:
            eroded = keep_largest_cc(eroded, min_area=MIN_COMPONENT_AREA)
            dilated = cv2.dilate(eroded, k, iterations=1)
            out = cv2.bitwise_and(dilated, binary_255)
    if MASK_KEEP_LARGEST_BLOB:
        out = keep_largest_cc(out, min_area=MIN_COMPONENT_AREA)
    kc = MORPH_CLOSE_KERNEL if MORPH_CLOSE_KERNEL % 2 == 1 else MORPH_CLOSE_KERNEL + 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    ko = MORPH_OPEN_KERNEL if MORPH_OPEN_KERNEL % 2 == 1 else MORPH_OPEN_KERNEL + 1
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    if MASK_KEEP_LARGEST_BLOB:
        out = keep_largest_cc(out, min_area=MIN_COMPONENT_AREA)
    return fill_holes(out)


def mask_to_polygon(mask_bool, transform):
    mask_u8 = mask_bool.astype(np.uint8)
    polygons = [shape(geom) for geom, val in
                shapes(mask_u8, mask=(mask_u8 == 1), transform=transform) if val == 1]
    if not polygons:
        return None
    return max(polygons, key=lambda p: p.area)


# ── VISUAL ────────────────────────────────────────────────────
def save_visual(rgb, mask_bool, score, bbox_xyxy, base_name, visual_dir):
    H, W = rgb.shape[:2]
    original = rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original, "bbox YOLO", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    overlay = rgb.copy()
    if mask_bool is not None and mask_bool.any():
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * 0.55
            + np.array([220, 50, 50], dtype=np.float32) * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    cv2.putText(overlay, f"SAM3 bbox | {score:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    binary = np.zeros((H, W, 3), dtype=np.uint8)
    if mask_bool is not None:
        binary[mask_bool] = (255, 255, 255)

    spacer = np.ones((H, 6, 3), dtype=np.uint8) * 40
    panel = np.concatenate([original, spacer, overlay, spacer, binary], axis=1)
    cv2.imwrite(os.path.join(visual_dir, f"sam3bbox_{base_name}.png"), panel[:, :, ::-1])


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print(f"  SAM3 BBOX PROMPT — {CROP_SIZE}", flush=True)
    print(f"  Device: {DEVICE} | Input: {INPUT_FOLDER}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)

    visual_dir = os.path.join(OUTPUT_DIR, "visualizacao")
    os.makedirs(visual_dir, exist_ok=True)

    meta_dict = {}
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta_dict = {m["filename"]: m for m in json.load(f)}
        print(f"  Metadados: {len(meta_dict)}", flush=True)

    tif_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                        if f.lower().endswith((".tif", ".tiff"))])
    total = len(tif_files)
    print(f"  TIFs: {total}\n", flush=True)

    results = []
    sucesso = skip = falha = 0
    crs = None
    t0 = time.time()

    for idx, tif_name in enumerate(tif_files):
        base_name = os.path.splitext(tif_name)[0]
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] ok={sucesso} skip={skip} err={falha}", flush=True)
        try:
            with rasterio.open(os.path.join(INPUT_FOLDER, tif_name)) as ds:
                geo_transform = ds.transform
                crs = ds.crs
                raster = ds.read()

            rgb = np.moveaxis(raster[:3], 0, -1)
            pil_img = Image.fromarray(rgb).convert("RGB")
            oh, ow = rgb.shape[:2]

            meta = meta_dict.get(base_name, {})
            bbox = meta.get("bbox_in_crop",
                            [int(ow * 0.1), int(oh * 0.1), int(ow * 0.9), int(oh * 0.9)])

            mask_raw, score = sam3_bbox_segment(pil_img, bbox)

            if mask_raw is None:
                skip += 1
                continue
            if mask_raw.shape != (oh, ow):
                mask_raw = cv2.resize(mask_raw, (ow, oh), interpolation=cv2.INTER_NEAREST)

            mask_clean = clean_mask(mask_raw)
            mask_bool = mask_clean > 0
            if mask_bool.sum() == 0:
                skip += 1
                continue

            polygon = mask_to_polygon(mask_bool, geo_transform)
            if polygon is None or polygon.is_empty:
                skip += 1
                continue

            sucesso += 1
            results.append({"geometry": polygon, "id": meta.get("id", 0),
                "conf_yolo": meta.get("conf", 0.0), "area_m2": round(polygon.area, 2),
                "sam3_score": round(score, 4), "filename": tif_name})
            save_visual(rgb, mask_bool, score, bbox, base_name, visual_dir)

        except Exception as e:
            falha += 1
            print(f"    [ERRO] {tif_name}: {e}", flush=True)
        torch.cuda.empty_cache()
        gc.collect()

    dt = time.time() - t0

    if results and crs:
        gdf = gpd.GeoDataFrame(results, crs=crs)
        gpkg = os.path.join(OUTPUT_DIR, f"cecropia_SAM3_bbox_{CROP_SIZE}.gpkg")
        gdf.to_file(gpkg, driver="GPKG")
        print(f"\n  GPKG: {gpkg}", flush=True)

    taxa = (sucesso / total * 100) if total > 0 else 0
    area_m = np.mean([r["area_m2"] for r in results]) if results else 0
    score_m = np.mean([r["sam3_score"] for r in results]) if results else 0

    print("\n" + "=" * 60, flush=True)
    print(f"  SAM3 BBOX — {CROP_SIZE} — CONCLUIDO", flush=True)
    print("=" * 60, flush=True)
    print(f"  Total: {total} | Sucesso: {sucesso} | Skip: {skip} | Falha: {falha}", flush=True)
    print(f"  Taxa: {taxa:.1f}% | Area: {area_m:.2f} m2 | Score: {score_m:.4f}", flush=True)
    print(f"  Tempo: {dt:.0f}s ({dt/60:.1f} min)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()