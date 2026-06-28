import cv2
import numpy as np
import json
import os
import time
import tracemalloc
from datetime import datetime
from multiprocessing import Pool, cpu_count

from deteccao import detectar, salvar_passo_a_passo

CHECKPOINT_INTERVAL = 50


# ── Checkpoint ────────────────────────────────────────────────────────────────

def salvar_checkpoint(path, run_id, resultados, todas_areas, tiles_pretos, pixel_size):
    with open(path, "w") as f:
        json.dump({
            "run_id":       run_id,
            "pixel_size":   pixel_size,
            "tiles_pretos": tiles_pretos,
            "todas_areas":  todas_areas,
            "resultados":   resultados,
        }, f)


def carregar_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arquivar_checkpoint(checkpoint_path, checkpoint_dir, run_id):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dest = os.path.join(checkpoint_dir, f"run_{run_id}.json")
    os.rename(checkpoint_path, dest)
    print(f"  [OK] Checkpoint arquivado: checkpoints/run_{run_id}.json")


# ── Worker (top-level para multiprocessing) ───────────────────────────────────

def processar_tile(args):
    info, pixel_size, tiles_dir, passo_a_passo_dir = args
    img_bgr = cv2.imread(os.path.join(tiles_dir, info["arquivo"]))
    if img_bgr is None:
        return None
    if np.mean(img_bgr) < 15:
        return "preto"
    tracemalloc.start()
    t0 = time.time()
    r  = detectar(img_bgr)
    tempo = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    salvar_passo_a_passo(r, info["arquivo"], passo_a_passo_dir)
    area_total_px = sum(r["areas_px"])
    return {
        "tile_id":       info["tile_id"],
        "arquivo":       info["arquivo"],
        "tempo_s":       tempo,
        "deteccoes":     r["count"],
        "area_total_px": area_total_px,
        "area_total_m2": area_total_px * pixel_size ** 2,
        "coverage_pct":  r["coverage_pct"],
        "memoria_mb":    peak_bytes / 1024 / 1024,
        "areas_px":      r["areas_px"],
    }


# ── Processamento paralelo ────────────────────────────────────────────────────

def processar_todos_tiles(tiles_dir, passo_a_passo_dir, output_dir):
    checkpoint_dir    = os.path.join(output_dir, "checkpoints")
    checkpoint_latest = os.path.join(output_dir, "checkpoint_latest.json")

    with open(os.path.join(tiles_dir, "metadados_tiles.json")) as f:
        meta = json.load(f)

    pixel_size  = meta["transform_global"][0]
    total_tiles = meta["total_tiles"]

    checkpoint = carregar_checkpoint(checkpoint_latest)
    if checkpoint:
        run_id       = checkpoint["run_id"]
        resultados   = checkpoint["resultados"]
        todas_areas  = checkpoint["todas_areas"]
        tiles_pretos = checkpoint["tiles_pretos"]
        ja_feitos    = {r["tile_id"] for r in resultados}
        print(f"\n[INFO] Retomando run {run_id} ({len(ja_feitos)} tiles já processados)...")
    else:
        run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
        resultados, todas_areas = [], []
        tiles_pretos = 0
        ja_feitos    = set()
        print(f"\n[INFO] Nova run: {run_id}")

    pendentes = [info for info in meta["tiles"] if info["tile_id"] not in ja_feitos]
    n_workers = max(1, cpu_count() // 2)
    print(f"[INFO] Processando {len(pendentes)}/{total_tiles} tiles com {n_workers} workers...")

    args = [(info, pixel_size, tiles_dir, passo_a_passo_dir) for info in pendentes]
    t0_global = time.time()

    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(processar_tile, args, chunksize=4), 1):
            if res is None:
                pass
            elif res == "preto":
                tiles_pretos += 1
            elif isinstance(res, dict):
                todas_areas.extend(res.pop("areas_px"))
                resultados.append(res)
            if i % CHECKPOINT_INTERVAL == 0:
                salvar_checkpoint(checkpoint_latest, run_id, resultados,
                                  todas_areas, tiles_pretos, pixel_size)
            if i % 100 == 0 or i == len(pendentes):
                print(f"  {i}/{len(pendentes)} tiles processados...")

    t_total = time.time() - t0_global
    print(f"[INFO] Concluído em {t_total:.1f}s ({tiles_pretos} tiles pretos ignorados)")
    arquivar_checkpoint(checkpoint_latest, checkpoint_dir, run_id)
    return resultados, todas_areas, pixel_size, t_total, tiles_pretos
