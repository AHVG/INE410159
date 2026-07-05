"""Orquestração do processamento paralelo dos tiles e gerência de checkpoints.

Cada tile é processado por um worker top-level (`processar_tile`) num
`multiprocessing.Pool`. O progresso é persistido periodicamente para que uma
run interrompida possa ser retomada de onde parou.
"""

import json
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Any, Optional

import cv2
import numpy as np

from .deteccao import detectar_cronometrado, salvar_figuras_etapas

CHECKPOINT_INTERVAL: int = 50


# ── Checkpoint ────────────────────────────────────────────────────────────────

def salvar_checkpoint(
    path: str,
    run_id: str,
    resultados: list[dict[str, Any]],
    todas_areas: list[float],
    tiles_pretos: int,
    pixel_size: float,
) -> None:
    """Grava o estado parcial da run em JSON para permitir retomada."""
    with open(path, "w") as f:
        json.dump({
            "run_id":       run_id,
            "pixel_size":   pixel_size,
            "tiles_pretos": tiles_pretos,
            "todas_areas":  todas_areas,
            "resultados":   resultados,
        }, f)


def carregar_checkpoint(path: str) -> Optional[dict[str, Any]]:
    """Carrega o checkpoint mais recente, ou ``None`` se não existir."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arquivar_checkpoint(checkpoint_path: str, checkpoint_dir: str, run_id: str) -> None:
    """Move o checkpoint da run concluída para o histórico em `checkpoint_dir`."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    dest = os.path.join(checkpoint_dir, f"run_{run_id}.json")
    os.rename(checkpoint_path, dest)
    print(f"  [OK] Checkpoint arquivado: checkpoints/run_{run_id}.json")


# ── Worker (top-level para multiprocessing) ───────────────────────────────────

def processar_tile(
    args: tuple[dict[str, Any], float, str, str],
) -> Optional[dict[str, Any] | str]:
    """Processa um único tile (worker do Pool).

    Grava todos os artefatos do tile numa pasta própria ``saida_dir/<base>/``:
    o JSON de detecções e, para tiles não pretos, o grid e as etapas do pipeline.

    Args:
        args: Tupla ``(info, pixel_size, tiles_dir, saida_dir)``, onde `info` é a
            entrada do tile em ``metadados_tiles.json``.

    Returns:
        ``None`` se o arquivo não pôde ser lido, a string ``"preto"`` se o tile
        está fora da área mapeada (quase todo preto), ou um dicionário com as
        métricas do tile (tempo, detecções, áreas, memória de pico).
    """
    info, pixel_size, tiles_dir, saida_dir = args
    base = os.path.splitext(info["arquivo"])[0]
    tile_dir = os.path.join(saida_dir, base)
    os.makedirs(tile_dir, exist_ok=True)
    json_path = os.path.join(tile_dir, base + ".json")
    img_bgr = cv2.imread(os.path.join(tiles_dir, info["arquivo"]))
    if img_bgr is None:
        return None
    if np.mean(img_bgr) < 15:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "tile_id": info["tile_id"],
                "arquivo": info["arquivo"],
                "ignorado": True,
                "motivo": "tile preto",
                "embaubas": [],
                "n_deteccoes": 0,
            }, f, indent=2, ensure_ascii=False)
        return "preto"
    r, tempo, memoria_mb = detectar_cronometrado(img_bgr)
    salvar_figuras_etapas(r, info["arquivo"], saida_dir)
    area_total_px = sum(r["areas_px"])
    embaubas = []
    for c in r["contornos"]:
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)
        embaubas.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": round(float(cv2.contourArea(c)), 1),
            "poligono": hull.reshape(-1, 2).tolist(),
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "tile_id": info["tile_id"],
            "arquivo": info["arquivo"],
            "ignorado": False,
            "embaubas": embaubas,
            "n_candidatos": r["n_candidatos"],
            "n_deteccoes": r["count"],
            "area_total_px": area_total_px,
            "area_total_m2": area_total_px * pixel_size ** 2,
            "coverage_pct": r["coverage_pct"],
            "tempo_s": tempo,
            "memoria_mb": memoria_mb,
        }, f, indent=2, ensure_ascii=False)
    return {
        "tile_id":       info["tile_id"],
        "arquivo":       info["arquivo"],
        "tempo_s":       tempo,
        "deteccoes":     r["count"],
        "area_total_px": area_total_px,
        "area_total_m2": area_total_px * pixel_size ** 2,
        "coverage_pct":  r["coverage_pct"],
        "memoria_mb":    memoria_mb,
        "areas_px":      r["areas_px"],
    }


# ── Processamento paralelo ────────────────────────────────────────────────────

def processar_todos_tiles(
    tiles_dir: str,
    output_dir: str,
) -> tuple[list[dict[str, Any]], list[float], float, float, int]:
    """Processa todos os tiles em paralelo, com checkpoint e retomada.

    Lê ``metadados_tiles.json``, pula os tiles já feitos (se houver checkpoint),
    distribui os pendentes entre os workers e salva o progresso a cada
    `CHECKPOINT_INTERVAL` tiles. Cada tile grava seus artefatos (JSON + figuras)
    em ``output_dir/tiles/<base>/``. Ao final, arquiva o checkpoint da run.

    Args:
        tiles_dir: Diretório com os tiles e o JSON de metadados.
        output_dir: Diretório raiz de saída (contém `tiles/` e `checkpoints/`).

    Returns:
        Tupla ``(resultados, todas_areas, pixel_size, t_total, tiles_pretos)``.
    """
    checkpoint_dir    = os.path.join(output_dir, "checkpoints")
    checkpoint_latest = os.path.join(output_dir, "checkpoint_latest.json")
    saida_dir         = os.path.join(output_dir, "tiles")
    os.makedirs(saida_dir, exist_ok=True)

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
    n_workers = min(6, max(1, cpu_count() // 2))
    print(f"[INFO] Processando {len(pendentes)}/{total_tiles} tiles com {n_workers} workers...")

    args = [(info, pixel_size, tiles_dir, saida_dir) for info in pendentes]
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
    salvar_checkpoint(checkpoint_latest, run_id, resultados, todas_areas, tiles_pretos, pixel_size)
    arquivar_checkpoint(checkpoint_latest, checkpoint_dir, run_id)
    return resultados, todas_areas, pixel_size, t_total, tiles_pretos
