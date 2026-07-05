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

from .deteccao import detectar_cronometrado, salvar_passo_a_passo

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
    args: tuple[dict[str, Any], float, str, str, str, bool],
) -> Optional[dict[str, Any] | str]:
    """Processa um único tile (worker do Pool).

    Args:
        args: Tupla ``(info, pixel_size, tiles_dir, passo_a_passo_dir,
            json_dir, gerar_passo_a_passo)``, onde `info` é a entrada do tile em
            ``metadados_tiles.json``.

    Returns:
        ``None`` se o arquivo não pôde ser lido, a string ``"preto"`` se o tile
        está fora da área mapeada (quase todo preto), ou um dicionário com as
        métricas do tile (tempo, detecções, áreas, memória de pico).
    """
    info, pixel_size, tiles_dir, passo_a_passo_dir, json_dir, gerar_passo_a_passo = args
    nome_json = os.path.splitext(info["arquivo"])[0] + ".json"
    json_path = os.path.join(json_dir, nome_json)
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
    if gerar_passo_a_passo:
        salvar_passo_a_passo(r, info["arquivo"], passo_a_passo_dir)
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
            "passo_a_passo": gerar_passo_a_passo,
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


def _passo_a_passo_sincronizado(path: str) -> bool:
    if not os.path.exists(path):
        return False
    img = cv2.imread(path)
    return img is not None and img.shape[:2] == (2094, 1920)


def _regenerar_passo_tile(info: dict[str, Any], tiles_dir: str, passo_a_passo_dir: str) -> str:
    out = os.path.join(passo_a_passo_dir, info["arquivo"].replace(".jpg", ".png"))
    if _passo_a_passo_sincronizado(out):
        return "sincronizado"

    img_bgr = cv2.imread(os.path.join(tiles_dir, info["arquivo"]))
    if img_bgr is None:
        return "erro"
    if np.mean(img_bgr) < 15:
        return "preto"
    r, _, _ = detectar_cronometrado(img_bgr)
    salvar_passo_a_passo(r, info["arquivo"], passo_a_passo_dir)
    return "ok"


def regenerar_passo_a_passo_todos(
    tiles_dir: str,
    passo_a_passo_dir: str,
) -> tuple[int, int, int]:
    """Regenera as figuras de passo a passo para todos os tiles não pretos."""
    os.makedirs(passo_a_passo_dir, exist_ok=True)
    with open(os.path.join(tiles_dir, "metadados_tiles.json")) as f:
        meta = json.load(f)

    ok = sincronizados = pretos = erros = 0
    total = len(meta["tiles"])
    print(f"[INFO] Sincronizando passo a passo de {total} tiles...")

    for i, info in enumerate(meta["tiles"], 1):
        status = _regenerar_passo_tile(info, tiles_dir, passo_a_passo_dir)
        if status == "ok":
            ok += 1
        elif status == "sincronizado":
            sincronizados += 1
        elif status == "preto":
            pretos += 1
        else:
            erros += 1
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} verificados ({ok} regenerados, {sincronizados} já sincronizados)...")

    print(f"[INFO] Passo a passo sincronizado: {ok} regenerados, "
          f"{sincronizados} já estavam certos, {pretos} tiles pretos ignorados, {erros} erros.")
    return ok + sincronizados, pretos, erros


def selecionar_tiles_passo_a_passo(
    tiles: list[dict[str, Any]],
    tiles_dir: str,
    intervalo: int,
) -> Optional[set[int]]:
    """Seleciona cada N-ésimo tile não preto para salvar passo a passo."""
    if intervalo <= 1:
        return None

    selecionados: set[int] = set()
    validos = 0
    for info in tiles:
        img_bgr = cv2.imread(os.path.join(tiles_dir, info["arquivo"]))
        if img_bgr is None or np.mean(img_bgr) < 15:
            continue
        validos += 1
        if validos % intervalo == 0:
            selecionados.add(info["tile_id"])

    print(f"[INFO] Passo a passo: {len(selecionados)}/{validos} tiles válidos (1 a cada {intervalo})")
    return selecionados


# ── Processamento paralelo ────────────────────────────────────────────────────

def processar_todos_tiles(
    tiles_dir: str,
    passo_a_passo_dir: str,
    output_dir: str,
    passo_a_passo_cada: int = 1,
) -> tuple[list[dict[str, Any]], list[float], float, float, int]:
    """Processa todos os tiles em paralelo, com checkpoint e retomada.

    Lê ``metadados_tiles.json``, pula os tiles já feitos (se houver checkpoint),
    distribui os pendentes entre os workers e salva o progresso a cada
    `CHECKPOINT_INTERVAL` tiles. Ao final, arquiva o checkpoint da run.

    Args:
        tiles_dir: Diretório com os tiles e o JSON de metadados.
        passo_a_passo_dir: Diretório de saída das figuras de passo a passo.
        output_dir: Diretório raiz de saída (contém `checkpoints/`).
        passo_a_passo_cada: intervalo de tiles não pretos para salvar a figura
            de passo a passo. Use 1 para salvar todos.

    Returns:
        Tupla ``(resultados, todas_areas, pixel_size, t_total, tiles_pretos)``.
    """
    checkpoint_dir    = os.path.join(output_dir, "checkpoints")
    checkpoint_latest = os.path.join(output_dir, "checkpoint_latest.json")
    json_dir          = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

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

    tiles_com_passo = selecionar_tiles_passo_a_passo(
        meta["tiles"], tiles_dir, passo_a_passo_cada
    )
    args = [
        (
            info,
            pixel_size,
            tiles_dir,
            passo_a_passo_dir,
            json_dir,
            tiles_com_passo is None or info["tile_id"] in tiles_com_passo,
        )
        for info in pendentes
    ]
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
