import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time
import tracemalloc
from multiprocessing import Pool, cpu_count

TILES_DIR      = "tiles"
OUTPUT_DIR     = "output"
PASSO_A_PASSO_DIR = os.path.join(OUTPUT_DIR, "passo_a_passo")
TILE_DEMO      = "tiles/tile_0053.jpg"

# ── Parâmetros (espelho de embaubaHSVmask.py) ─────────────────────────────────
HSV_LOWER   = np.array([44, 144, 104])
HSV_UPPER   = np.array([58, 231, 203])
BLUR_KERNEL = (9, 9)
CLOSE_SIZE  = 25
OPEN_SIZE   = 9
AREA_MIN    = 10_000


# ── Pipeline ──────────────────────────────────────────────────────────────────

def detectar(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # suavização
    blurred = cv2.GaussianBlur(img_bgr, BLUR_KERNEL, 0)

    # segmentação por cor HSV
    hsv        = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_bruta = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # morfologia: fecha buracos, remove ruído
    k_close    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    k_open     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
    mask_fecha = cv2.morphologyEx(mask_bruta, cv2.MORPH_CLOSE, k_close)
    mask_limpa = cv2.morphologyEx(mask_fecha, cv2.MORPH_OPEN,  k_open)

    # filtra contornos por área mínima
    todos, _  = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in todos if cv2.contourArea(c) > AREA_MIN]

    # desenha convex hull de cada detecção
    resultado = img_rgb.copy()
    for c in contornos:
        hull = cv2.convexHull(c)
        x, y, _, _ = cv2.boundingRect(hull)
        cv2.drawContours(resultado, [hull], -1, (0, 200, 255), 3)
        cv2.putText(resultado, 'Cecropia', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 3)

    return {
        "img_rgb":      img_rgb,
        "suavizado":    cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
        "hsv":          hsv,
        "mask_bruta":   mask_bruta,
        "mask_fecha":   mask_fecha,
        "mask_limpa":   mask_limpa,
        "contornos":    contornos,
        "resultado":    resultado,
        "count":        len(contornos),
        "areas_px":     [cv2.contourArea(c) for c in contornos],
        "coverage_pct": float(np.count_nonzero(mask_limpa)) / mask_limpa.size * 100,
    }


# ── Passo a passo ─────────────────────────────────────────────────────────────

def _salvar_passo_a_passo(r, nome_tile):
    paineis = [
        (r["img_rgb"],        "1. Original",                   None),
        (r["suavizado"],      "2. Gaussian Blur (9×9)",         None),
        (r["hsv"][:, :, 0],  "3. Canal H (Matiz)",             "hsv"),
        (r["hsv"][:, :, 1],  "4. Canal S (Saturação)",         "gray"),
        (r["hsv"][:, :, 2],  "5. Canal V (Valor/Brilho)",      "gray"),
        (r["mask_bruta"],     "6. Máscara Bruta (inRange)",     "gray"),
        (r["mask_limpa"],     "7. Máscara Limpa (CLOSE+OPEN)",  "gray"),
        (r["resultado"],      f"8. Resultado — {r['count']} detecções", None),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f"Pipeline — Cecropia / Embaúba ({nome_tile})",
                 fontsize=14, fontweight="bold", y=1.01)
    for ax, (img, titulo, cmap) in zip(axs.flat, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=9, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    caminho = os.path.join(PASSO_A_PASSO_DIR, nome_tile.replace(".jpg", ".png"))
    plt.savefig(caminho, dpi=80, bbox_inches="tight")
    plt.close()


# ── Worker (top-level para multiprocessing) ───────────────────────────────────

def _processar_tile(args):
    info, pixel_size = args
    img_bgr = cv2.imread(os.path.join(TILES_DIR, info["arquivo"]))
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
    _salvar_passo_a_passo(r, info["arquivo"])
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


# ── Processamento completo ────────────────────────────────────────────────────

def processar_todos_tiles():
    with open(os.path.join(TILES_DIR, "metadados_tiles.json")) as f:
        meta = json.load(f)

    pixel_size  = meta["transform_global"][0]
    total_tiles = meta["total_tiles"]
    n_workers   = cpu_count()
    print(f"\n[INFO] Processando {total_tiles} tiles com {n_workers} workers...")

    args = [(info, pixel_size) for info in meta["tiles"]]
    resultados, todas_areas = [], []
    tiles_pretos = 0
    t0_global = time.time()

    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_processar_tile, args, chunksize=4), 1):
            if res is None:
                pass
            elif res == "preto":
                tiles_pretos += 1
            elif isinstance(res, dict):
                todas_areas.extend(res.pop("areas_px"))
                resultados.append(res)
            if i % 100 == 0 or i == total_tiles:
                print(f"  {i}/{total_tiles} tiles processados...")

    t_total = time.time() - t0_global
    print(f"[INFO] Concluído em {t_total:.1f}s ({tiles_pretos} tiles pretos ignorados)")
    return resultados, todas_areas, pixel_size, t_total, tiles_pretos


# ── Figuras de estatísticas ───────────────────────────────────────────────────

def gerar_figuras(resultados, todas_areas):
    tempos     = [r["tempo_s"]      for r in resultados]
    deteccoes  = [r["deteccoes"]    for r in resultados]
    coberturas = [r["coverage_pct"] for r in resultados]

    _hist(deteccoes, range(0, max(deteccoes) + 2), "#2a9d8f",
          "Número de detecções por tile", "Frequência (tiles)",
          "Distribuição de Detecções por Tile", "hist_deteccoes_por_tile.png")

    if todas_areas:
        _hist(todas_areas, 40, "#e76f51",
              "Área da detecção (px²)", "Frequência",
              "Distribuição das Áreas de Detecção", "hist_areas_deteccao.png")

    top10  = sorted(resultados, key=lambda x: x["deteccoes"], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar([r["arquivo"].replace(".jpg", "") for r in top10],
                  [r["deteccoes"] for r in top10], color="#264653", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Tile"); ax.set_ylabel("Detecções")
    ax.set_title("Top 10 Tiles por Número de Detecções", fontweight="bold")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    _salvar("top10_tiles.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(tempos, bins=40, color="#457b9d", edgecolor="white")
    ax.axvline(np.mean(tempos), color="red", linestyle="--", label=f"Média: {np.mean(tempos):.3f}s")
    ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Frequência (tiles)")
    ax.set_title("Distribuição do Tempo de Processamento", fontweight="bold")
    ax.legend(); plt.tight_layout()
    _salvar("tempo_processamento.png")

    _hist(coberturas, 40, "#a8dadc",
          "Cobertura HSV (% do tile)", "Frequência (tiles)",
          "Distribuição da Cobertura da Máscara HSV", "cobertura_percentual.png")

    memorias = [r["memoria_mb"] for r in resultados]
    _hist(memorias, 40, "#9b5de5",
          "Memória de pico por tile (MB)", "Frequência (tiles)",
          "Distribuição do Uso de Memória por Tile", "memoria_por_tile.png")


# ── Relatório Markdown ────────────────────────────────────────────────────────

def gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos):
    tempos     = [r["tempo_s"]      for r in resultados]
    deteccoes  = [r["deteccoes"]    for r in resultados]
    coberturas = [r["coverage_pct"] for r in resultados]
    n          = len(resultados)
    n_com      = sum(1 for d in deteccoes if d > 0)
    total_det  = sum(deteccoes)
    area_tot   = sum(r["area_total_m2"] for r in resultados)

    top10  = sorted(resultados, key=lambda x: x["deteccoes"], reverse=True)[:10]
    linhas = "\n".join(
        f"| {r['arquivo']} | {r['deteccoes']} | {r['area_total_px']:,.0f} | {r['area_total_m2']:.4f} | {r['coverage_pct']:.2f}% |"
        for r in top10
    )
    fmt = lambda arr, f: f"{arr:{f}}" if todas_areas else "—"

    md = f"""# Relatório de Detecção de Embaúba (Cecropia) — Visão Computacional Clássica

## 1. Pipeline

| Etapa | Função | Parâmetros |
|-------|--------|------------|
| 1 | `suavizar` | Gaussian Blur {BLUR_KERNEL}, σ=0 |
| 2 | `segmentar_hsv` | H:[44,58] S:[144,231] V:[104,203] |
| 3 | `fechar_mascara` | CLOSE elipse {CLOSE_SIZE}×{CLOSE_SIZE} |
| 4 | `abrir_mascara` | OPEN elipse {OPEN_SIZE}×{OPEN_SIZE} |
| 5 | `filtrar_contornos` | área > {AREA_MIN:,} px² |
| 6 | `desenhar_hull` | convex hull por contorno |

## 2. Dataset

| Parâmetro | Valor |
|-----------|-------|
| Tiles | 992 × 2048×2048 px |
| Sobreposição | 512 px (25%) |
| CRS | EPSG:31982 (UTM 22S) |
| Resolução | {pixel_size:.6f} u/px |

## 3. Estatísticas

| Métrica | Valor |
|---------|-------|
| Tiles processados | {n} |
| Tiles pretos (ignorados) | {tiles_pretos} |
| Com ≥1 detecção | {n_com} ({100*n_com/n:.1f}%) |
| Total de detecções | {total_det:,} |
| Área total estimada | {area_tot:.2f} u² |
| Tempo total | {t_total:.1f}s |
| Tempo médio/tile | {np.mean(tempos)*1000:.1f}ms ± {np.std(tempos)*1000:.1f}ms |
| Detecções/tile | {np.mean(deteccoes):.2f} ± {np.std(deteccoes):.2f} |
| Cobertura HSV média | {np.mean(coberturas):.3f}% |
| Área/região: média | {fmt(np.mean(todas_areas), '.0f')} px² |
| Área/região: máx | {fmt(np.max(todas_areas), '.0f')} px² |
| **Memória (pico por tile)** | |
| Média | {np.mean([r["memoria_mb"] for r in resultados]):.1f} MB |
| Máximo | {np.max([r["memoria_mb"] for r in resultados]):.1f} MB |

## 4. Top 10 Tiles

| Arquivo | Detecções | Área (px²) | Área (u²) | Cobertura |
|---------|-----------|------------|-----------|-----------|
{linhas}

## 5. Figuras

| Arquivo | Descrição |
|---------|-----------|
| `passo_a_passo/tile_XXXX.png` | 8 etapas do pipeline por tile (992 arquivos) |
| `hist_deteccoes_por_tile.png` | Distribuição de detecções |
| `hist_areas_deteccao.png` | Distribuição de áreas |
| `top10_tiles.png` | Top 10 por detecções |
| `tempo_processamento.png` | Tempo por tile |
| `cobertura_percentual.png` | Cobertura HSV |

*Gerado por `relatorio_embauba.py`*
"""
    caminho = os.path.join(OUTPUT_DIR, "relatorio.md")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  [OK] {caminho}")

    print(f"\n{'='*55}")
    print("  RESUMO — Detecção de Embaúba (Cecropia)")
    print(f"{'='*55}")
    print(f"  Tiles processados     : {n}")
    print(f"  Tiles pretos (pulados): {tiles_pretos}")
    print(f"  Tiles c/ detecções    : {n_com} ({100*n_com/n:.1f}%)")
    print(f"  Total de detecções    : {total_det:,}")
    print(f"  Tempo total/médio     : {t_total:.1f}s / {np.mean(tempos)*1000:.1f}ms por tile")
    print(f"  Detecções/tile        : {np.mean(deteccoes):.2f} ± {np.std(deteccoes):.2f}")
    print(f"  Cobertura HSV média   : {np.mean(coberturas):.3f}%")
    memorias = [r["memoria_mb"] for r in resultados]
    print(f"  Memória média/tile    : {np.mean(memorias):.1f} MB (pico: {np.max(memorias):.1f} MB)")
    print(f"{'='*55}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _salvar(nome):
    caminho = os.path.join(OUTPUT_DIR, nome)
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {caminho}")

def _hist(dados, bins, cor, xlabel, ylabel, titulo, nome):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dados, bins=bins, color=cor, edgecolor="white")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold")
    plt.tight_layout()
    _salvar(nome)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PASSO_A_PASSO_DIR, exist_ok=True)

    print("\n=== PARTE 1: Processamento de todos os tiles + passo a passo ===")
    resultados, todas_areas, pixel_size, t_total, tiles_pretos = processar_todos_tiles()

    print("\n=== PARTE 3: Figuras de estatísticas ===")
    gerar_figuras(resultados, todas_areas)

    print("\n=== PARTE 4: Relatório Markdown ===")
    gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos)

    print(f"\n[CONCLUÍDO] Arquivos em '{OUTPUT_DIR}/'")
