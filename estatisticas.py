import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from deteccao import BLUR_KERNEL, CLOSE_SIZE, OPEN_SIZE, AREA_MIN


# ── Helpers internos ──────────────────────────────────────────────────────────

def _salvar(nome, output_dir):
    caminho = os.path.join(output_dir, nome)
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {caminho}")


def _hist(dados, bins, cor, xlabel, ylabel, titulo, nome, output_dir):
    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dados, bins=bins, color=cor, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold")
    plt.tight_layout()
    _salvar(nome, output_dir)


# ── Figuras estatísticas ──────────────────────────────────────────────────────

def gerar_figuras(resultados, todas_areas, output_dir):
    tempos     = [r["tempo_s"]      for r in resultados]
    deteccoes  = [r["deteccoes"]    for r in resultados]
    coberturas = [r["coverage_pct"] for r in resultados]

    _hist(deteccoes, range(0, max(deteccoes) + 2), "#2a9d8f",
          "Número de detecções por tile", "Frequência (tiles)",
          "Distribuição de Detecções por Tile", "hist_deteccoes_por_tile.png", output_dir)

    if todas_areas:
        _hist(todas_areas, 40, "#e76f51",
              "Área da detecção (px²)", "Frequência",
              "Distribuição das Áreas de Detecção", "hist_areas_deteccao.png", output_dir)

    top10 = sorted(resultados, key=lambda x: x["deteccoes"], reverse=True)[:10]
    _, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar([r["arquivo"].replace(".jpg", "") for r in top10],
                  [r["deteccoes"] for r in top10], color="#264653", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Tile")
    ax.set_ylabel("Detecções")
    ax.set_title("Top 10 Tiles por Número de Detecções", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _salvar("top10_tiles.png", output_dir)

    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(tempos, bins=40, color="#457b9d", edgecolor="white")
    ax.axvline(np.mean(tempos), color="red", linestyle="--",
               label=f"Média: {np.mean(tempos):.3f}s")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (tiles)")
    ax.set_title("Distribuição do Tempo de Processamento", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _salvar("tempo_processamento.png", output_dir)

    _hist(coberturas, 40, "#a8dadc",
          "Cobertura HSV (% do tile)", "Frequência (tiles)",
          "Distribuição da Cobertura da Máscara HSV", "cobertura_percentual.png", output_dir)

    memorias = [r["memoria_mb"] for r in resultados]
    _hist(memorias, 40, "#9b5de5",
          "Memória de pico por tile (MB)", "Frequência (tiles)",
          "Distribuição do Uso de Memória por Tile", "memoria_por_tile.png", output_dir)


# ── Relatório Markdown ────────────────────────────────────────────────────────

def gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos, output_dir):
    tempos     = [r["tempo_s"]      for r in resultados]
    deteccoes  = [r["deteccoes"]    for r in resultados]
    coberturas = [r["coverage_pct"] for r in resultados]
    memorias   = [r["memoria_mb"]   for r in resultados]
    n          = len(resultados)
    n_com      = sum(1 for d in deteccoes if d > 0)
    total_det  = sum(deteccoes)
    area_tot   = sum(r["area_total_m2"] for r in resultados)

    top10  = sorted(resultados, key=lambda x: x["deteccoes"], reverse=True)[:10]
    linhas = "\n".join(
        f"| {r['arquivo']} | {r['deteccoes']} | {r['area_total_px']:,.0f} | {r['area_total_m2']:.4f} | {r['coverage_pct']:.2f}% |"
        for r in top10
    )
    fmt = lambda v, f: f"{v:{f}}" if todas_areas else "—"

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
| Memória média/tile | {np.mean(memorias):.1f} MB |
| Memória pico/tile | {np.max(memorias):.1f} MB |

## 4. Top 10 Tiles

| Arquivo | Detecções | Área (px²) | Área (u²) | Cobertura |
|---------|-----------|------------|-----------|-----------|
{linhas}

## 5. Figuras

| Arquivo | Descrição |
|---------|-----------|
| `passo_a_passo/tile_XXXX.png` | 8 etapas do pipeline por tile |
| `hist_deteccoes_por_tile.png` | Distribuição de detecções |
| `hist_areas_deteccao.png` | Distribuição de áreas |
| `top10_tiles.png` | Top 10 por detecções |
| `tempo_processamento.png` | Tempo por tile |
| `cobertura_percentual.png` | Cobertura HSV |
| `memoria_por_tile.png` | Uso de memória por tile |

*Gerado por `main.py`*
"""
    caminho = os.path.join(output_dir, "relatorio.md")
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
    print(f"  Memória média/tile    : {np.mean(memorias):.1f} MB (pico: {np.max(memorias):.1f} MB)")
    print(f"{'='*55}")
