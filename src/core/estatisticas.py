"""Geração das figuras estatísticas e do relatório Markdown a partir dos
resultados do pipeline."""

import os
import json
import tempfile
from typing import Any, Sequence

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .deteccao import (
    BLUR_KERNEL,
    CLOSE_SIZE,
    OPEN_SIZE,
    AREA_MIN,
    AREA_GRANDE,
    CIRC_MAX,
    AREA_MAX,
    SOL_MIN,
)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _salvar(nome: str, output_dir: str) -> None:
    """Salva a figura matplotlib atual em `output_dir/nome` e a fecha."""
    caminho = os.path.join(output_dir, nome)
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {caminho}")


def _hist(
    dados: Sequence[float],
    bins: Any,
    cor: str,
    xlabel: str,
    ylabel: str,
    titulo: str,
    nome: str,
    output_dir: str,
) -> None:
    """Desenha e salva um histograma simples com rótulos e título."""
    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dados, bins=bins, color=cor, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold")
    plt.tight_layout()
    _salvar(nome, output_dir)


def _bytes_legivel(n_bytes: int) -> str:
    """Formata tamanho em disco em unidade legível."""
    valor = float(n_bytes)
    for unidade in ("B", "KB", "MB", "GB", "TB"):
        if valor < 1024 or unidade == "TB":
            return f"{valor:.1f} {unidade}"
        valor /= 1024


def _tamanho_dir(path: str) -> str:
    """Calcula o tamanho total de arquivos dentro de um diretório."""
    total = 0
    if os.path.exists(path):
        for raiz, _, nomes in os.walk(path):
            for nome in nomes:
                try:
                    total += os.path.getsize(os.path.join(raiz, nome))
                except OSError:
                    pass
    return _bytes_legivel(total)


def _pct(num: float, den: float) -> float:
    """Percentual robusto para tabelas do relatório."""
    return 100 * num / den if den else 0.0


def calcular_validacao(validacao_dir: str) -> dict[str, Any]:
    """Métricas do detector a partir dos JSONs revisados de validação.

    Cada JSON traz `embaubas` (saída do detector), `falsos_positivos` (índices das
    detecções que o revisor marcou como erro) e `faltantes` (copas que o detector
    não pegou). As métricas somam apenas os tiles revisados:
        TP = detecções confirmadas;  FP = falsos positivos;  FN = faltantes.
    """
    jsons = sorted(
        os.path.join(raiz, nome)
        for raiz, _, nomes in os.walk(validacao_dir)
        for nome in nomes
        if nome.endswith(".json")
    ) if os.path.exists(validacao_dir) else []

    revisados = tp = fp = fn = 0
    for path in jsons:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("revisado", False):
            continue
        revisados += 1
        n_det = len(meta.get("embaubas", []))
        n_fp = len(meta.get("falsos_positivos", []))
        fp += n_fp
        tp += n_det - n_fp
        fn += len(meta.get("faltantes", []))

    return {
        "jsons": len(jsons),
        "revisados": revisados,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precisao": _pct(tp, tp + fp),
        "recall": _pct(tp, tp + fn),
        "f1": _pct(2 * tp, 2 * tp + fp + fn),
    }


# ── Figuras estatísticas ──────────────────────────────────────────────────────

def gerar_figuras(
    resultados: list[dict[str, Any]],
    todas_areas: list[float],
    output_dir: str,
) -> None:
    """Gera todos os PNGs estatísticos (histogramas, top 10, tempo, memória).

    Args:
        resultados: Lista de métricas por tile (saída de `processar_todos_tiles`).
        todas_areas: Áreas em px² de todas as regiões detectadas.
        output_dir: Diretório onde as figuras são gravadas.
    """
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
    ax.axvline(float(np.mean(tempos)), color="red", linestyle="--",
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

def gerar_relatorio(
    resultados: list[dict[str, Any]],
    todas_areas: list[float],
    pixel_size: float,
    t_total: float,
    tiles_pretos: int,
    output_dir: str,
) -> None:
    """Escreve `relatorio.md` e imprime um resumo no terminal.

    Args:
        resultados: Métricas por tile.
        todas_areas: Áreas em px² de todas as regiões detectadas.
        pixel_size: Tamanho do pixel em unidades do CRS (para converter px²→u²).
        t_total: Tempo total de processamento, em segundos.
        tiles_pretos: Quantidade de tiles ignorados por estarem fora da área.
        output_dir: Diretório onde `relatorio.md` é gravado.
    """
    tempos     = [r["tempo_s"]      for r in resultados]
    deteccoes  = [r["deteccoes"]    for r in resultados]
    coberturas = [r["coverage_pct"] for r in resultados]
    memorias   = [r["memoria_mb"]   for r in resultados]
    n          = len(resultados)
    n_com      = sum(1 for d in deteccoes if d > 0)
    total_det  = sum(deteccoes)
    area_tot   = sum(r["area_total_m2"] for r in resultados)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    tiles_dir = os.path.join(data_dir, "tiles")
    validacao_dir = os.path.join(data_dir, "validacao")
    tamanho_tiles = _tamanho_dir(tiles_dir)
    validacao = calcular_validacao(validacao_dir)

    top10  = sorted(resultados, key=lambda x: x["deteccoes"], reverse=True)[:10]
    linhas = "\n".join(
        f"| {r['arquivo']} | {r['deteccoes']} | {r['area_total_px']:,.0f} | {r['area_total_m2']:.4f} | {r['coverage_pct']:.2f}% |"
        for r in top10
    )
    fmt = lambda v, f: f"{v:{f}}" if todas_areas else "—"

    md = f"""# Relatório de Detecção de Embaúba (Cecropia) — Visão Computacional Clássica

## 1. Introdução

Este trabalho detecta árvores **Cecropia** (embaúba) em mosaico aéreo por visão
computacional clássica, sem deep learning. A abordagem combina segmentação por
cor no espaço HSV, operações morfológicas e filtragem geométrica dos contornos.

O objetivo é localizar copas com a coloração típica da embaúba e reduzir falsos
positivos frequentes, especialmente centros de palmeiras e manchas de vegetação
que entram na mesma faixa de cor. A escolha por métodos clássicos permite um
pipeline simples, interpretável e executável com poucos recursos.

## 2. Dataset

| Parâmetro | Valor |
|-----------|-------|
| Tiles | 992 imagens JPEG de 2048×2048 px |
| Sobreposição | 512 px (25%) |
| CRS | EPSG:31982 (UTM 22S) |
| Resolução | {pixel_size:.6f} u/px |
| Tamanho aproximado em disco (`data/tiles`) | {tamanho_tiles} |
| Tiles pretos ignorados | {tiles_pretos} |

Os tiles cobrem um mosaico aéreo georreferenciado. Imagens praticamente pretas,
fora da área mapeada, são descartadas automaticamente antes da detecção para não
distorcer as estatísticas.

O conjunto de validação fica em `data/validacao/`, com uma subpasta por imagem
revisada (`<nome>.jpg`, `<nome>.json`, `<nome>_vis.png`). Atualmente há
{validacao['jsons']} JSONs de validação, {validacao['revisados']} revisados
manualmente. Cada JSON armazena a saída do detector (`embaubas`), os índices
marcados como falso positivo na revisão (`falsos_positivos`) e caixas `faltantes`
para copas que o detector não pegou.

## 3. Pipeline

| Etapa | O que faz | Motivação | Parâmetros |
|-------|-----------|-----------|------------|
| 1. Suavização | aplica Gaussian Blur | reduz ruído fino de textura antes da cor | kernel {BLUR_KERNEL}, σ=0 |
| 2. Conversão BGR→HSV | muda o espaço de cor | separa matiz, saturação e brilho | `cv2.COLOR_BGR2HSV` |
| 3. Limiarização por cor | isola pixels na faixa das copas | destaca o verde característico da embaúba | H:[44,58] S:[144,231] V:[104,203] |
| 4. Fechamento morfológico | conecta fragmentos próximos | fecha buracos e quebras dentro da copa | elipse {CLOSE_SIZE}×{CLOSE_SIZE} |
| 5. Abertura morfológica | remove componentes pequenos | reduz respingos de ruído após o fechamento | elipse {OPEN_SIZE}×{OPEN_SIZE} |
| 6. Detecção de contornos | extrai regiões conectadas | transforma a máscara em candidatos | `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE` |
| 7. Filtro de área | descarta regiões pequenas | remove manchas muito menores que uma copa | área > {AREA_MIN:,} px² |
| 8. Filtro final | classifica candidatos como embaúba | reduz falsos positivos por forma e tamanho | (área ≥ {AREA_GRANDE:,} ou circ ≤ {CIRC_MAX}) e área ≤ {AREA_MAX:,} e solidez ≥ {SOL_MIN} |
| 9. Convex Hull | aproxima a copa para desenho/exportação | contorno mais estável para visualização e bounds | `cv2.convexHull` |

## 4. Resultados

### Estatísticas gerais

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

### Top 10 tiles por número de detecções

| Arquivo | Detecções | Área (px²) | Área (u²) | Cobertura |
|---------|-----------|------------|-----------|-----------|
{linhas}

### Validação manual

A validação compara a saída do detector com a revisão manual de
{validacao['revisados']}/{validacao['jsons']} tiles. Na revisão, cada detecção é
confirmada como embaúba ou marcada como falso positivo, e as copas que o detector
não pegou são anotadas como faltantes.

| Métrica | Valor |
|---------|-------|
| Tiles revisados | {validacao['revisados']}/{validacao['jsons']} |
| Verdadeiros positivos (TP) | {validacao['tp']} |
| Falsos positivos (FP) | {validacao['fp']} |
| Falsos negativos / faltantes (FN) | {validacao['fn']} |
| Precisão | {validacao['precisao']:.1f}% |
| Recall | {validacao['recall']:.1f}% |
| F1 | {validacao['f1']:.1f}% |

A precisão mede quantas das detecções são realmente embaúba (o filtro de forma e
tamanho remove a maior parte das palmeiras), e o recall mede quantas das copas
reais o detector encontrou — o complemento são as `faltantes` que a segmentação
HSV não chegou a produzir.

### Figuras geradas

| Arquivo | Descrição |
|---------|-----------|
| `passo_a_passo/tile_XXXX.png` | 8 etapas do pipeline por tile selecionado |
| `hist_deteccoes_por_tile.png` | Distribuição de detecções |
| `hist_areas_deteccao.png` | Distribuição de áreas |
| `top10_tiles.png` | Top 10 por detecções |
| `tempo_processamento.png` | Tempo por tile |
| `cobertura_percentual.png` | Cobertura HSV |
| `memoria_por_tile.png` | Uso de memória por tile |

## 5. Conclusão

O pipeline processou {n} tiles válidos e encontrou {total_det:,} regiões
classificadas como embaúba. Na validação manual de {validacao['revisados']} tiles,
o detector atingiu {validacao['precisao']:.1f}% de precisão e
{validacao['recall']:.1f}% de recall, mostrando que o filtro de forma e tamanho
remove a maior parte dos falsos positivos (palmeiras). O método é direto e
auditável: as figuras de passo a passo permitem inspecionar as representações
intermediárias dos tiles selecionados.

As principais limitações são falsos positivos em vegetação com cor semelhante,
sensibilidade a iluminação/sombra e duplicação possível por sobreposição entre
tiles. Trabalhos futuros incluem consolidar detecções entre tiles vizinhos,
ampliar a revisão manual de `faltantes` na validação e ajustar parâmetros por
condições de iluminação.

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
