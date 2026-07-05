"""Ponto de entrada principal do projeto.

Sem argumentos, processa `data/tiles/` e gera o output completo em
`data/output/` (uma pasta por tile em `tiles/` com JSON + figuras do pipeline,
gráficos e relatório). Com uma imagem ou pasta, gera a mesma pasta por imagem
com JSON + figuras.
"""

import argparse
import json
import os
import sys

import cv2

from core.execucao import processar_todos_tiles
from core.estatisticas import gerar_figuras, gerar_relatorio, calcular_validacao, gerar_exemplos_validacao
from core.deteccao import detectar_cronometrado, detectar, salvar_figuras_etapas

DATA:              str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TILES_DIR:         str = os.path.join(DATA, "tiles")
VALIDACAO_DIR:     str = os.path.join(DATA, "validacao")
OUTPUT_DIR:        str = os.path.join(DATA, "output")
EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def pipeline_completo() -> None:
    """Processa todos os tiles oficiais e gera os artefatos completos."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== PARTE 1: Processamento de todos os tiles (JSON + figuras por tile) ===")
    resultados, todas_areas, pixel_size, t_total, tiles_pretos = processar_todos_tiles(
        TILES_DIR, OUTPUT_DIR
    )

    print("\n=== PARTE 2: Figuras de estatísticas ===")
    gerar_figuras(resultados, todas_areas, OUTPUT_DIR)

    print("\n=== PARTE 3: Relatório Markdown ===")
    gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos, OUTPUT_DIR)

    print(f"\n[CONCLUÍDO] Arquivos em '{OUTPUT_DIR}/'")


def _saida_padrao(caminho: str) -> str:
    base = caminho.rstrip("/")
    if os.path.isfile(base):
        base = os.path.splitext(base)[0]
    return base + "_deteccoes"


def _imagens(caminho: str) -> list[str]:
    if os.path.isdir(caminho):
        return [os.path.join(caminho, f) for f in sorted(os.listdir(caminho))
                if f.lower().endswith(EXTS)]
    if os.path.isfile(caminho):
        return [caminho]
    print(f"[ERRO] caminho não encontrado: '{caminho}'.")
    sys.exit(1)


def _processar_imagem(caminho: str, saida: str) -> int:
    nome = os.path.basename(caminho)
    img = cv2.imread(caminho)
    if img is None:
        print(f"  [!] ignorada (não pôde ser lida): {nome}")
        return 0

    r, tempo, memoria_mb = detectar_cronometrado(img)
    base = os.path.splitext(nome)[0]
    pasta = os.path.join(saida, base)
    os.makedirs(pasta, exist_ok=True)

    embaubas = []
    for c in r["contornos"]:
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)
        embaubas.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": round(float(cv2.contourArea(c)), 1),
            "poligono": hull.reshape(-1, 2).tolist(),
        })
    with open(os.path.join(pasta, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "imagem": nome,
            "embaubas": embaubas,
            "n_candidatos": r["n_candidatos"],
            "n_deteccoes": r["count"],
            "coverage_pct": r["coverage_pct"],
            "tempo_s": tempo,
            "memoria_mb": memoria_mb,
        }, f, indent=2, ensure_ascii=False)
    salvar_figuras_etapas(r, nome, saida)

    print(f"  {nome}: {r['count']} detecções "
          f"({r['n_candidatos']} candidatos, {tempo*1000:.0f}ms)")
    return r["count"]


def detectar_caminho(caminho: str, saida: str) -> None:
    """Gera JSON + figuras do pipeline por imagem para uma imagem ou pasta."""
    arquivos = _imagens(caminho)
    if not arquivos:
        print(f"[ERRO] Nenhuma imagem encontrada em '{caminho}'.")
        sys.exit(1)

    os.makedirs(saida, exist_ok=True)
    total = sum(_processar_imagem(c, saida) for c in arquivos)
    print(f"\n[CONCLUÍDO] {len(arquivos)} imagem(ns), {total} detecções no total. Saída em '{saida}/'")


def imprimir_validacao() -> None:
    """Imprime as métricas de validação a partir de `data/validacao`."""
    v = calcular_validacao(VALIDACAO_DIR)
    print(f"\n=== Validação ({v['validados']} tiles validados) ===")
    print(f"  TP: {v['tp']}   FP: {v['fp']}   FN (faltantes): {v['fn']}")
    print(f"  Precisão: {v['precisao']:.1f}%   Recall: {v['recall']:.1f}%   F1: {v['f1']:.1f}%")


def gerar_etapas(tile: str, saida: str) -> str:
    """Gera as figuras de etapas do pipeline de um tile em `saida/<tile>/`."""
    img = cv2.imread(os.path.join(TILES_DIR, f"{tile}.jpg"))
    if img is None:
        img = cv2.imread(os.path.join(VALIDACAO_DIR, tile, f"{tile}.jpg"))
    if img is None:
        print(f"[ERRO] tile não encontrado: '{tile}'.")
        sys.exit(1)
    os.makedirs(saida, exist_ok=True)
    salvar_figuras_etapas(detectar(img), f"{tile}.jpg", saida)
    return os.path.join(saida, tile)


def main() -> None:
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description="Detecta embaúba (Cecropia) em tiles, imagem ou pasta.")
    ap.add_argument("caminho", nargs="?", default=None,
                    help="imagem ou pasta de imagens; omitido roda data/tiles completo")
    ap.add_argument("--saida", default=None,
                    help="pasta de saída (padrão: data/output no pipeline completo ou <caminho>_deteccoes)")
    ap.add_argument("--validacao", action="store_true",
                    help="só imprime as métricas de validação (data/validacao) e sai")
    ap.add_argument("--exemplos", nargs="?", default=None, const="", metavar="TILE",
                    help="gera a figura de exemplos TP/FP/FN e sai (padrão: melhores do "
                         "conjunto validado; informe um tile para usar só ele)")
    ap.add_argument("--etapas", default=None, metavar="TILE",
                    help="gera as figuras de etapas do pipeline de um tile e sai")
    args = ap.parse_args()

    # --saida também vale para os modos sem caminho (pipeline/exemplos/etapas),
    # permitindo acumular todas as figuras numa mesma pasta.
    if args.caminho is None and args.saida is not None:
        OUTPUT_DIR = args.saida

    if args.validacao:
        imprimir_validacao()
        return

    if args.exemplos is not None:
        out = gerar_exemplos_validacao(VALIDACAO_DIR, TILES_DIR, OUTPUT_DIR, args.exemplos or None)
        print(f"[OK] figura gerada: {out}")
        return

    if args.etapas is not None:
        out = gerar_etapas(args.etapas, OUTPUT_DIR)
        print(f"[OK] etapas geradas em: {out}/")
        return

    if args.caminho is None:
        pipeline_completo()
    else:
        detectar_caminho(args.caminho, args.saida or _saida_padrao(args.caminho))


if __name__ == "__main__":
    main()
