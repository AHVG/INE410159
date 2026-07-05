"""Ponto de entrada principal do projeto.

Sem argumentos, processa `data/tiles/` e gera o output completo em
`data/output/` (passo a passo, gráficos e relatório). Com uma imagem ou pasta,
gera JSON por imagem e, opcionalmente, visualizações.
"""

import argparse
import json
import os
import sys

import cv2

from core.execucao import processar_todos_tiles, regenerar_passo_a_passo_todos
from core.estatisticas import gerar_figuras, gerar_relatorio, calcular_validacao
from core.deteccao import detectar, analisar, salvar_passo_a_passo

DATA:              str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TILES_DIR:         str = os.path.join(DATA, "tiles")
VALIDACAO_DIR:     str = os.path.join(DATA, "validacao")
OUTPUT_DIR:        str = os.path.join(DATA, "output")
PASSO_A_PASSO_DIR: str = os.path.join(OUTPUT_DIR, "passo_a_passo")
CHECKPOINT_DIR:    str = os.path.join(OUTPUT_DIR, "checkpoints")
EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def pipeline_completo(passo_a_passo_cada: int = 1) -> None:
    """Processa todos os tiles oficiais e gera os artefatos completos."""
    os.makedirs(PASSO_A_PASSO_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n=== PARTE 1: Processamento de todos os tiles + passo a passo ===")
    resultados, todas_areas, pixel_size, t_total, tiles_pretos = processar_todos_tiles(
        TILES_DIR, PASSO_A_PASSO_DIR, OUTPUT_DIR, passo_a_passo_cada
    )

    print("\n=== PARTE 2: Figuras de estatísticas ===")
    gerar_figuras(resultados, todas_areas, OUTPUT_DIR)

    print("\n=== PARTE 3: Relatório Markdown ===")
    gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos, OUTPUT_DIR)

    print(f"\n[CONCLUÍDO] Arquivos em '{OUTPUT_DIR}/'")


def regenerar_passo_a_passo() -> None:
    """Recria todas as figuras de passo a passo dos tiles oficiais."""
    ok, pretos, erros = regenerar_passo_a_passo_todos(TILES_DIR, PASSO_A_PASSO_DIR)
    print(f"\n[CONCLUÍDO] {ok} figuras em '{PASSO_A_PASSO_DIR}/' "
          f"({pretos} tiles pretos ignorados, {erros} erros).")


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


def _processar_imagem(caminho: str, saida: str, vis: bool, passo_a_passo: bool) -> int:
    nome = os.path.basename(caminho)
    img = cv2.imread(caminho)
    if img is None:
        print(f"  [!] ignorada (não pôde ser lida): {nome}")
        return 0

    r = analisar(img)
    base = os.path.splitext(nome)[0]
    with open(os.path.join(saida, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump({"imagem": nome, **r}, f, indent=2, ensure_ascii=False)

    if vis or passo_a_passo:
        det = detectar(img)
        if vis:
            cv2.imwrite(os.path.join(saida, f"{base}_resultado.png"),
                        cv2.cvtColor(det["resultado"], cv2.COLOR_RGB2BGR))
        if passo_a_passo:
            salvar_passo_a_passo(det, nome, saida)

    print(f"  {nome}: {r['n_deteccoes']} detecções "
          f"({r['n_candidatos']} candidatos, {r['tempo_s']*1000:.0f}ms)")
    return r["n_deteccoes"]


def detectar_caminho(caminho: str, saida: str, vis: bool, passo_a_passo: bool) -> None:
    """Gera JSON por imagem para uma imagem ou pasta arbitrária."""
    arquivos = _imagens(caminho)
    if not arquivos:
        print(f"[ERRO] Nenhuma imagem encontrada em '{caminho}'.")
        sys.exit(1)

    os.makedirs(saida, exist_ok=True)
    total = sum(_processar_imagem(c, saida, vis, passo_a_passo) for c in arquivos)
    print(f"\n[CONCLUÍDO] {len(arquivos)} imagem(ns), {total} detecções no total. Saída em '{saida}/'")


def imprimir_validacao() -> None:
    """Imprime as métricas de validação a partir de `data/validacao`."""
    v = calcular_validacao(VALIDACAO_DIR)
    print(f"\n=== Validação ({v['revisados']}/{v['jsons']} tiles revisados) ===")
    print(f"  TP: {v['tp']}   FP: {v['fp']}   FN (faltantes): {v['fn']}")
    print(f"  Precisão: {v['precisao']:.1f}%   Recall: {v['recall']:.1f}%   F1: {v['f1']:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecta embaúba (Cecropia) em tiles, imagem ou pasta.")
    ap.add_argument("caminho", nargs="?", default=None,
                    help="imagem ou pasta de imagens; omitido roda data/tiles completo")
    ap.add_argument("--saida", default=None,
                    help="pasta de saída (padrão: data/output no pipeline completo ou <caminho>_deteccoes)")
    ap.add_argument("--vis", action="store_true",
                    help="com <caminho>, também salva PNG anotado por imagem")
    ap.add_argument("--passo-a-passo", action="store_true",
                    help="com <caminho>, também salva a figura com as etapas do pipeline")
    ap.add_argument("--passo-a-passo-cada", type=int, default=1,
                    help="no pipeline completo, salva passo a passo a cada N tiles não pretos")
    ap.add_argument("--somente-passo-a-passo", action="store_true",
                    help="regenera o passo a passo de todos os tiles oficiais e sai")
    ap.add_argument("--validacao", action="store_true",
                    help="só imprime as métricas de validação (data/validacao) e sai")
    args = ap.parse_args()
    if args.passo_a_passo_cada < 1:
        ap.error("--passo-a-passo-cada deve ser maior ou igual a 1")

    if args.validacao:
        imprimir_validacao()
        return

    if args.somente_passo_a_passo:
        regenerar_passo_a_passo()
        return

    if args.caminho is None:
        if args.saida is not None:
            global OUTPUT_DIR, PASSO_A_PASSO_DIR, CHECKPOINT_DIR
            OUTPUT_DIR = args.saida
            PASSO_A_PASSO_DIR = os.path.join(OUTPUT_DIR, "passo_a_passo")
            CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
        pipeline_completo(args.passo_a_passo_cada)
    else:
        detectar_caminho(args.caminho, args.saida or _saida_padrao(args.caminho),
                         args.vis, args.passo_a_passo)


if __name__ == "__main__":
    main()
