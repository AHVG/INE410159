import os

from execucao import processar_todos_tiles
from estatisticas import gerar_figuras, gerar_relatorio

TILES_DIR         = "tiles"
OUTPUT_DIR        = "output"
PASSO_A_PASSO_DIR = os.path.join(OUTPUT_DIR, "passo_a_passo")
CHECKPOINT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")

if __name__ == "__main__":
    os.makedirs(PASSO_A_PASSO_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n=== PARTE 1: Processamento de todos os tiles + passo a passo ===")
    resultados, todas_areas, pixel_size, t_total, tiles_pretos = processar_todos_tiles(
        TILES_DIR, PASSO_A_PASSO_DIR, OUTPUT_DIR
    )

    print("\n=== PARTE 2: Figuras de estatísticas ===")
    gerar_figuras(resultados, todas_areas, OUTPUT_DIR)

    print("\n=== PARTE 3: Relatório Markdown ===")
    gerar_relatorio(resultados, todas_areas, pixel_size, t_total, tiles_pretos, OUTPUT_DIR)

    print(f"\n[CONCLUÍDO] Arquivos em '{OUTPUT_DIR}/'")
