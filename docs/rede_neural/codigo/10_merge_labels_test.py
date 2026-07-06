"""
10_merge_labels_train.py
==========================
1. Faz BACKUP da pasta de labels original do train
2. Faz MERGE (append) dos labels novos curados com os existentes
3. Limpa caches

Autor: Gabriel A. Ferreira Gualda
"""

import os
import shutil


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Labels novos curados do train
LABELS_NOVOS_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/nao_anotadas_train/labels_novos"

# Labels originais do train
LABELS_TRAIN_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/train"

# Backup dos labels originais
LABELS_BACKUP_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/train_backup_original"

# Cache base dir
LABELS_BASE_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels"


# ==============================================================================
# PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print("MERGE DE LABELS — TRAIN")
    print("=" * 60)

    # =========================================================================
    # 1. Backup
    # =========================================================================
    print("\n[1/3] Fazendo backup dos labels originais do train...")

    if os.path.exists(LABELS_BACKUP_DIR):
        print(f"  Backup já existe: {LABELS_BACKUP_DIR}")
    else:
        shutil.copytree(LABELS_TRAIN_DIR, LABELS_BACKUP_DIR)
        n_backup = len([f for f in os.listdir(LABELS_BACKUP_DIR) if f.endswith('.txt')])
        print(f"  Backup criado: {LABELS_BACKUP_DIR}")
        print(f"  Arquivos copiados: {n_backup}")

    # =========================================================================
    # 2. Merge
    # =========================================================================
    print("\n[2/3] Fazendo merge dos labels novos no train...")

    novos_files = [f for f in os.listdir(LABELS_NOVOS_DIR) if f.endswith('.txt')]
    print(f"  Labels novos (curados): {len(novos_files)} arquivos")

    total_bboxes_adicionados = 0
    arquivos_modificados = 0

    for filename in novos_files:
        novo_path = os.path.join(LABELS_NOVOS_DIR, filename)
        existente_path = os.path.join(LABELS_TRAIN_DIR, filename)

        with open(novo_path, "r") as f:
            novas_linhas = [line.strip() for line in f if line.strip()]

        if len(novas_linhas) == 0:
            continue

        with open(existente_path, "a") as f:
            for linha in novas_linhas:
                f.write("\n" + linha)

        total_bboxes_adicionados += len(novas_linhas)
        arquivos_modificados += 1

    print(f"  Arquivos modificados: {arquivos_modificados}")
    print(f"  Bboxes adicionados: {total_bboxes_adicionados}")

    # Contar total
    total_labels = 0
    for filename in os.listdir(LABELS_TRAIN_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(LABELS_TRAIN_DIR, filename)) as f:
                total_labels += sum(1 for line in f if line.strip())

    print(f"  Total de labels no train agora: {total_labels}")

    # =========================================================================
    # 3. Limpar caches
    # =========================================================================
    print("\n[3/3] Limpando caches...")

    for cache_name in ["train.cache", "val.cache", "test.cache"]:
        cache_path = os.path.join(LABELS_BASE_DIR, cache_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"  Cache removido: {cache_name}")
        else:
            print(f"  {cache_name} não encontrado (OK)")

    print("\n" + "=" * 60)
    print("MERGE TRAIN CONCLUÍDO!")
    print("=" * 60)
    print(f"  Labels originais preservados em: {LABELS_BACKUP_DIR}")
    print(f"  Bboxes adicionados: {total_bboxes_adicionados}")
    print(f"  Total de labels no train: {total_labels}")
    print(f"\n  Próximo passo: retreinar o modelo")
    print("=" * 60)


if __name__ == "__main__":
    main()