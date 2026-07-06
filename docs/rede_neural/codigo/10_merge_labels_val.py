"""
10_merge_labels_val.py
========================
1. Faz BACKUP da pasta de labels original do val
2. Faz MERGE (append) dos labels novos curados com os existentes
3. Limpa cache do val

Autor: Gabriel A. Ferreira Gualda
"""

import os
import shutil


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Labels novos curados do val
LABELS_NOVOS_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/nao_anotadas_val/labels_novos"

# Labels originais do val
LABELS_VAL_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/val"

# Backup dos labels originais
LABELS_BACKUP_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/val_backup_original"

# Cache do val (precisa deletar para o YOLO reconhecer os novos labels)
VAL_CACHE = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/val.cache"


# ==============================================================================
# PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print("MERGE DE LABELS — VAL")
    print("=" * 60)

    # =========================================================================
    # 1. Backup
    # =========================================================================
    print("\n[1/3] Fazendo backup dos labels originais do val...")

    if os.path.exists(LABELS_BACKUP_DIR):
        print(f"  Backup já existe: {LABELS_BACKUP_DIR}")
    else:
        shutil.copytree(LABELS_VAL_DIR, LABELS_BACKUP_DIR)
        n_backup = len([f for f in os.listdir(LABELS_BACKUP_DIR) if f.endswith('.txt')])
        print(f"  Backup criado: {LABELS_BACKUP_DIR}")
        print(f"  Arquivos copiados: {n_backup}")

    # =========================================================================
    # 2. Merge
    # =========================================================================
    print("\n[2/3] Fazendo merge dos labels novos no val...")

    novos_files = [f for f in os.listdir(LABELS_NOVOS_DIR) if f.endswith('.txt')]
    print(f"  Labels novos (curados): {len(novos_files)} arquivos")

    total_bboxes_adicionados = 0
    arquivos_modificados = 0

    for filename in novos_files:
        novo_path = os.path.join(LABELS_NOVOS_DIR, filename)
        existente_path = os.path.join(LABELS_VAL_DIR, filename)

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
    for filename in os.listdir(LABELS_VAL_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(LABELS_VAL_DIR, filename)) as f:
                total_labels += sum(1 for line in f if line.strip())

    print(f"  Total de labels no val agora: {total_labels}")

    # =========================================================================
    # 3. Limpar cache
    # =========================================================================
    print("\n[3/3] Limpando cache...")

    if os.path.exists(VAL_CACHE):
        os.remove(VAL_CACHE)
        print(f"  Cache removido: {VAL_CACHE}")
    else:
        print(f"  Cache não encontrado (OK)")

    # Limpar cache do test também
    test_cache = VAL_CACHE.replace("val.cache", "test.cache")
    if os.path.exists(test_cache):
        os.remove(test_cache)
        print(f"  Cache test removido: {test_cache}")

    # Limpar cache do train
    train_cache = VAL_CACHE.replace("val.cache", "train.cache")
    if os.path.exists(train_cache):
        os.remove(train_cache)
        print(f"  Cache train removido: {train_cache}")

    print("\n" + "=" * 60)
    print("MERGE VAL CONCLUÍDO!")
    print("=" * 60)
    print(f"  Labels originais preservados em: {LABELS_BACKUP_DIR}")
    print(f"  Próximo passo: retreinar o modelo com labels corrigidos")
    print("=" * 60)


if __name__ == "__main__":
    main()