"""
11_pos_processamento_inferencia.py
====================================
Pós-processamento das detecções de Cecropia:
  1. Remove ruído: bbox grande (area_m2 > 30) com baixa confiança (conf < 0.51)
  2. Remove ruído: bbox pequeno (area_m2 < 2) com baixa confiança (conf < 0.51)
  3. Exporta GPKG e SHP limpos

Autor: Gabriel A. Ferreira Gualda
"""

import geopandas as gpd
import os


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Arquivo de entrada (detecções brutas da inferência)
INPUT_SHP = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/inference_BEST_val_test/shapefile/orto_MataDIV_8_4/cecropia_detections_final - Copia.gpkg"

# Pasta de saída
OUTPUT_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/inference_BEST_val_test/shapefile/orto_MataDIV_8_4"

# Limiares
CONF_THRESHOLD = 0.51
AREA_MAX = 30  # m²
AREA_MIN = 2   # m²


# ==============================================================================
# PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print("PÓS-PROCESSAMENTO — Filtragem de Detecções")
    print("=" * 60)

    # =========================================================================
    # 1. Carregar detecções
    # =========================================================================
    print("\n[1/4] Carregando detecções...")
    gdf = gpd.read_file(INPUT_SHP)
    total_inicial = len(gdf)
    print(f"  Total de detecções: {total_inicial}")
    print(f"  Confiança média: {gdf['conf'].mean():.4f}")
    print(f"  Área média: {gdf['area_m2'].mean():.2f} m²")

    # =========================================================================
    # 2. Regra 1: area > 30 m² E conf < 0.51 → ruído
    # =========================================================================
    print(f"\n[2/4] Regra 1: area_m2 > {AREA_MAX} E conf < {CONF_THRESHOLD}")
    mask_grande = (gdf['area_m2'] > AREA_MAX) & (gdf['conf'] < CONF_THRESHOLD)
    n_grande = mask_grande.sum()
    print(f"  Detecções removidas (bbox grande + baixa conf): {n_grande}")

    # =========================================================================
    # 3. Regra 2: area < 2 m² E conf < 0.51 → ruído
    # =========================================================================
    print(f"\n[3/4] Regra 2: area_m2 < {AREA_MIN} E conf < {CONF_THRESHOLD}")
    mask_pequeno = (gdf['area_m2'] < AREA_MIN) & (gdf['conf'] < CONF_THRESHOLD)
    n_pequeno = mask_pequeno.sum()
    print(f"  Detecções removidas (bbox pequeno + baixa conf): {n_pequeno}")

    # =========================================================================
    # 4. Aplicar filtros e salvar
    # =========================================================================
    print("\n[4/4] Aplicando filtros e salvando...")

    # Combinar máscaras (remover onde qualquer regra é verdadeira)
    mask_remover = mask_grande | mask_pequeno
    gdf_filtrado = gdf[~mask_remover].copy()

    # Resetar index
    gdf_filtrado = gdf_filtrado.reset_index(drop=True)

    total_removido = total_inicial - len(gdf_filtrado)
    total_final = len(gdf_filtrado)

    # Salvar GPKG
    gpkg_path = os.path.join(OUTPUT_DIR, "cecropia_detections_filtered.gpkg")
    gdf_filtrado.to_file(gpkg_path, driver="GPKG")
    print(f"  GPKG: {gpkg_path}")

    # Salvar SHP
    shp_path = os.path.join(OUTPUT_DIR, "cecropia_detections_filtered.shp")
    gdf_filtrado.to_file(shp_path, driver="ESRI Shapefile")
    print(f"  SHP: {shp_path}")

    # =========================================================================
    # Resumo
    # =========================================================================
    print("\n" + "=" * 60)
    print("PÓS-PROCESSAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"  Detecções iniciais:  {total_inicial}")
    print(f"  Removidas regra 1 (grande + baixa conf): {n_grande}")
    print(f"  Removidas regra 2 (pequeno + baixa conf): {n_pequeno}")
    print(f"  Total removidas:     {total_removido}")
    print(f"  Detecções finais:    {total_final}")
    print(f"\n  Confiança média (filtrado): {gdf_filtrado['conf'].mean():.4f}")
    print(f"  Área média (filtrado): {gdf_filtrado['area_m2'].mean():.2f} m²")
    print(f"  Área min: {gdf_filtrado['area_m2'].min():.2f} m²")
    print(f"  Área max: {gdf_filtrado['area_m2'].max():.2f} m²")
    print("=" * 60)


if __name__ == "__main__":
    main()