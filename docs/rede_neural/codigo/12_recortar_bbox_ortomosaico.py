"""
12_recortar_bbox_ortomosaico.py
=================================
Recorta o ortomosaico nas regiões das detecções de Cecropia,
gerando GeoTIFFs individuais com padding para segmentação SAM.
Também gera cópias em PNG para visualização rápida.

Cada recorte:
  - GeoTIFF georreferenciado (para SAM2/SAM3)
  - PNG para visualização rápida
  - Metadados salvos em JSON (id, conf, area, bbox no recorte)

Autor: Gabriel A. Ferreira Gualda
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image
from tqdm import tqdm
import json


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Ortomosaico
MOSAICO_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/01_GEO/02_ortomosaico/01_mosaic_OFICIAL/orto_MataDIV_8_4.tif"

# Detecções filtradas (GPKG)
DETECTIONS_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/05_MODEL/04_treinamento_labels_corrigidas_BEST/inference_BEST_val_test/shapefile/orto_MataDIV_8_4/pos_processing_inference/cecropia_detection_no_mask_V1.shp"

# Pasta de saída dos recortes
OUTPUT_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/04_IMAGES/recortes_bbox_orto_350cm"

# Padding em metros ao redor da bbox
PADDING_METERS = 3.50


# ==============================================================================
# PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print("RECORTE DE BBOX DO ORTOMOSAICO PARA SAM")
    print("=" * 60)

    # Criar pastas
    tif_dir = os.path.join(OUTPUT_DIR, "tif")
    png_dir = os.path.join(OUTPUT_DIR, "png")
    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # =========================================================================
    # 1. Carregar detecções
    # =========================================================================
    print("\n[1/3] Carregando detecções...")
    gdf = gpd.read_file(DETECTIONS_PATH)
    print(f"  Total de detecções: {len(gdf)}")
    print(f"  Colunas: {list(gdf.columns)}")

    # =========================================================================
    # 2. Abrir ortomosaico
    # =========================================================================
    print("\n[2/3] Abrindo ortomosaico...")
    dataset = rasterio.open(MOSAICO_PATH)
    transform = dataset.transform
    crs = dataset.crs
    width = dataset.width
    height = dataset.height
    nbands = min(dataset.count, 3)

    # Resolução em metros por pixel
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    padding_px_x = int(PADDING_METERS / res_x)
    padding_px_y = int(PADDING_METERS / res_y)

    print(f"  Dimensões: {width} x {height} pixels")
    print(f"  Resolução: {res_x:.4f} x {res_y:.4f} m/px")
    print(f"  Padding: {PADDING_METERS} m = {padding_px_x} x {padding_px_y} px")
    print(f"  Bandas: {dataset.count} (usando {nbands})")
    print(f"  Saída TIF: {tif_dir}")
    print(f"  Saída PNG: {png_dir}")

    # =========================================================================
    # 3. Recortar cada bbox
    # =========================================================================
    print(f"\n[3/3] Recortando {len(gdf)} detecções...")

    metadata_list = []
    erros = 0

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  Recortando"):
        try:
            geom = row.geometry
            bbox = geom.bounds  # (minx, miny, maxx, maxy)

            # Converter coordenadas geográficas para pixels
            col_min, row_max = ~transform * (bbox[0], bbox[1])
            col_max, row_min = ~transform * (bbox[2], bbox[3])

            # Converter para int e adicionar padding
            row_min = int(row_min) - padding_px_y
            row_max = int(row_max) + padding_px_y
            col_min = int(col_min) - padding_px_x
            col_max = int(col_max) + padding_px_x

            # Garantir limites dentro do ortomosaico
            row_min = max(0, row_min)
            row_max = min(height, row_max)
            col_min = max(0, col_min)
            col_max = min(width, col_max)

            # Dimensões do recorte
            win_width = col_max - col_min
            win_height = row_max - row_min

            if win_width <= 0 or win_height <= 0:
                erros += 1
                print(f"  ERRO idx={idx}: dimensao zero ({win_width}x{win_height})")
                continue

            # Ler pixels
            window = Window(col_min, row_min, win_width, win_height)
            raster_crop = dataset.read(
                indexes=list(range(1, nbands + 1)),
                window=window
            )

            # Nova transformada affine para o recorte
            new_transform = rasterio.windows.transform(window, transform)

            # IDs e metadados — converte para int/float de forma segura
            det_id = row.get('id', idx)
            try:
                det_id = int(det_id)
            except (ValueError, TypeError):
                det_id = idx

            conf = row.get('conf', 0)
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.0

            area = row.get('area_m2', 0)
            try:
                area = float(area)
            except (ValueError, TypeError):
                area = 0.0

            base_name = f"bbox_{det_id:04d}_conf{conf:.2f}"

            # ----- Salvar GeoTIFF -----
            tif_path = os.path.join(tif_dir, base_name + ".tif")
            with rasterio.open(
                tif_path, "w", driver="GTiff",
                width=win_width, height=win_height,
                count=nbands, dtype=raster_crop.dtype,
                crs=crs, transform=new_transform,
            ) as dst:
                dst.write(raster_crop)

            # ----- Salvar PNG -----
            png_path = os.path.join(png_dir, base_name + ".png")
            img_rgb = np.moveaxis(raster_crop[:3], 0, -1)
            Image.fromarray(img_rgb).save(png_path)

            # ----- Metadados -----
            bbox_rel_col_min = max(0, padding_px_x)
            bbox_rel_row_min = max(0, padding_px_y)
            bbox_rel_col_max = min(win_width, win_width - padding_px_x)
            bbox_rel_row_max = min(win_height, win_height - padding_px_y)

            metadata_list.append({
                "id": det_id,
                "filename": base_name,
                "conf": round(conf, 4),
                "area_m2": round(area, 2),
                "crop_width": win_width,
                "crop_height": win_height,
                "bbox_in_crop": [
                    bbox_rel_col_min,
                    bbox_rel_row_min,
                    bbox_rel_col_max,
                    bbox_rel_row_max
                ],
                "geo_bounds": [
                    round(bbox[0], 6),
                    round(bbox[1], 6),
                    round(bbox[2], 6),
                    round(bbox[3], 6)
                ],
            })

        except Exception as e:
            erros += 1
            if erros <= 10:
                print(f"  ERRO idx={idx}: {e}")
            continue

    dataset.close()

    # Salvar metadados em JSON
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    # =========================================================================
    # Resumo
    # =========================================================================
    print("\n" + "=" * 60)
    print("RECORTE CONCLUÍDO!")
    print("=" * 60)
    print(f"  Recortes gerados: {len(metadata_list)}")
    print(f"  Erros: {erros}")
    print(f"  Padding: {PADDING_METERS} m")
    print(f"  GeoTIFFs: {tif_dir}")
    print(f"  PNGs: {png_dir}")
    print(f"  Metadados: {meta_path}")
    print(f"\n  Próximo passo: rodar SAM nos recortes")
    print("=" * 60)


if __name__ == "__main__":
    main()