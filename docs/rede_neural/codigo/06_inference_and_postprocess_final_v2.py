"""
06_inference_ortomosaico.py
============================
Inferência do YOLOv8x treinado sobre ortomosaico georreferenciado
para detecção de Cecropia, gerando GeoPackage/Shapefile com as
bounding boxes georreferenciadas.

Pipeline:
  1. Recorta o ortomosaico em tiles de 896x896 px com overlap de 50%
  2. Roda a inferência YOLO em cada tile (normal + overlap)
  3. Converte bounding boxes de pixels para coordenadas geográficas
  4. Pós-processamento: remove detecções duplicadas entre tiles
  5. Exporta como GeoPackage e Shapefile

Ortomosaico: orto_MataDIV_8_4.tif (48264 x 47029 px, 8.46 GB)

Baseado no script de José Matheus S. M. Viveiros, adaptado para
detecção (sem máscaras) por Gabriel A. Ferreira Gualda.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Ortomosaico de entrada
MOSAICO_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/ortomosaico/mosaic_oficial/orto_MataDIV_8_4.tif"

# Modelo treinado (best.pt do treino definitivo)
MODEL_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/model_BEST_val_test/yolov8x_det_AdamW_corrigidos_BEST_val_test/weights/best.pt"

# Pasta de resultados
RESULTS_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/inference_BEST_val_test"

# Parâmetros de recorte
PATCH_SIZE = 896       # Mesmo tamanho usado no treino (893 → 896 múltiplo de 32)
OVERLAP = 448          # 50% de overlap entre tiles

# Confiança mínima para detecção
CONF_THRESHOLD = 0.25

# Margem de borda para filtrar detecções nas bordas do tile (pixels)
BORDER_OFFSET = 10


# ==============================================================================
# 1. RECORTAR ORTOMOSAICO EM TILES
# ==============================================================================
def recortar_mosaico(arquivo_entrada, pasta_tiles, patch_size, overlap):
    """
    Recorta o ortomosaico em tiles com overlap, mantendo georreferenciamento.
    Gera duas rodadas: tiles normais + tiles com offset (overlap).
    Retorna o CRS e o número total de tiles gerados.
    """
    print("\n[1/5] Recortando ortomosaico em tiles...")

    dataset = rasterio.open(arquivo_entrada)
    width = dataset.width
    height = dataset.height
    dtype = dataset.dtypes[0]
    crs = dataset.crs
    nbands = min(dataset.count, 3)  # Usar apenas 3 bandas (RGB)

    print(f"  Dimensões: {width} x {height} pixels")
    print(f"  Bandas: {dataset.count} (usando {nbands})")
    print(f"  Tile size: {patch_size} x {patch_size}")
    print(f"  Overlap: {overlap} px (50%)")

    step = patch_size - overlap
    count = 0

    # --- Tiles normais ---
    num_x = (width - overlap) // step
    num_y = (height - overlap) // step
    print(f"  Tiles normais: ~{num_x} x {num_y} = ~{num_x * num_y}")

    for j in tqdm(range(num_y), desc="  Tiles normais"):
        for i in range(num_x):
            xoff = i * step
            yoff = j * step
            xsize = min(patch_size, width - xoff)
            ysize = min(patch_size, height - yoff)

            # Pular tiles menores que o patch_size (bordas)
            if xsize < patch_size or ysize < patch_size:
                continue

            count += 1
            file_name = f"tile_{count:06d}.tif"
            file_path = os.path.join(pasta_tiles, file_name)

            new_transform = dataset.transform * Affine.translation(xoff, yoff)
            raster_crop = dataset.read(
                window=Window(xoff, yoff, xsize, ysize)
            )

            with rasterio.open(
                file_path, "w", driver="GTiff",
                width=xsize, height=ysize, count=nbands,
                dtype=dtype, crs=crs, transform=new_transform,
            ) as dest:
                dest.write(raster_crop[:nbands])

    normal_count = count
    print(f"  Tiles normais gerados: {normal_count}")

    # --- Tiles com overlap (offset de meio patch) ---
    for y in tqdm(range(patch_size, height, patch_size), desc="  Tiles overlap"):
        for x in range(patch_size, width, patch_size):
            xmin = x - patch_size // 2
            ymin = y - patch_size // 2
            xmax = xmin + patch_size
            ymax = ymin + patch_size

            if xmax > width or ymax > height:
                continue

            count += 1
            file_name = f"tile_{count:06d}_overlap.tif"
            file_path = os.path.join(pasta_tiles, file_name)

            new_transform = dataset.transform * Affine.translation(xmin, ymin)
            raster_crop = dataset.read(
                window=Window(xmin, ymin, patch_size, patch_size)
            )

            with rasterio.open(
                file_path, "w", driver="GTiff",
                width=patch_size, height=patch_size, count=nbands,
                dtype=dtype, crs=crs, transform=new_transform,
            ) as dest:
                dest.write(raster_crop[:nbands])

    overlap_count = count - normal_count
    print(f"  Tiles overlap gerados: {overlap_count}")
    print(f"  Total de tiles: {count}")

    dataset.close()
    return crs, count


# ==============================================================================
# 2. VERIFICAR SE BBOX ESTÁ NA BORDA DO TILE
# ==============================================================================
def check_image_bound(bbox, img_size, offset):
    """
    Retorna True se a bbox NÃO está na borda do tile.
    Detecções na borda provavelmente são cortadas e serão
    capturadas pelo tile vizinho.
    """
    x_min, y_min, x_max, y_max = bbox
    if x_min <= offset:
        return False
    if y_min <= offset:
        return False
    if x_max >= img_size - offset:
        return False
    if y_max >= img_size - offset:
        return False
    return True


# ==============================================================================
# 3. CONVERTER BBOX DE PIXELS PARA COORDENADAS GEOGRÁFICAS
# ==============================================================================
def bbox_to_geo_polygon(bbox, transform):
    """
    Converte bounding box em pixels para polígono georreferenciado.
    """
    x_min, y_min, x_max, y_max = bbox
    p1 = transform * (x_min, y_min)  # top-left
    p2 = transform * (x_max, y_min)  # top-right
    p3 = transform * (x_max, y_max)  # bottom-right
    p4 = transform * (x_min, y_max)  # bottom-left
    return Polygon([p1, p2, p3, p4])


# ==============================================================================
# 4. INFERÊNCIA NOS TILES
# ==============================================================================
def exec_inference(pasta_tiles, model, use_overlap=False):
    """
    Roda a inferência YOLO em todos os tiles.
    Retorna GeoDataFrame com bounding boxes georreferenciadas.
    """
    data = []
    id_counter = 1

    # Filtrar tiles
    all_files = sorted(os.listdir(pasta_tiles))
    if use_overlap:
        tile_files = [f for f in all_files if f.endswith("_overlap.tif")]
        label = "overlap"
    else:
        tile_files = [f for f in all_files if f.endswith(".tif") and not f.endswith("_overlap.tif")]
        label = "normal"

    print(f"\n  Inferência nos tiles {label}: {len(tile_files)} tiles")

    crs = None

    for filename in tqdm(tile_files, desc=f"  Inferência {label}"):
        file_path = os.path.join(pasta_tiles, filename)

        # Ler georreferenciamento do tile
        with rasterio.open(file_path) as ds:
            transform = ds.transform
            crs = ds.crs

        # Rodar YOLO
        results = model(
            file_path,
            conf=CONF_THRESHOLD,
            verbose=False,
            save=False,
        )

        # Extrair detecções
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        confs = results[0].boxes.conf.tolist()

        if len(boxes) > 0:
            for bbox, class_id, conf in zip(boxes, classes, confs):
                # Filtrar detecções na borda
                if check_image_bound(bbox, PATCH_SIZE, BORDER_OFFSET):
                    geo_polygon = bbox_to_geo_polygon(bbox, transform)
                    class_name = model.names[int(class_id)]

                    data.append({
                        "id": id_counter,
                        "geometry": geo_polygon,
                        "class": class_name,
                        "conf": conf,
                        "tile": filename,
                    })
                    id_counter += 1

    if len(data) > 0:
        gdf = gpd.GeoDataFrame(data, columns=["id", "geometry", "class", "conf", "tile"])
        gdf.crs = crs
    else:
        gdf = gpd.GeoDataFrame(columns=["id", "geometry", "class", "conf", "tile"])
        if crs:
            gdf.crs = crs

    print(f"  Detecções {label}: {len(data)}")
    return gdf


# ==============================================================================
# 5. PÓS-PROCESSAMENTO — REMOVER DUPLICATAS ENTRE TILES
# ==============================================================================
def remove_duplicates_between(gdf_normal, gdf_overlap, min_overlap_ratio=0.1):
    """
    Remove detecções duplicadas entre tiles normais e overlap.
    Mantém a detecção com maior confiança.
    """
    print("\n[4/5] Pós-processamento: removendo duplicatas entre tiles...")

    if len(gdf_normal) == 0 or len(gdf_overlap) == 0:
        print("  Nenhuma sobreposição para processar")
        return gdf_normal, gdf_overlap

    remove_normal = set()
    remove_overlap = set()

    gdf_overlap["orig_area"] = gdf_overlap.geometry.area
    gdf_normal["orig_area"] = gdf_normal.geometry.area

    for idx_o in tqdm(range(len(gdf_overlap)), desc="  Verificando sobreposições"):
        try:
            geom_o = gdf_overlap.iloc[idx_o].geometry
            area_o = gdf_overlap.iloc[idx_o]["orig_area"]
            conf_o = gdf_overlap.iloc[idx_o]["conf"]
            id_o = gdf_overlap.iloc[idx_o]["id"]

            # Encontrar interseções com tiles normais
            possible_matches = gdf_normal[gdf_normal.geometry.intersects(geom_o)]

            for idx_n, row_n in possible_matches.iterrows():
                intersection = row_n.geometry.intersection(geom_o)
                inter_area = intersection.area
                min_area = min(row_n["orig_area"], area_o)

                if inter_area >= min_area * min_overlap_ratio:
                    if row_n["conf"] >= conf_o:
                        remove_overlap.add(id_o)
                    else:
                        remove_normal.add(row_n["id"])
        except Exception as e:
            continue

    # Remover duplicatas
    gdf_normal = gdf_normal[~gdf_normal["id"].isin(remove_normal)]
    gdf_overlap = gdf_overlap[~gdf_overlap["id"].isin(remove_overlap)]

    # Limpar colunas auxiliares
    if "orig_area" in gdf_normal.columns:
        gdf_normal = gdf_normal.drop(columns=["orig_area"])
    if "orig_area" in gdf_overlap.columns:
        gdf_overlap = gdf_overlap.drop(columns=["orig_area"])

    print(f"  Removidas: {len(remove_normal)} normais + {len(remove_overlap)} overlap")
    print(f"  Restantes: {len(gdf_normal)} normais + {len(gdf_overlap)} overlap")

    return gdf_normal, gdf_overlap


def remove_duplicates_within(gdf, min_overlap_ratio=0.1):
    """
    Remove detecções duplicadas dentro do mesmo GeoDataFrame.
    Mantém a detecção com maior confiança.
    """
    print("  Removendo duplicatas internas...")

    if len(gdf) == 0:
        return gdf

    remove_ids = set()
    gdf["orig_area"] = gdf.geometry.area

    for idx in tqdm(range(len(gdf)), desc="  Verificando sobreposições internas"):
        try:
            row = gdf.iloc[idx]
            if row["id"] in remove_ids:
                continue

            geom = row.geometry
            area = row["orig_area"]
            conf = row["conf"]
            id_current = row["id"]

            # Encontrar interseções
            possible_matches = gdf[
                (gdf["id"] != id_current) &
                (~gdf["id"].isin(remove_ids)) &
                (gdf.geometry.intersects(geom))
            ]

            for _, row_m in possible_matches.iterrows():
                intersection = geom.intersection(row_m.geometry)
                inter_area = intersection.area
                min_area = min(area, row_m["orig_area"])

                if inter_area >= min_area * min_overlap_ratio:
                    if conf >= row_m["conf"]:
                        remove_ids.add(row_m["id"])
                    else:
                        remove_ids.add(id_current)
                        break
        except Exception:
            continue

    gdf = gdf[~gdf["id"].isin(remove_ids)]

    if "orig_area" in gdf.columns:
        gdf = gdf.drop(columns=["orig_area"])

    print(f"  Removidas internamente: {len(remove_ids)}")
    print(f"  Detecções finais: {len(gdf)}")

    return gdf


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("INFERÊNCIA EM ORTOMOSAICO — Detecção de Cecropia")
    print("=" * 60)
    print(f"  Ortomosaico: {MOSAICO_PATH}")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Tile size: {PATCH_SIZE} px")
    print(f"  Overlap: {OVERLAP} px (50%)")
    print(f"  Confiança mínima: {CONF_THRESHOLD}")
    print(f"  Resultados: {RESULTS_DIR}")
    print("=" * 60)

    # Criar pastas
    mosaico_name = os.path.splitext(os.path.basename(MOSAICO_PATH))[0]
    pasta_tiles = os.path.join(RESULTS_DIR, "tiles", mosaico_name)
    pasta_shapefile = os.path.join(RESULTS_DIR, "shapefile", mosaico_name)

    os.makedirs(pasta_tiles, exist_ok=True)
    os.makedirs(pasta_shapefile, exist_ok=True)

    # =========================================================================
    # ETAPA 1: Recortar mosaico
    # =========================================================================
    # Verificar se tiles já existem (para não recortar de novo)
    existing_tiles = [f for f in os.listdir(pasta_tiles) if f.endswith(".tif")]
    if len(existing_tiles) > 100:
        print(f"\n[1/5] Tiles já existem ({len(existing_tiles)} tiles). Pulando recorte...")
        with rasterio.open(MOSAICO_PATH) as ds:
            crs = ds.crs
    else:
        crs, total_tiles = recortar_mosaico(MOSAICO_PATH, pasta_tiles, PATCH_SIZE, OVERLAP)

    # =========================================================================
    # ETAPA 2: Carregar modelo
    # =========================================================================
    print("\n[2/5] Carregando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print(f"  Modelo carregado: {MODEL_PATH}")
    print(f"  Classes: {model.names}")

    # =========================================================================
    # ETAPA 3: Inferência
    # =========================================================================
    print("\n[3/5] Rodando inferência...")

    # Tiles normais
    gdf_normal = exec_inference(pasta_tiles, model, use_overlap=False)

    # Tiles com overlap
    gdf_overlap = exec_inference(pasta_tiles, model, use_overlap=True)

    # Salvar resultados brutos
    raw_dir = os.path.join(pasta_shapefile, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if len(gdf_normal) > 0:
        gdf_normal.to_file(os.path.join(raw_dir, "raw_normal_boxes.gpkg"), driver="GPKG")
        print(f"  Salvo: raw_normal_boxes.gpkg ({len(gdf_normal)} detecções)")

    if len(gdf_overlap) > 0:
        gdf_overlap.to_file(os.path.join(raw_dir, "raw_overlap_boxes.gpkg"), driver="GPKG")
        print(f"  Salvo: raw_overlap_boxes.gpkg ({len(gdf_overlap)} detecções)")

    # =========================================================================
    # ETAPA 4: Pós-processamento
    # =========================================================================
    # 4a: Remover duplicatas entre tiles normais e overlap
    gdf_normal, gdf_overlap = remove_duplicates_between(gdf_normal, gdf_overlap)

    # Combinar
    if len(gdf_normal) > 0 and len(gdf_overlap) > 0:
        gdf_all = gpd.GeoDataFrame(
            pd.concat([gdf_normal, gdf_overlap], ignore_index=True),
            crs=gdf_normal.crs
        )
    elif len(gdf_normal) > 0:
        gdf_all = gdf_normal
    elif len(gdf_overlap) > 0:
        gdf_all = gdf_overlap
    else:
        print("\n❌ Nenhuma detecção encontrada!")
        return

    # Salvar pós-processamento 1
    gdf_all.to_file(
        os.path.join(pasta_shapefile, "boxes_pos1_between_tiles.gpkg"),
        driver="GPKG"
    )
    print(f"  Salvo: boxes_pos1_between_tiles.gpkg ({len(gdf_all)} detecções)")

    # 4b: Remover duplicatas internas
    gdf_final = remove_duplicates_within(gdf_all)

    # =========================================================================
    # ETAPA 5: Salvar resultados finais
    # =========================================================================
    print("\n[5/5] Salvando resultados finais...")

    # GeoPackage (formato principal)
    gpkg_path = os.path.join(pasta_shapefile, "cecropia_detections_final.gpkg")
    gdf_final.to_file(gpkg_path, driver="GPKG")
    print(f"  ✅ GeoPackage: {gpkg_path}")

    # Shapefile (compatibilidade)
    shp_path = os.path.join(pasta_shapefile, "cecropia_detections_final.shp")
    gdf_final.to_file(shp_path, driver="ESRI Shapefile")
    print(f"  ✅ Shapefile: {shp_path}")

    # Resumo
    print("\n" + "=" * 60)
    print("INFERÊNCIA CONCLUÍDA!")
    print("=" * 60)
    print(f"  Total de Cecropias detectadas: {len(gdf_final)}")
    print(f"  Confiança média: {gdf_final['conf'].mean():.4f}")
    print(f"  Confiança mínima: {gdf_final['conf'].min():.4f}")
    print(f"  Confiança máxima: {gdf_final['conf'].max():.4f}")
    print(f"\n  Resultados em: {pasta_shapefile}")
    print(f"  Abra o arquivo .gpkg ou .shp no QGIS para visualizar!")
    print("=" * 60)


if __name__ == "__main__":
    main()