# =======================
# CUTTING OUT THE MOSAICS
# =======================

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from pathlib import Path

# Pasta com os arquivos raster (arquivos .tif)
input_folder = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/ortomosaico/mosaic_oficial'
# Pasta com os buffers para recorte (arquivo .shp)
mask_shapefile = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/split_dataset/tiles_train.shp'
# Pasta para salvar os recortes
output_folder = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/images/train'

# Crie a pasta de saída se ela não existir
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Função para verificar e corrigir geometrias inválidas
def fix_invalid_geometries(gdf):
    # Verifique geometrias inválidas
    invalid_geoms = gdf[~gdf.is_valid]

    # Corrija geometrias inválidas
    gdf.loc[~gdf.is_valid, 'geometry'] = invalid_geoms.buffer(0)

    return gdf

# Abra a camada de máscara
mask_gdf = gpd.read_file(mask_shapefile)

# Liste todos os arquivos .tif na pasta de entrada
tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# Loop através dos arquivos .tif na pasta de entrada
for tif_file in tif_files:
    input_raster = os.path.join(input_folder, tif_file)

    with rasterio.open(input_raster) as src:
        # Encontre a extensão do raster
        raster_extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

        # Loop através dos polígonos na camada de máscara
        for idx, row in mask_gdf.iterrows():
            # Verifique a interseção entre o polígono da máscara e a extensão do raster
            if row['geometry'].intersects(raster_extent):
                # Recorte a imagem usando o polígono como máscara
                out_image, out_transform = mask(src, [row['geometry']], crop=True)

                # Copie os metadados do raster original
                out_meta = src.meta.copy()

                # Atualize a transformação
                out_meta.update({'transform': out_transform, 'height': out_image.shape[1], 'width': out_image.shape[2]})

                # Nome para o arquivo de saída
                #output_filename = os.path.join(output_folder, f'clip_{tif_file}_{row["class"]}_{idx + 1}.tif')
                output_filename = os.path.join(output_folder, f'clip_{os.path.splitext(tif_file)[0]}_{row["ID_Class"]}_{idx + 1}.tif')

                # Salve o recorte como uma nova imagem .tif
                with rasterio.open(output_filename, 'w', **out_meta) as dest:
                    dest.write(out_image)

                print(f'Recorte para id {row["ID_Class"]}, sequência {idx + 1} salvo em {output_filename}')

# ========================================
# CREATE A TEXT FILE WITH MASK ANNOTATIONS
# ========================================

import os
import geopandas as gpd
import rasterio
import numpy as np
from pathlib import Path
from shapely.geometry import box
from rasterio.features import geometry_mask
import decimal

# Pasta com os arquivos raster (arquivos .tif)
input_folder = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/images/train'
# Pasta com a camada de máscara (arquivo .shp)
mask_shapefile = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/split_dataset/labels_train.shp'
# Pasta para salvar as anotações em texto
output_folder = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/labels/train'

# Crie a pasta de saída se ela não existir
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Abra a camada de máscara
mask_gdf = gpd.read_file(mask_shapefile)

# Função para verificar e corrigir geometrias inválidas
def fix_invalid_geometries(gdf):
    invalid_geoms = gdf[~gdf.is_valid]
    gdf.loc[~gdf.is_valid, 'geometry'] = invalid_geoms.buffer(0)
    return gdf

# Liste todos os arquivos .tif na pasta de entrada
tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# create a new context for this task
ctx = decimal.Context()
# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

# Loop através dos arquivos .tif na pasta de entrada
for tif_file in tif_files:
    input_raster = os.path.join(input_folder, tif_file)

    with rasterio.open(input_raster) as src:
        # Encontre a extensão do raster como uma geometria
        raster_extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

        # Inicialize um array para a banda de 'id' das máscaras
        new_id_band = np.zeros((src.height, src.width), dtype=np.uint8)

        # Lista para armazenar anotações
        annotations = []

        # Loop através dos polígonos na camada de máscara
        for idx, row in mask_gdf.iterrows():
            geom = row['geometry']

            if geom.overlaps(raster_extent) or geom.within(raster_extent):

                # Normalize as coordenadas entre 0 e 1
                normalized_coords = np.array(geom.exterior.xy) - np.array([src.bounds.left, src.bounds.bottom]).reshape(-1, 1)
                size = np.array([src.bounds.right - src.bounds.left, src.bounds.top - src.bounds.bottom]).reshape(-1, 1)

                normalized_coords /= size

                normalized_coords[1, :] = 1 - normalized_coords[1, :]

                normalized_coords = normalized_coords.clip(0, 1)
                #print(normalized_coords.transpose().flatten().tolist())
                #raise Exception

                # Adicione a anotação à lista
                annotation = [row['ID_Class']] + normalized_coords.transpose().flatten().tolist()
                annotations.append(annotation)

        # Crie um novo arquivo de texto com as anotações
        #change_extension = lambda x : x[:-2] + 'xt'
        txt_file_path = os.path.join(output_folder, tif_file[:-4] + ".txt")
        with open(txt_file_path, 'w') as txt_file:
            for annotation in annotations:
                txt_file.write(' '.join(map(float_to_str, annotation)) + '\n')

        print(f'Anotações salvas em {txt_file_path}')

# ================
# CREATE YAML FILE
# ================

yaml_data = """
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset'  # dataset root dir
train: 'images/train'  # train images (relative to 'path') 4 images
val: 'images/val'  # val images (relative to 'path') 4 images
test: 'images/test' # (optional)

# Classes
names:
  0: Cecropia pachystachya

"""
save_dir = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/Bbox_cecropia/dataset/model_dataset_structure.yaml'

with open(save_dir, "w") as f:
  f.write(yaml_data)