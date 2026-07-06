# =========================================
# CREATE A VECTOR FOR CUTTING OUT THE TILES
# =========================================

# import geopandas as gpd
# from shapely.geometry import box
# import fiona

# # Função para criar bounding boxes com dimensões x e y
# def create_bounding_box(polygon, x_dim, y_dim):
#     bounding_box = polygon.bounds
#     minx, miny, maxx, maxy = bounding_box
#     center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
#     half_x, half_y = x_dim / 2, y_dim / 2
#     new_minx, new_miny = center_x - half_x, center_y - half_y
#     new_maxx, new_maxy = center_x + half_x, center_y + half_y
#     return box(new_minx, new_miny, new_maxx, new_maxy)

# # Carregue o arquivo Shapefile original
# input_shapefile = 'C:/__PROJECTS/Scientific_works/Papers/Mother_Tree_Segmentation/01_data/vector/labels/corrigido/species_24_RGB_full.gpkg'
# gdf = gpd.read_file(input_shapefile)

# # Defina as dimensões x e y para as bounding boxes
# x_dim = 42  # Substitua pelo valor desejado
# y_dim = 42  # Substitua pelo valor desejado

# # Crie uma nova lista para armazenar as bounding boxes com 'id' correspondentes
# bounding_boxes = []

# # Itere sobre os polígonos no GeoDataFrame
# for idx, row in gdf.iterrows():
#     polygon = row['geometry']
#     bounding_box = create_bounding_box(polygon, x_dim, y_dim)
#     bounding_boxes.append((row['ID_Class'], bounding_box))

# # Crie um novo GeoDataFrame com as bounding boxes
# bbox_gdf = gpd.GeoDataFrame(bounding_boxes, columns=['ID_Class', 'geometry'], geometry='geometry')

# # Copie o sistema de referência do arquivo original para as bounding boxes
# bbox_gdf.crs = gdf.crs

# # Salve o novo GeoDataFrame como um arquivo Shapefile
# output_shapefile = 'C:/__PROJECTS/Scientific_works/Papers/Mother_Tree_Segmentation/01_data/vector/tiles/tiles_species_24_RGB_full.gpkg'
# bbox_gdf.to_file(output_shapefile, driver='GPKG')

# print(f'Bounding boxes salvas em {output_shapefile}')

# ===================================================================================================
# CREATE A VECTOR FOR CUTTING OUT THE TILES AND FILTER GEOMETRIES INVALID
# ===================================================================================================

import geopandas as gpd
from shapely.geometry import box

# =========================
# FUNÇÃO BOUNDING BOX
# =========================

def create_bounding_box(polygon, x_dim, y_dim):
    minx, miny, maxx, maxy = polygon.bounds

    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    half_x = x_dim / 2
    half_y = y_dim / 2

    new_minx = center_x - half_x
    new_miny = center_y - half_y
    new_maxx = center_x + half_x
    new_maxy = center_y + half_y

    return box(new_minx, new_miny, new_maxx, new_maxy)

# =========================
# LOAD DATA
# =========================

input_shapefile = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia.gpkg'  #CAMADA BBOX GERADOS NO QGIS
gdf = gpd.read_file(input_shapefile)

# =========================
# SPLIT DATA
# =========================

# Geometrias inválidas (None ou vazias)
gdf_geom_none = gdf[
    (gdf.geometry.isna()) | (gdf.geometry.is_empty)
]

# Geometrias válidas
gdf_valid = gdf[
    (gdf.geometry.notna()) & (~gdf.geometry.is_empty)
]

print(f'Total original: {len(gdf)}')
print(f'Geometrias válidas: {len(gdf_valid)}')
print(f'Geometrias inválidas (None/vazias): {len(gdf_geom_none)}')

# =========================
# SAVE SPLIT FILES
# =========================

output_valid = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia_FILTER.gpkg'
output_invalid = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia_GEO_NONE.gpkg'

gdf_valid.to_file(output_valid, driver='GPKG')
gdf_geom_none.to_file(output_invalid, driver='GPKG')

print(f'\nArquivo filtrado salvo em: {output_valid}')
print(f'Arquivo com erros salvo em: {output_invalid}')

# =========================
# CREATE BOUNDING BOXES
# =========================

x_dim = 7.5
y_dim = 7.5

bounding_boxes = []

for idx, row in gdf_valid.iterrows():
    polygon = row.geometry

    try:
        bbox = create_bounding_box(polygon, x_dim, y_dim)
        bounding_boxes.append((row['ID_Class'], bbox))
    except Exception as e:
        print(f"[ERRO] Índice {idx}: {e}")

# =========================
# CREATE GDF
# =========================

bbox_gdf = gpd.GeoDataFrame(
    bounding_boxes,
    columns=['ID_Class', 'geometry'],
    geometry='geometry',
    crs=gdf.crs
)

# =========================
# SAVE BBOX
# =========================

output_bbox = 'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia_FILTER_TILES.gpkg'

bbox_gdf.to_file(output_bbox, driver='GPKG')

print(f'\nBounding boxes salvas em: {output_bbox}')

# =============================
# SPLITTING DATASETS BY CLUSTER
# =============================

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import shutil

# Configurações de Caminhos
LABEL_PATH = r'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia_FILTER.gpkg'
TILE_PATH = r'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/Bbox_cecropia_FILTER_TILES.gpkg'
OUTPUT_DIR = r'F:/MESTRADO_Bilada/01_Mestrado/GEO/vetor/BBox_cecropia/split_dataset'

# Proporções Alvo
TARGET_PROPS = {'Train': 0.70, 'Val': 0.15, 'Test': 0.15}
OVERLAP_THRESHOLD = 0.20

def setup_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório criado: {output_dir}")
    else:
        print(f"Diretório de saída já existe: {output_dir}")

def get_connected_components(nodes, edges):
    """
    Implementação simples de componentes conexos (BFS) para evitar dependência de networkx.
    nodes: lista de ids de tiles
    edges: lista de tuplas (tile_id_a, tile_id_b)
    """
    adj_list = {node: [] for node in nodes}
    for u, v in edges:
        if u in adj_list: adj_list[u].append(v)
        if v in adj_list: adj_list[v].append(u)
    
    visited = set()
    components = []
    
    for node in nodes:
        if node not in visited:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in adj_list[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components

def perform_split():
    print("Carregando shapefiles...")
    labels = gpd.read_file(LABEL_PATH)
    tiles = gpd.read_file(TILE_PATH)
    
    # Garantir CRS consistente
    if labels.crs != tiles.crs:
        print("Avis: CRS divergente. Reprojentando labels para o CRS dos tiles.")
        labels = labels.to_crs(tiles.crs)
    
    # Resetar index para garantir unicidade
    labels = labels.reset_index(drop=True)

    # Identificador único para rótulos para facilitar rastreamento
    labels['label_idx'] = labels.index

    # Criar ID único para tiles se não existir
    if 'tile_id' not in tiles.columns:
        tiles['tile_id'] = range(len(tiles))
    
    # ---------------------------------------------------------
    # PASSO 1: Associação Primária (Label -> Tile "Dono")
    # Baseada na menor distância ao centroide (já validado anteriormente)
    # ---------------------------------------------------------
    print("Associando Rótulos ao 'Tile Dono' (Menor Distância ao Centroide)...")
    
    tile_centroids = tiles.copy()
    tile_centroids['geometry'] = tiles.centroid
    
    # Join espacial (labels dentro de tiles)
    joined_primary = gpd.sjoin(labels, tiles[['tile_id', 'geometry']], how='left', predicate='intersects') # intersects é mais seguro para pegar candidatas
    
    # Calcular distância para desambiguar
    joined_primary = joined_primary.join(tile_centroids.set_index('tile_id')['geometry'].rename('tile_center'), on='tile_id')
    joined_primary['dist'] = joined_primary.geometry.centroid.distance(joined_primary['tile_center'])
    
    # Selecionar o tile mais próximo para cada label
    # Ordenar por label_idx e distancia
    joined_primary = joined_primary.sort_values(['label_idx', 'dist'])
    # Manter a primeira ocorrência (menor distância)
    primary_association = joined_primary.drop_duplicates(subset=['label_idx'], keep='first')[['label_idx', 'tile_id']]
    primary_association = primary_association.rename(columns={'tile_id': 'primary_tile_id'})
    
    # Merge de volta no labels
    labels = labels.merge(primary_association, on='label_idx', how='left')
    
    # Filtrar labels órfãos
    missing = labels['primary_tile_id'].isna().sum()
    if missing > 0:
        print(f"AVISO: {missing} rótulos sem tile primário (não intersectam nenhum tile). Serão ignorados.")
        labels = labels.dropna(subset=['primary_tile_id'])
    
    # ---------------------------------------------------------
    # PASSO 2: Detecção de Vazamento Geométrico e Clusterização
    # Se Lab L intersecta Tile A (>20%) e Tile B (>20%), A e B devem estar juntos.
    # ---------------------------------------------------------
    print("Analisando sobreposições geométricas para clusterização...")
    
    # Precisamos calcular a intersecção EXATA de cada label com os tiles que ele toca
    # joined_primary já tem os candidatos (via intersects)
    # Vamos iterar apenas sobre os casos onde um rótulo toca múltiplos tiles
    
    # Contar quantos tiles cada label toca
    label_tile_counts = joined_primary.groupby('label_idx')['tile_id'].count()
    multi_tile_labels = label_tile_counts[label_tile_counts > 1].index
    
    print(f"Rótulos tocando múltiplos tiles: {len(multi_tile_labels)}")
    
    edges = set()
    
    # Subset para otimizar
    candidates = joined_primary[joined_primary['label_idx'].isin(multi_tile_labels)].copy()
    
    # Dicionário de geoms para acesso rápido
    label_geoms = labels.set_index('label_idx')['geometry']
    tile_geoms = tiles.set_index('tile_id')['geometry']
    
    # Para cada rótulo multi-tile, verificar área de interseção
    # Agrupar por label para processar
    # Note: pode demorar um pouco se houver muitos overlaps
    processing_count = 0
    total_candidates = len(multi_tile_labels)
    
    for lbl_idx, group in candidates.groupby('label_idx'):
        label_geom = label_geoms[lbl_idx]
        label_area = label_geom.area
        
        valid_tiles = []
        
        for tid in group['tile_id']:
            tile_geom = tile_geoms[tid]
            intersection_area = label_geom.intersection(tile_geom).area
            
            overlap_pct = intersection_area / label_area
            
            if overlap_pct > OVERLAP_THRESHOLD:
                valid_tiles.append(tid)
        
        # Se houver mais de 1 tile válido com significant overlap, eles formam conexões
        if len(valid_tiles) > 1:
            # Criar arestas (clique) entre todos os valid_tiles
            for i in range(len(valid_tiles)):
                for j in range(i + 1, len(valid_tiles)):
                    # Adicionar aresta não direcionada (menor, maior)
                    u, v = sorted((valid_tiles[i], valid_tiles[j]))
                    edges.add((u, v))
        
        processing_count += 1
        if processing_count % 100 == 0:
            print(f"Processando overlaps: {processing_count}/{total_candidates}...", end='\r')

    print(f"\nArestas criadas (conexões entre tiles): {len(edges)}")
    
    # Criar Clusters
    all_tile_ids = tiles['tile_id'].unique()
    clusters = get_connected_components(all_tile_ids, edges)
    
    print(f"Clusters formados: {len(clusters)}")
    
    # Mapear Tile -> Cluster ID
    tile_to_cluster = {}
    for c_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            tile_to_cluster[node] = c_id
            
    # Adicionar info de cluster no labels (via primary_tile_id)
    labels['cluster_id'] = labels['primary_tile_id'].map(tile_to_cluster)
    
    # ---------------------------------------------------------
    # PASSO 3: Split por Cluster (Otimização Monte Carlo)
    # ---------------------------------------------------------
    
    # Contagem de classes por Cluster
    # Assumindo coluna 'ID_Class'
    if 'ID_Class' not in labels.columns:
        raise ValueError("Coluna 'ID_Class' não executada.")
        
    cluster_class_counts = labels.groupby(['cluster_id', 'ID_Class']).size().unstack(fill_value=0)
    
    # Clusters vazios
    all_cluster_ids = list(range(len(clusters)))
    ids_with_labels = cluster_class_counts.index
    empty_clusters = np.setdiff1d(all_cluster_ids, ids_with_labels)
    
    if len(empty_clusters) > 0:
        empty_df = pd.DataFrame(0, index=empty_clusters, columns=cluster_class_counts.columns)
        cluster_class_counts = pd.concat([cluster_class_counts, empty_df])
    
    cluster_ids = cluster_class_counts.index.values
    classes = cluster_class_counts.columns.values
    total_counts = cluster_class_counts.sum()
    
    def try_one_split(seed_val, c_ids, c_counts, t_counts, t_props):
        """
        Executa uma única tentativa de split com uma seed específica.
        Retorna: splits (dict), split_counts (dict), global_error (float)
        """
        np.random.seed(seed_val)
        np.random.shuffle(c_ids)
        
        my_splits = {'Train': [], 'Val': [], 'Test': []}
        my_counts = {'Train': pd.Series(0, index=classes), 
                     'Val': pd.Series(0, index=classes), 
                     'Test': pd.Series(0, index=classes)}
        
        for cid in c_ids:
            current_c = c_counts.loc[cid]
            best_s = None
            max_deficit = -float('inf')
            
            for s_name in ['Train', 'Val', 'Test']:
                # Max Deficit
                c_ratio = my_counts[s_name] / t_counts.replace(0, 1)
                t_tgt = t_props[s_name]
                deficit = np.mean(t_tgt - c_ratio)
                
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_s = s_name
            
            my_splits[best_s].append(cid)
            my_counts[best_s] += current_c
            
        # Calcular Erro Global (MSE) desta tentativa
        # MSE = mean( (Target - Actual)^2 ) sobre todas as classes e splits
        errors = []
        for s_name in ['Train', 'Val', 'Test']:
            final_ratios = my_counts[s_name] / t_counts.replace(0, 1)
            target_r = t_props[s_name]
            sq_diff = (final_ratios - target_r) ** 2
            errors.extend(sq_diff.values)
            
        global_mse = np.mean(errors)
        return my_splits, my_counts, global_mse

    # Loop Monte Carlo
    num_attempts = 1000
    best_error = float('inf')
    best_config = None # (splits, counts, seed)
    
    print(f"Iniciando Otimização Monte Carlo com {num_attempts} tentativas...")
    
    for i in range(num_attempts):
        # Usar cópia dos IDs para não afetar outras iterações
        c_ids_copy = cluster_ids.copy()
        try_splits, try_counts, try_error = try_one_split(i, c_ids_copy, cluster_class_counts, total_counts, TARGET_PROPS)
        
        if try_error < best_error:
            best_error = try_error
            best_config = (try_splits, try_counts, i)
        
        if (i+1) % 100 == 0:
            print(f"Tentativa {i+1}/{num_attempts} - Melhor Erro: {best_error:.6f}...", end='\r')
            
    print(f"\nMelhor configuração encontrada na seed {best_config[2]} com MSE={best_error:.6f}")
    
    splits, split_counts, best_seed = best_config

    # ---------------------------------------------------------
    # PASSO 4: Salvar Outputs
    # ---------------------------------------------------------
    setup_output_dir(OUTPUT_DIR)
    
    report = []
    report.append("RELATÓRIO DE DIVISÃO (CLUSTER-BASED SPLIT REPORT)")
    report.append(f"Overlap Threshold: {OVERLAP_THRESHOLD*100}%")
    report.append("="*40)
    
    for split_name in ['Train', 'Val', 'Test']:
        selected_clusters = splits[split_name]
        
        # Obter tiles pertencentes a esses clusters
        # Mapeamento reverso manual ou iterar
        selected_tiles = [t for t, c in tile_to_cluster.items() if c in selected_clusters]
        
        # Filtrar Rótulos (pelo tile primário estar no set)
        split_labels = labels[labels['primary_tile_id'].isin(selected_tiles)].copy()
        
        # Remover colunas auxiliares antes de salvar
        if 'label_idx' in split_labels.columns: del split_labels['label_idx']
        if 'primary_tile_id' in split_labels.columns: del split_labels['primary_tile_id']
        if 'cluster_id' in split_labels.columns: del split_labels['cluster_id']

        # Filtrar Tiles Geometria
        split_tiles_geom = tiles[tiles['tile_id'].isin(selected_tiles)].copy()
        if 'tile_id' in split_tiles_geom.columns: del split_tiles_geom['tile_id'] # opcional, manter se util
        
        # Salvar
        l_out = os.path.join(OUTPUT_DIR, f"labels_{split_name.lower()}.shp")
        t_out = os.path.join(OUTPUT_DIR, f"tiles_{split_name.lower()}.shp")
        
        # Evitar salvar vazio se algum split ficar zerado (raro, mas possivel)
        if not split_labels.empty:
            split_labels.to_file(l_out)
        if not split_tiles_geom.empty:
            split_tiles_geom.to_file(t_out)
        
        count = len(split_labels)
        report.append(f"\nConjunto: {split_name}")
        report.append(f"Num. Clusters: {len(selected_clusters)}")
        report.append(f"Num. Tiles: {len(selected_tiles)}")
        report.append(f"Num. Rótulos: {count}")
        report.append("Proporções por Classe (vs Total Global):")
        
        final_counts = split_counts[split_name]
        percents = (final_counts / total_counts * 100).round(2)
        
        for cls_id, pct in percents.items():
            report.append(f"  Class {cls_id}: {pct}% (Meta: {TARGET_PROPS[split_name]*100}%)")
            
        report.append("-" * 20)

    report_path = os.path.join(OUTPUT_DIR, "split_report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
        
    print("\n".join(report))
    print(f"\nProcessamento concluído. Arquivos salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    perform_split()