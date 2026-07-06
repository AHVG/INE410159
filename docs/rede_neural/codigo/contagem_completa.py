"""
Contagem completa de cecropias por parcela + análise + figuras + relatório
==========================================================================
Autor: Gabriel A. Ferreira Gualda
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

PARCELAS_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/01_GEO/01_vetor/02_parcelas_MataDIV/MataDIV_PARCELAS_ESTRATOS.shp"
DETECCOES_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/05_MODEL/04_treinamento_labels_corrigidas_BEST/inference_BEST_val_test/shapefile/orto_MataDIV_8_4/pos_processing_inference/cecropia_detection_final_v1.gpkg"
SAM3_TEXT_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/01_GEO/01_vetor/05_copas_Cecropia_SAM3_MATADIV/copas_Cecropia_SAM3_MataDIV.shp"
SAM3_BBOX_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/04_IMAGES/teste_sistematico_sam3_bbox/300cm/cecropia_SAM3_bbox_300cm.gpkg"

PARCELA_ID_COL = "plot_id"
PLANTADAS_COL = "plantadas"
COMP_COL = "estrato"
MATCH_THRESHOLD = 0.1

OUTPUT_DIR = Path(r"F:/MESTRADO_Bilada/01_Mestrado/01_GEO/01_vetor/02_parcelas_MataDIV/resultados_analise")


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def assign_to_plot(gdf, parcelas, parcela_id_col, label="features"):
    if gdf.crs != parcelas.crs:
        gdf = gdf.to_crs(parcelas.crs)
    centroids = gdf.copy()
    centroids['geometry'] = gdf.geometry.centroid
    centroids['_orig_idx'] = range(len(centroids))
    joined = gpd.sjoin(centroids, parcelas[[parcela_id_col, 'geometry']],
                       how='left', predicate='within')
    dentro = joined[joined[parcela_id_col].notna()]
    fora = joined[joined[parcela_id_col].isna()]
    print(f"  {label}: {len(dentro)} dentro de parcelas, {len(fora)} fora")
    return joined


def find_matches(gdf_a, gdf_b, threshold=0.1):
    if gdf_a.crs != gdf_b.crs:
        gdf_b = gdf_b.to_crs(gdf_a.crs)
    matched_a = set()
    matched_b = set()
    sindex_b = gdf_b.sindex
    for idx_a, row_a in gdf_a.iterrows():
        geom_a = row_a.geometry
        area_a = geom_a.area
        possible = list(sindex_b.intersection(geom_a.bounds))
        for pos_idx in possible:
            idx_b = gdf_b.index[pos_idx]
            if idx_b in matched_b:
                continue
            geom_b = gdf_b.loc[idx_b, 'geometry']
            try:
                inter_area = geom_a.intersection(geom_b).area
                min_area = min(area_a, geom_b.area)
                if min_area > 0 and inter_area / min_area >= threshold:
                    matched_a.add(idx_a)
                    matched_b.add(idx_b)
                    break
            except Exception:
                continue
    return matched_a, matched_b


def calc_area_stats(gdf, parcelas, parcela_id_col, label="copa"):
    if gdf.crs != parcelas.crs:
        gdf = gdf.to_crs(parcelas.crs)
    gdf = gdf.copy()
    gdf['area_m2'] = gdf.geometry.area
    centroids = gdf.copy()
    centroids['geometry'] = gdf.geometry.centroid
    joined = gpd.sjoin(centroids, parcelas[[parcela_id_col, 'geometry']],
                       how='left', predicate='within')
    dentro = joined[joined[parcela_id_col].notna()]
    stats_parcela = dentro.groupby(parcela_id_col)['area_m2'].agg(
        area_media='mean', area_mediana='median',
        area_min='min', area_max='max', area_std='std'
    ).reset_index()
    print(f"\n  Estatísticas de área ({label}):")
    print(f"    N: {len(dentro)} | Média: {dentro['area_m2'].mean():.2f} m²")
    print(f"    Mediana: {dentro['area_m2'].median():.2f} m² | Mín: {dentro['area_m2'].min():.2f} m² | Máx: {dentro['area_m2'].max():.2f} m²")
    print(f"    Desvio padrão: {dentro['area_m2'].std():.2f} m²")
    return stats_parcela, dentro['area_m2']


# ==============================================================================
# FIGURAS
# ==============================================================================

def fig1_plantadas_vs_detectadas(resumo, comp_col, output_dir):
    """Barras agrupadas: plantadas vs detectadas por composição."""
    fig, ax = plt.subplots(figsize=(12, 6))
    comps = resumo[comp_col].values
    x = np.arange(len(comps))
    w = 0.35
    bars1 = ax.bar(x - w/2, resumo['plant'], w, label='Plantadas', color='#4C72B0', edgecolor='white')
    bars2 = ax.bar(x + w/2, resumo['total'], w, label='Detectadas', color='#55A868', edgecolor='white')
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, f'{int(h)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, f'{int(h)}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Composição')
    ax.set_ylabel('Número de Cecropia pachystachya')
    ax.set_title('Cecropias plantadas vs. detectadas por composição')
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "fig01_plantadas_vs_detectadas.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


def fig2_categorias_empilhadas(resumo, comp_col, output_dir):
    """Barras empilhadas: YOLO+SAM3, só YOLO, só SAM3 por composição."""
    fig, ax = plt.subplots(figsize=(12, 6))
    comps = resumo[comp_col].values
    x = np.arange(len(comps))
    # Calcular YOLO+SAM3 = YOLO - só YOLO... mas temos yolo e soSAM3
    # yolo inclui YOLO+SAM3 e só YOLO
    # Precisamos separar: yolo_com_sam3 = yolo - sam3b (aprox)
    # Na verdade: total = yolo + soSAM3, e yolo = yolo_com_sam3 + só_yolo
    # só_yolo ≈ sam3b (as que receberam bbox segmentation)
    # yolo_com_sam3 = yolo - sam3b
    resumo['yolo_com_sam3'] = resumo['yolo'] - resumo['sam3b']
    resumo['yolo_com_sam3'] = resumo['yolo_com_sam3'].clip(lower=0)
    
    b1 = ax.bar(x, resumo['yolo_com_sam3'], label='YOLO + SAM3 (ambos)', color='#4C72B0', edgecolor='white')
    b2 = ax.bar(x, resumo['sam3b'], bottom=resumo['yolo_com_sam3'], label='Só YOLO (novas)', color='#DD8452', edgecolor='white')
    b3 = ax.bar(x, resumo['soSAM3'], bottom=resumo['yolo_com_sam3'] + resumo['sam3b'],
                label='Só SAM3 (YOLO não detectou)', color='#55A868', edgecolor='white')
    
    # Valores no topo
    totals = resumo['yolo_com_sam3'] + resumo['sam3b'] + resumo['soSAM3']
    for i, t in enumerate(totals):
        if t > 0:
            ax.text(i, t + 3, f'{int(t)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Composição')
    ax.set_ylabel('Número de Cecropia pachystachya')
    ax.set_title('Categorias de detecção por composição')
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.legend(loc='upper right')
    fig.tight_layout()
    path = output_dir / "fig02_categorias_empilhadas.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


def fig3_histograma_areas(todas_areas, output_dir):
    """Histograma da distribuição de áreas das copas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(todas_areas, bins=40, color='#55A868', edgecolor='white', alpha=0.85)
    ax.axvline(todas_areas.mean(), color='#C44E52', linestyle='--', linewidth=2,
               label=f'Média: {todas_areas.mean():.2f} m²')
    ax.axvline(todas_areas.median(), color='#4C72B0', linestyle='--', linewidth=2,
               label=f'Mediana: {todas_areas.median():.2f} m²')
    ax.set_xlabel('Área da copa (m²)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de áreas das copas segmentadas de C. pachystachya')
    ax.legend()
    fig.tight_layout()
    path = output_dir / "fig03_histograma_areas.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


def fig4_boxplot_areas(all_masks, parcelas, parcela_id_col, comp_col, output_dir):
    """Boxplot de áreas das copas por composição."""
    if all_masks.crs != parcelas.crs:
        all_masks = all_masks.to_crs(parcelas.crs)
    masks = all_masks.copy()
    masks['area_m2'] = masks.geometry.area
    centroids = masks.copy()
    centroids['geometry'] = masks.geometry.centroid
    joined = gpd.sjoin(centroids, parcelas[[parcela_id_col, comp_col, 'geometry']],
                       how='left', predicate='within')
    dentro = joined[joined[comp_col].notna()]
    
    # Filtrar apenas composições com CP
    comps_com_cp = dentro.groupby(comp_col).size()
    comps_com_cp = comps_com_cp[comps_com_cp > 0].index.tolist()
    dados = dentro[dentro[comp_col].isin(comps_com_cp)]
    
    # Ordenar composições
    comp_order = sorted(comps_com_cp, key=lambda x: int(x.replace('D','').replace('d','')))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [dados[dados[comp_col] == c]['area_m2'].values for c in comp_order]
    bp = ax.boxplot(box_data, labels=comp_order, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                    medianprops=dict(color='#C44E52', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    ax.set_xlabel('Composição')
    ax.set_ylabel('Área da copa (m²)')
    ax.set_title('Distribuição de áreas das copas por composição')
    fig.tight_layout()
    path = output_dir / "fig04_boxplot_areas_composicao.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


def fig5_taxa_deteccao(resumo, comp_col, output_dir):
    """Barras: taxa de detecção (%) por composição."""
    # Filtrar apenas composições com plantadas > 0
    r = resumo[resumo['plant'] > 0].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comps = r[comp_col].values
    x = np.arange(len(comps))
    colors = ['#55A868' if t >= 50 else '#DD8452' if t >= 30 else '#C44E52' for t in r['taxa_%']]
    bars = ax.bar(x, r['taxa_%'], color=colors, edgecolor='white')
    for bar, val in zip(bars, r['taxa_%']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Composição')
    ax.set_ylabel('Taxa de detecção (%)')
    ax.set_title('Taxa de detecção (detectadas / plantadas) por composição')
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylim(0, max(r['taxa_%']) * 1.15)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.legend()
    fig.tight_layout()
    path = output_dir / "fig05_taxa_deteccao.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


def fig6_donut_categorias(yolo_com_sam3, yolo_sem_sam3, sam3_sem_yolo, output_dir):
    """Gráfico donut: proporção das 3 categorias."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [yolo_com_sam3, yolo_sem_sam3, sam3_sem_yolo]
    labels = [f'YOLO + SAM3\n({yolo_com_sam3})',
              f'Só YOLO\n({yolo_sem_sam3})',
              f'Só SAM3\n({sam3_sem_yolo})']
    colors = ['#4C72B0', '#DD8452', '#55A868']
    explode = (0.02, 0.02, 0.02)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      explode=explode, pctdistance=0.75,
                                      textprops={'fontsize': 12})
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_fontsize(11)
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)
    total = sum(sizes)
    ax.text(0, 0, f'Total\n{total}', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.set_title('Proporção das categorias de detecção', fontsize=14, pad=20)
    fig.tight_layout()
    path = output_dir / "fig06_donut_categorias.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {path.name}")


# ==============================================================================
# RELATÓRIO
# ==============================================================================

def gerar_relatorio(result, resumo, comp_col, yolo_com_sam3, yolo_sem_sam3, sam3_sem_yolo,
                    todas_areas, output_dir):
    """Gera relatório markdown com todos os resultados."""
    total_plantadas = result[PLANTADAS_COL].sum() if PLANTADAS_COL in result.columns else 0
    total_mapeadas = result['cp_total_mapeadas'].sum()
    total_segm = result['cp_segmentadas'].sum()
    total_yolo = result['cp_yolo'].sum()
    total_sam3t = result['cp_sam3_text'].sum()
    total_sam3b = result['cp_sam3_bbox'].sum()
    total_so_sam3 = result['cp_so_sam3'].sum()
    taxa_global = total_mapeadas / total_plantadas * 100 if total_plantadas > 0 else 0

    report = f"""# Relatório de Resultados — Contagem de Cecropia por Parcela
## Pipeline: SAM3 + YOLOv8x | MataDIV (ESALQ/USP)
**Autor:** Gabriel A. Ferreira Gualda
**Data:** {pd.Timestamp.now().strftime('%d/%m/%Y')}

---

## 1. Resumo Geral

| Métrica | Valor |
|---|---|
| Total de parcelas | {len(result)} |
| Parcelas com Cecropia (plantadas > 0) | {(result[PLANTADAS_COL] > 0).sum() if PLANTADAS_COL in result.columns else 'N/A'} |
| Total de CP plantadas | {int(total_plantadas)} |
| Total de CP mapeadas | {int(total_mapeadas)} |
| Taxa de detecção global | {taxa_global:.1f}% |
| Total com segmentação | {int(total_segm)} |

## 2. Categorias de Detecção

| Categoria | Quantidade | Proporção |
|---|---|---|
| YOLO + SAM3 (ambos) | {yolo_com_sam3} | {yolo_com_sam3/total_mapeadas*100:.1f}% |
| Só YOLO (novas) | {yolo_sem_sam3} | {yolo_sem_sam3/total_mapeadas*100:.1f}% |
| Só SAM3 (YOLO não detectou) | {sam3_sem_yolo} | {sam3_sem_yolo/total_mapeadas*100:.1f}% |
| **Total mapeadas** | **{int(total_mapeadas)}** | **100%** |

## 3. Detalhamento

| Fonte | Quantidade |
|---|---|
| Detecções YOLO (dentro parcelas) | {int(total_yolo)} |
| Máscaras SAM3 text prompt | {int(total_sam3t)} |
| Máscaras SAM3 bbox prompt | {int(total_sam3b)} |
| SAM3 que YOLO não detectou | {int(total_so_sam3)} |

## 4. Estatísticas de Área das Copas

| Estatística | Valor |
|---|---|
| N copas segmentadas | {len(todas_areas)} |
| Área média | {todas_areas.mean():.2f} m² |
| Área mediana | {todas_areas.median():.2f} m² |
| Área mínima | {todas_areas.min():.2f} m² |
| Área máxima | {todas_areas.max():.2f} m² |
| Desvio padrão | {todas_areas.std():.2f} m² |

## 5. Resultados por Composição

"""
    if resumo is not None:
        report += "| Comp | Parcelas | Plantadas | YOLO | só SAM3 | Total | Segm | Taxa (%) | Área média (m²) |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"
        for _, row in resumo.iterrows():
            area_str = f"{row.get('area_med', 0):.1f}" if not pd.isna(row.get('area_med', 0)) else "N/A"
            report += f"| {row[comp_col]} | {int(row['n_parc'])} | {int(row.get('plant', 0))} | {int(row['yolo'])} | {int(row['soSAM3'])} | {int(row['total'])} | {int(row['segm'])} | {row.get('taxa_%', 0):.1f} | {area_str} |\n"

    report += f"""

## 6. Figuras Geradas

1. `fig01_plantadas_vs_detectadas.png` — Barras: plantadas vs detectadas por composição
2. `fig02_categorias_empilhadas.png` — Barras empilhadas: 3 categorias por composição
3. `fig03_histograma_areas.png` — Histograma: distribuição de áreas das copas
4. `fig04_boxplot_areas_composicao.png` — Boxplot: áreas por composição
5. `fig05_taxa_deteccao.png` — Barras: taxa de detecção (%) por composição
6. `fig06_donut_categorias.png` — Donut: proporção das 3 categorias

---

*Relatório gerado automaticamente por `contar_cp_por_parcela.py`*
"""

    report_path = output_dir / "relatorio_resultados.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] {report_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CONTAGEM COMPLETA DE CECROPIAS POR PARCELA")
    print("=" * 70)

    # 1. Carregar
    print("\n[1/9] Carregando camadas...")
    parcelas = gpd.read_file(PARCELAS_PATH)
    deteccoes = gpd.read_file(DETECCOES_PATH)
    sam3_text = gpd.read_file(SAM3_TEXT_PATH)
    sam3_bbox = gpd.read_file(SAM3_BBOX_PATH)

    print(f"  Parcelas: {len(parcelas)}")
    print(f"  Detecções YOLO: {len(deteccoes)}")
    print(f"  Máscaras SAM3 text: {len(sam3_text)}")
    print(f"  Máscaras SAM3 bbox: {len(sam3_bbox)}")
    if PLANTADAS_COL in parcelas.columns:
        print(f"  Total plantadas: {parcelas[PLANTADAS_COL].sum()}")

    ref_crs = parcelas.crs
    if deteccoes.crs != ref_crs: deteccoes = deteccoes.to_crs(ref_crs)
    if sam3_text.crs != ref_crs: sam3_text = sam3_text.to_crs(ref_crs)
    if sam3_bbox.crs != ref_crs: sam3_bbox = sam3_bbox.to_crs(ref_crs)

    # 2. Match YOLO × SAM3
    print("\n[2/9] Cruzando YOLO × SAM3 text...")
    matched_yolo, matched_sam3 = find_matches(deteccoes, sam3_text, MATCH_THRESHOLD)
    yolo_com_sam3 = len(matched_yolo)
    yolo_sem_sam3 = len(deteccoes) - yolo_com_sam3
    sam3_sem_yolo = len(sam3_text) - len(matched_sam3)
    print(f"  YOLO+SAM3: {yolo_com_sam3} | Só YOLO: {yolo_sem_sam3} | Só SAM3: {sam3_sem_yolo}")

    # 3. Associar YOLO
    print("\n[3/9] Associando YOLO às parcelas...")
    yolo_joined = assign_to_plot(deteccoes, parcelas, PARCELA_ID_COL, "YOLO")
    yolo_por_parcela = yolo_joined.groupby(PARCELA_ID_COL).size().reset_index(name='cp_yolo')

    # 4. Associar SAM3 text
    print("\n[4/9] Associando SAM3 text às parcelas...")
    sam3_text_joined = assign_to_plot(sam3_text, parcelas, PARCELA_ID_COL, "SAM3 text")
    sam3_text_por_parcela = sam3_text_joined.groupby(PARCELA_ID_COL).size().reset_index(name='cp_sam3_text')

    sam3_sem_yolo_gdf = sam3_text.loc[~sam3_text.index.isin(matched_sam3)]
    sam3_sem_yolo_joined = assign_to_plot(sam3_sem_yolo_gdf, parcelas, PARCELA_ID_COL, "SAM3 sem YOLO")
    sam3_sem_yolo_parcela = sam3_sem_yolo_joined.groupby(PARCELA_ID_COL).size().reset_index(name='cp_so_sam3')

    # 5. Associar SAM3 bbox
    print("\n[5/9] Associando SAM3 bbox às parcelas...")
    sam3_bbox_joined = assign_to_plot(sam3_bbox, parcelas, PARCELA_ID_COL, "SAM3 bbox")
    sam3_bbox_por_parcela = sam3_bbox_joined.groupby(PARCELA_ID_COL).size().reset_index(name='cp_sam3_bbox')

    # 6. Áreas
    print("\n[6/9] Calculando áreas das copas...")
    all_masks = gpd.GeoDataFrame(pd.concat([sam3_text, sam3_bbox], ignore_index=True), crs=ref_crs)
    area_stats_parcela, todas_areas = calc_area_stats(all_masks, parcelas, PARCELA_ID_COL, "todas as máscaras")
    area_stats_parcela = area_stats_parcela.rename(columns={
        'area_media': 'copa_area_media_m2', 'area_mediana': 'copa_area_mediana_m2',
        'area_min': 'copa_area_min_m2', 'area_max': 'copa_area_max_m2',
        'area_std': 'copa_area_std_m2'
    })

    # 7. Montar tabela
    print("\n[7/9] Montando tabela final...")
    result = parcelas.copy()
    result = result.merge(yolo_por_parcela, on=PARCELA_ID_COL, how='left')
    result = result.merge(sam3_text_por_parcela, on=PARCELA_ID_COL, how='left')
    result = result.merge(sam3_bbox_por_parcela, on=PARCELA_ID_COL, how='left')
    result = result.merge(sam3_sem_yolo_parcela, on=PARCELA_ID_COL, how='left')
    result = result.merge(area_stats_parcela, on=PARCELA_ID_COL, how='left')

    for col in ['cp_yolo', 'cp_sam3_text', 'cp_sam3_bbox', 'cp_so_sam3']:
        result[col] = result[col].fillna(0).astype(int)

    result['cp_total_mapeadas'] = result['cp_yolo'] + result['cp_so_sam3']
    result['cp_segmentadas'] = result['cp_sam3_text'] + result['cp_sam3_bbox']

    if PLANTADAS_COL in result.columns:
        result['taxa_deteccao'] = np.where(
            result[PLANTADAS_COL] > 0,
            (result['cp_total_mapeadas'] / result[PLANTADAS_COL] * 100).round(1), 0)

    # Salvar
    result.to_file(OUTPUT_DIR / "parcelas_contagem_completa.gpkg", driver="GPKG")
    result.drop(columns=['geometry']).to_csv(OUTPUT_DIR / "parcelas_contagem_completa.csv",
                                              index=False, encoding='utf-8-sig')
    todas_areas.to_csv(OUTPUT_DIR / "areas_copas_m2.csv", index=False, header=['area_m2'])
    print(f"  ✅ Dados salvos em: {OUTPUT_DIR}")

    # Resumo por composição
    resumo = None
    if COMP_COL in result.columns:
        agg_dict = {
            'n_parc': (PARCELA_ID_COL, 'count'),
            'yolo': ('cp_yolo', 'sum'),
            'sam3t': ('cp_sam3_text', 'sum'),
            'sam3b': ('cp_sam3_bbox', 'sum'),
            'soSAM3': ('cp_so_sam3', 'sum'),
            'total': ('cp_total_mapeadas', 'sum'),
            'segm': ('cp_segmentadas', 'sum'),
            'area_med': ('copa_area_media_m2', 'mean'),
        }
        if PLANTADAS_COL in result.columns:
            agg_dict['plant'] = (PLANTADAS_COL, 'sum')
        resumo = result.groupby(COMP_COL).agg(**agg_dict).reset_index()
        if PLANTADAS_COL in result.columns:
            resumo['taxa_%'] = np.where(resumo['plant'] > 0,
                                        (resumo['total'] / resumo['plant'] * 100).round(1), 0)
        resumo.to_csv(OUTPUT_DIR / "resumo_por_composicao.csv", index=False, encoding='utf-8-sig')

    # 8. Figuras
    print("\n[8/9] Gerando figuras...")
    if resumo is not None and PLANTADAS_COL in result.columns:
        resumo_cp = resumo[resumo.get('plant', pd.Series(0)) > 0].copy()
        fig1_plantadas_vs_detectadas(resumo_cp, COMP_COL, OUTPUT_DIR)
        fig2_categorias_empilhadas(resumo_cp, COMP_COL, OUTPUT_DIR)
        fig5_taxa_deteccao(resumo_cp, COMP_COL, OUTPUT_DIR)

    fig3_histograma_areas(todas_areas, OUTPUT_DIR)
    fig4_boxplot_areas(all_masks, parcelas, PARCELA_ID_COL, COMP_COL, OUTPUT_DIR)
    fig6_donut_categorias(yolo_com_sam3, yolo_sem_sam3, sam3_sem_yolo, OUTPUT_DIR)

    # 9. Relatório
    print("\n[9/9] Gerando relatório...")
    gerar_relatorio(result, resumo, COMP_COL, yolo_com_sam3, yolo_sem_sam3,
                    sam3_sem_yolo, todas_areas, OUTPUT_DIR)

    # Resumo no terminal
    print("\n" + "=" * 70)
    print("  RESUMO FINAL")
    print("=" * 70)
    print(f"  YOLO+SAM3: {yolo_com_sam3} | Só YOLO: {yolo_sem_sam3} | Só SAM3: {sam3_sem_yolo}")
    print(f"  Total mapeadas: {result['cp_total_mapeadas'].sum()}")
    print(f"  Total segmentadas: {result['cp_segmentadas'].sum()}")
    if PLANTADAS_COL in result.columns:
        tp = result[PLANTADAS_COL].sum()
        tm = result['cp_total_mapeadas'].sum()
        print(f"  Plantadas: {int(tp)} | Taxa global: {tm/tp*100:.1f}%")
    print(f"\n  Resultados em: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()