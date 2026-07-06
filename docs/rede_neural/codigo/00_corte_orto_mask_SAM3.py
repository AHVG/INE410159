"""
=============================================================================
PIPELINE COMPLETO: Ortomosaico → Tiles → SAM 3 ("tree") → Shapefile
Tudo em um único script
=============================================================================

COMO RODAR:
  python pipeline_orto_sam3_copas.py

O QUE FAZ:
  1. Lê o ortomosaico GeoTIFF
  2. Recorta em tiles (4096x4096 para menos cortes nas bordas)
  3. Segmenta todas as copas com SAM 3 (prompt: "tree")
  4. Converte máscaras em polígonos georreferenciados
  5. Exporta como Shapefile + GeoPackage para QGIS
=============================================================================
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import Sam3Processor, Sam3Model

# =============================================================================
# CONFIGURAÇÕES — ALTERE AQUI
# =============================================================================

# Caminho do ortomosaico GeoTIFF
ORTOMOSAICO_PATH = r"F:/MESTRADO_Bilada/01_Mestrado/01_GEO/02_ortomosaico/01_mosaic_OFICIAL/orto_MataDIV_8_4.tif"

# Pasta de saída (tiles + resultados)
OUTPUT_DIR = r"F:/MESTRADO_Bilada/01_Mestrado/pipeline_completo"

# Tamanho do tile (maior = menos cortes nas bordas, mas mais VRAM)
# SAM 3 redimensiona internamente para 1024x1024
# Com 16 GB VRAM, 4096 deve funcionar. Se der OOM, reduza para 2048.
TAMANHO_TILE = 4096

# Sobreposição entre tiles (em pixels)
SOBREPOSICAO = 1024

# Prompt de texto para segmentar
TEXT_PROMPT = "tree"

# Thresholds
THRESHOLD = 0.2       # Baixo para pegar máximo de copas
MASK_THRESHOLD = 0.4

# NODATA: porcentagem máxima de pixels pretos para processar o tile
MAX_NODATA_RATIO = 0.30

# Salvar tiles recortados como imagens JPG
SALVAR_TILES = True

# Salvar imagens anotadas com segmentações
SALVAR_IMAGENS_ANOTADAS = False


# =============================================================================
# ETAPA 1: RECORTAR ORTOMOSAICO EM TILES
# =============================================================================

def recortar_ortomosaico(caminho_orto, pasta_saida, tamanho_tile, sobreposicao):
    """Recorta o ortomosaico em tiles com sobreposição."""
    import rasterio
    from rasterio.windows import Window

    pasta_tiles = os.path.join(pasta_saida, "tiles")
    os.makedirs(pasta_tiles, exist_ok=True)

    print("=" * 70)
    print("  ETAPA 1: Recortando ortomosaico em tiles")
    print("=" * 70)

    with rasterio.open(caminho_orto) as src:
        largura_total = src.width
        altura_total = src.height
        transform_global = src.transform
        crs = src.crs

        print(f"  📷 Ortomosaico: {largura_total} x {altura_total} pixels")
        print(f"  📐 CRS: {crs}")
        print(f"  📏 Tile: {tamanho_tile} x {tamanho_tile} px | Overlap: {sobreposicao} px")

        passo = tamanho_tile - sobreposicao
        metadados_tiles = []
        tile_id = 0

        for y in range(0, altura_total, passo):
            for x in range(0, largura_total, passo):
                w = min(tamanho_tile, largura_total - x)
                h = min(tamanho_tile, altura_total - y)

                if w < 256 or h < 256:
                    continue

                window = Window(x, y, w, h)
                tile_data = src.read(window=window)

                # Converter para RGB
                if tile_data.shape[0] >= 3:
                    rgb = tile_data[:3]
                else:
                    rgb = np.stack([tile_data[0]] * 3)

                rgb = np.moveaxis(rgb, 0, -1)

                # Salvar tile como JPG
                nome_tile = f"tile_{tile_id:04d}.jpg"
                caminho_tile = os.path.join(pasta_tiles, nome_tile)

                if SALVAR_TILES:
                    img = Image.fromarray(rgb.astype(np.uint8))
                    img.save(caminho_tile, quality=95)

                # Metadados de georreferência
                tile_transform = rasterio.windows.transform(window, transform_global)
                metadados_tiles.append({
                    "tile_id": tile_id,
                    "arquivo": nome_tile,
                    "x_pixel": x,
                    "y_pixel": y,
                    "largura": w,
                    "altura": h,
                    "transform": list(tile_transform)[:6],
                })

                tile_id += 1

        # Salvar metadados
        meta = {
            "crs": str(crs),
            "transform_global": list(transform_global)[:6],
            "tamanho_tile": tamanho_tile,
            "sobreposicao": sobreposicao,
            "total_tiles": tile_id,
            "tiles": metadados_tiles,
        }

        meta_path = os.path.join(pasta_tiles, "metadados_tiles.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  ✅ {tile_id} tiles criados em: {pasta_tiles}")
        print(f"  ✅ Metadados salvos em: {meta_path}")

    return meta, pasta_tiles


# =============================================================================
# ETAPA 2: SEGMENTAR TODAS AS COPAS COM SAM 3
# =============================================================================

def carregar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📌 Dispositivo: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("⏳ Carregando SAM 3...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    print("✅ Modelo carregado!")
    return model, processor, device


def verificar_nodata(tile_path, max_ratio=0.30):
    """Verifica se o tile tem muita área NODATA (pixels pretos)."""
    img = Image.open(tile_path).convert("RGB")
    arr = np.array(img)
    preto = np.all(arr <= 5, axis=2)
    ratio = np.sum(preto) / preto.size
    return ratio < max_ratio, ratio


def segmentar_tile_texto(model, processor, device, tile_path, texto):
    """Segmenta um tile usando prompt de texto."""
    image = Image.open(tile_path).convert("RGB")

    inputs = processor(
        images=image,
        text=texto,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    return image, results


def masks_para_poligonos(masks, boxes, scores):
    """Converte máscaras em polígonos com métricas."""
    import cv2

    poligonos = []

    for i in range(len(masks)):
        mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else np.array(masks[i])
        if mask.ndim == 3:
            mask = mask[0]

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        coords = contour.squeeze().tolist()
        if len(coords) < 3:
            continue

        score = scores[i].item() if torch.is_tensor(scores[i]) else float(scores[i])
        box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else np.array(boxes[i])

        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)
        circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0

        poligonos.append({
            "coords_pixel": coords,
            "score": score,
            "bbox": box[:4].tolist(),
            "area_pixels": area,
            "perimetro_pixels": perimetro,
            "circularidade": circularidade,
        })

    return poligonos


def salvar_imagem_anotada(image, results, titulo, caminho_saida):
    masks = results.get("masks", [])
    boxes = results.get("boxes", [])

    if len(masks) == 0:
        return

    plt.figure(figsize=(12, 12))
    plt.imshow(image)

    np.random.seed(42)
    cores = np.random.random((len(masks), 4))
    cores[:, 3] = 0.5

    for i in range(len(masks)):
        mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else np.array(masks[i])
        if mask.ndim == 3:
            mask = mask[0]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0.5] = cores[i]
        plt.imshow(overlay)

        if i < len(boxes):
            box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else np.array(boxes[i])
            x1, y1, x2, y2 = box[:4]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor=cores[i][:3], facecolor='none'
            )
            plt.gca().add_patch(rect)

    plt.title(f"{titulo} — {len(masks)} copa(s)", fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=100, bbox_inches='tight')
    plt.close()


# =============================================================================
# ETAPA 3: EXPORTAR COMO SHAPEFILE
# =============================================================================

def pixels_para_geo(coords_pixel, transform):
    from rasterio.transform import Affine
    if isinstance(transform, list):
        transform = Affine(*transform[:6])

    coords_geo = []
    for x_pixel, y_pixel in coords_pixel:
        x_geo, y_geo = transform * (x_pixel, y_pixel)
        coords_geo.append((x_geo, y_geo))

    if coords_geo[0] != coords_geo[-1]:
        coords_geo.append(coords_geo[0])

    return coords_geo


def exportar_shapefile(todas_deteccoes, metadados, caminho_saida):
    """Exporta polígonos como Shapefile georreferenciado."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from rasterio.transform import Affine

    if len(todas_deteccoes) == 0:
        print("  ⚠ Nenhuma detecção para exportar.")
        return None

    tiles_dict = {t["tile_id"]: t for t in metadados["tiles"]}
    crs = metadados["crs"]

    geometrias = []
    atributos = []

    for det in todas_deteccoes:
        tile = tiles_dict[det["tile_id"]]
        transform = Affine(*tile["transform"][:6])

        try:
            coords_geo = pixels_para_geo(det["coords_pixel"], transform)
            geom = Polygon(coords_geo)

            if geom.is_valid and geom.area > 0:
                geometrias.append(geom)
                atributos.append({
                    "score": round(det["score"], 3),
                    "area_m2": round(geom.area, 2),
                    "area_px": round(det["area_pixels"], 1),
                    "perim_px": round(det["perimetro_pixels"], 1),
                    "circular": round(det["circularidade"], 3),
                    "tile_id": det["tile_id"],
                    "tile": det["tile_arquivo"],
                    "especie": "",
                    "embauba": 0,
                })
        except Exception:
            continue

    if len(geometrias) == 0:
        print("  ⚠ Nenhum polígono válido.")
        return None

    gdf = gpd.GeoDataFrame(atributos, geometry=geometrias, crs=crs)

    # Shapefile
    gdf.to_file(caminho_saida, driver="ESRI Shapefile")
    print(f"\n  ✅ Shapefile: {caminho_saida}")

    # GeoPackage
    gpkg_path = caminho_saida.replace(".shp", ".gpkg")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"  ✅ GeoPackage: {gpkg_path}")

    print(f"     {len(gdf)} polígonos de copa")
    print(f"     CRS: {crs}")

    return gdf


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("  PIPELINE COMPLETO: Ortomosaico → Tiles → SAM 3 → Shapefile")
    print(f"  Prompt: '{TEXT_PROMPT}' | Tile: {TAMANHO_TILE}px | Threshold: {THRESHOLD}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SALVAR_IMAGENS_ANOTADAS:
        pasta_imagens = os.path.join(OUTPUT_DIR, "tiles_anotados")
        os.makedirs(pasta_imagens, exist_ok=True)

    tempo_total_inicio = time.time()

    # -----------------------------------------------------------------
    # ETAPA 1: Recortar ortomosaico
    # -----------------------------------------------------------------
    metadados, pasta_tiles = recortar_ortomosaico(
        ORTOMOSAICO_PATH, OUTPUT_DIR, TAMANHO_TILE, SOBREPOSICAO
    )

    total_tiles = metadados["total_tiles"]

    # -----------------------------------------------------------------
    # ETAPA 2: Segmentar com SAM 3
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ETAPA 2: Segmentando copas com SAM 3")
    print("=" * 70)

    model, processor, device = carregar_modelo()

    todas_deteccoes = []
    total_copas = 0
    tiles_pulados_nodata = 0
    tiles_processados = 0
    tiles_com_erro_oom = 0

    for i, tile_info in enumerate(metadados["tiles"]):
        tile_path = os.path.join(pasta_tiles, tile_info["arquivo"])
        tile_id = tile_info["tile_id"]

        if not os.path.exists(tile_path):
            continue

        # Filtro NODATA
        valido, ratio_nodata = verificar_nodata(tile_path, max_ratio=MAX_NODATA_RATIO)
        if not valido:
            tiles_pulados_nodata += 1
            continue

        tiles_processados += 1
        print(f"[{i+1}/{total_tiles}] {tile_info['arquivo']} ({tile_info['largura']}x{tile_info['altura']}px, NODATA: {ratio_nodata:.1%})", end=" → ")

        try:
            image, results = segmentar_tile_texto(
                model, processor, device, tile_path, TEXT_PROMPT
            )

            masks = results.get("masks", [])
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            n_deteccoes = len(masks)

            if n_deteccoes > 0:
                print(f"🌳 {n_deteccoes} copa(s)")

                poligonos = masks_para_poligonos(masks, boxes, scores)

                for poly in poligonos:
                    poly["tile_id"] = tile_id
                    poly["tile_arquivo"] = tile_info["arquivo"]
                    todas_deteccoes.append(poly)

                total_copas += len(poligonos)

                if SALVAR_IMAGENS_ANOTADAS:
                    salvar_imagem_anotada(
                        image, results,
                        tile_info["arquivo"],
                        os.path.join(pasta_imagens, f"anotado_{tile_info['arquivo']}")
                    )
            else:
                print("— nenhuma copa")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                tiles_com_erro_oom += 1
                print(f"❌ CUDA OOM! Tile muito grande. Reduza TAMANHO_TILE para 2048.")
                torch.cuda.empty_cache()
                if tiles_com_erro_oom >= 3:
                    print("\n⚠ Muitos erros OOM. Abortando. Reduza TAMANHO_TILE e rode novamente.")
                    break
            else:
                print(f"❌ Erro: {e}")
        except Exception as e:
            print(f"❌ Erro: {e}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # ETAPA 3: Exportar Shapefile
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ETAPA 3: Exportando resultados")
    print("=" * 70)

    shapefile_path = os.path.join(OUTPUT_DIR, "todas_copas.shp")
    gdf = exportar_shapefile(todas_deteccoes, metadados, shapefile_path)

    tempo_total = time.time() - tempo_total_inicio

    # -----------------------------------------------------------------
    # RESUMO FINAL
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESUMO FINAL — PIPELINE COMPLETO")
    print("=" * 70)
    print(f"  📷 Ortomosaico: {ORTOMOSAICO_PATH}")
    print(f"  📐 Tiles: {TAMANHO_TILE}x{TAMANHO_TILE}px | Overlap: {SOBREPOSICAO}px")
    print(f"  📊 Tiles gerados: {total_tiles}")
    print(f"  📊 Tiles processados: {tiles_processados}")
    print(f"  🚫 Tiles pulados (NODATA): {tiles_pulados_nodata}")
    if tiles_com_erro_oom > 0:
        print(f"  ⚠ Tiles com OOM: {tiles_com_erro_oom}")
    print(f"  🌳 Total de copas detectadas: {total_copas}")
    print(f"  🔤 Prompt: '{TEXT_PROMPT}'")
    print(f"  🎯 Threshold: {THRESHOLD}")
    print(f"  ⏱  Tempo total: {tempo_total:.1f}s ({tempo_total/60:.1f} min)")
    print(f"  ⏱  Média por tile: {tempo_total/max(tiles_processados, 1):.1f}s")
    print(f"  📁 Tiles em: {os.path.join(OUTPUT_DIR, 'tiles')}")
    print(f"  📁 Resultados em: {OUTPUT_DIR}")
    print()
    print("  📋 PRÓXIMOS PASSOS NO QGIS:")
    print("  1. Abra todas_copas.shp (ou .gpkg) sobre o ortomosaico")
    print("  2. Ative edição na camada")
    print("  3. Para copas de embaúba: mude campo 'embauba' para 1")
    print("  4. Apague polígonos que são erros")
    print("  5. Salve o shapefile editado")
    print("  6. Processing → Minimum Bounding Geometry → Envelope")
    print("  7. Converta bboxes para formato YOLO")
    print("=" * 70)


if __name__ == "__main__":
    main()