"""Algoritmos de visão computacional clássica para detecção de copas de Cecropia.

Pipeline: suavização → segmentação HSV → morfologia (close/open) → extração de
contornos → classificação por área/circularidade/densidade. Sem deep learning;
apenas `cv2` e `numpy`.
"""

import os
import re
import time
import tracemalloc
import tempfile
from typing import Any

import cv2
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HSV_LOWER:   np.ndarray         = np.array([44, 144, 104])  # limite inferior (H, S, V) da copa
HSV_UPPER:   np.ndarray         = np.array([58, 231, 203])  # limite superior (H, S, V) da copa
BLUR_KERNEL: tuple[int, int]    = (9, 9)
CLOSE_SIZE:  int                = 25
OPEN_SIZE:   int                = 9
AREA_MIN:    int                = 10_000
# Filtro para separar copas de embaúba de centros de palmeira.
#   - palmeira marca o centro de outra árvore → área pequena
#   - copa de embaúba é grande ou espalhada/irregular → circularidade baixa
# (a densidade na máscara bruta, `fill`, foi testada e descartada: não separa.)
#
# Limiares de PRODUÇÃO, escolhidos/validados nos 153 rótulos de data/validacao/.
# A decisão é: forma ('área grande OU pouco circular') E dentro de uma faixa de
# tamanho/solidez plausível. Cada termo foi medido no conjunto de validação:
#   - AREA_GRANDE / CIRC_MAX: o núcleo do filtro (precisão 0.73 vs 0.25 do
#     baseline área>10000).
#   - AREA_MAX: teto que descarta blobs fundidos gigantes, maiores que qualquer
#     copa real (ganho de precisão sem custo de recall).
#   - SOL_MIN: solidez (área_contorno / área_hull) mínima; descarta copas "ralas"
#     que vazam o filtro de forma. Em validação cruzada leave-one-tile-out o
#     limiar 0.48 foi estável (16/18 folds) e elevou a precisão para ~0.81.
AREA_GRANDE: int   = 33_798
CIRC_MAX:    float = 0.1551
AREA_MAX:    int   = 600_000
SOL_MIN:     float = 0.48


def filtro_forma_tamanho(candidato: dict[str, Any]) -> bool:
    """Regra de classificação de um candidato como embaúba, por forma e tamanho.

    Reúne todos os critérios geométricos: a área precisa estar na faixa plausível
    (`AREA_MIN` < área ≤ `AREA_MAX`, descartando manchas minúsculas e blobs
    fundidos gigantes), a forma tem que ser de copa (grande **ou** pouco circular)
    e a solidez ≥ `SOL_MIN` (descarta copas ralas). É aplicada dentro de
    `_pipeline_base`, que grava o resultado na flag ``aceito`` de cada candidato;
    ``get('solidity', 1.0)`` tolera dicts antigos sem a feature.
    """
    forma = candidato["area"] >= AREA_GRANDE or candidato["circular"] <= CIRC_MAX
    return (forma
            and AREA_MIN < candidato["area"] <= AREA_MAX
            and candidato.get("solidity", 1.0) >= SOL_MIN)


def _pipeline_base(img_bgr: np.ndarray) -> dict[str, Any]:
    """Executa as etapas comuns até os contornos candidatos, já classificados."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Etapa 1 — Gaussian Blur: reduz ruído de textura antes da segmentação.
    blurred = cv2.GaussianBlur(img_bgr, BLUR_KERNEL, 0)

    # Etapa 2 — Segmentação HSV: isola o verde característico das copas.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_bruta = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Etapa 3 — Morfologia CLOSE: conecta fragmentos de uma mesma copa.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    mask_fecha = cv2.morphologyEx(mask_bruta, cv2.MORPH_CLOSE, k_close)

    # Etapa 4 — Morfologia OPEN: remove respingos de ruído.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
    mask_limpa = cv2.morphologyEx(mask_fecha, cv2.MORPH_OPEN, k_open)

    # Etapa 5a — Contornos: cada região conectada vira um candidato com features.
    todos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in todos:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(c, True)
        harea = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        canvas = np.zeros(mask_bruta.shape, np.uint8)
        cv2.drawContours(canvas, [hull], -1, 255, -1)
        dentro = cv2.countNonZero(canvas)
        fill = cv2.countNonZero(cv2.bitwise_and(mask_bruta, canvas)) / dentro if dentro else 0
        cand = {
            "contorno": c,
            "area": round(float(area), 1),
            "circular": round(4 * np.pi * area / (peri * peri), 4) if peri else 0.0,
            "fill": round(fill, 4),
            "solidity": round(float(area / harea), 4) if harea else 0.0,
            "bbox": [int(x), int(y), int(w), int(h)],
            "hull": hull,
        }
        # Etapa 5b — Filtro forma/tamanho: classifica o candidato (embaúba vs lixo).
        cand["aceito"] = filtro_forma_tamanho(cand)
        candidatos.append(cand)
    return {
        "img_rgb": img_rgb,
        "suavizado": cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
        "hsv": hsv,
        "mask_bruta": mask_bruta,
        "mask_fecha": mask_fecha,
        "mask_limpa": mask_limpa,
        "candidatos": candidatos,
        "coverage_pct": float(np.count_nonzero(mask_limpa)) / mask_limpa.size * 100,
    }


def detectar(img_bgr: np.ndarray) -> dict[str, Any]:
    """Executa o pipeline completo de detecção em um tile.

    Args:
        img_bgr: Tile carregado pelo OpenCV no formato BGR.

    Returns:
        Dicionário com as imagens intermediárias de cada etapa (para o
        passo a passo), os contornos aceitos, a imagem anotada `resultado`,
        a contagem `count`, as áreas em px² e a cobertura percentual da
        máscara HSV.
    """
    base = _pipeline_base(img_bgr)
    candidatos = base["candidatos"]
    aceitos = [c for c in candidatos if c["aceito"]]
    contornos = [c["contorno"] for c in aceitos]

    # Etapa 6 — Convex Hull: contorno da copa aceita, desenhado e rotulado.
    resultado = base["img_rgb"].copy()
    for c in contornos:
        hull = cv2.convexHull(c)
        x, y, _, _ = cv2.boundingRect(hull)
        cv2.drawContours(resultado, [hull], -1, (0, 200, 255), 3)
        cv2.putText(resultado, 'Cecropia', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 3)

    return {
        "img_rgb":      base["img_rgb"],
        "suavizado":    base["suavizado"],
        "hsv":          base["hsv"],
        "mask_bruta":   base["mask_bruta"],
        "mask_fecha":   base["mask_fecha"],
        "mask_limpa":   base["mask_limpa"],
        "candidatos":    candidatos,
        "contornos":    contornos,
        "resultado":    resultado,
        "count":        len(contornos),
        "n_candidatos": len(candidatos),
        "areas_px":     [c["area"] for c in aceitos],
        "coverage_pct": base["coverage_pct"],
    }


def _desenhar_candidatos(
    img_rgb: np.ndarray,
    candidatos: list[dict[str, Any]],
) -> np.ndarray:
    """Desenha todos os candidatos coloridos pelo filtro de forma/tamanho:
    aceitos (embaúba) em azul, rejeitados (lixo) em amarelo."""
    vis = img_rgb.copy()
    for c in candidatos:
        cor = (0, 190, 255) if c["aceito"] else (255, 255, 0)
        cv2.drawContours(vis, [c["hull"]], -1, cor, 4)
    return vis


def _preparar_painel(img: np.ndarray, titulo: str, cmap: str | None) -> np.ndarray:
    if img.ndim == 2:
        if cmap == "hsv":
            vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_HSV)
        else:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    tamanho = 640
    faixa_titulo = 58
    vis = cv2.resize(vis, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    painel = np.full((tamanho + faixa_titulo, tamanho, 3), 255, dtype=np.uint8)
    painel[faixa_titulo:, :] = vis
    cv2.putText(painel, titulo, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, (25, 25, 25), 2, cv2.LINE_AA)
    return painel


def detectar_cronometrado(img_bgr: np.ndarray) -> tuple[dict[str, Any], float, float]:
    """Roda `detectar()` medindo tempo (s) e memória de pico (MB).

    Helper compartilhado por `analisar` e pela execução em lote
    (`execucao.processar_tile`), para não duplicar a cronometragem.

    Returns:
        ``(r, tempo_s, memoria_mb)`` onde `r` é o dicionário de `detectar`.
    """
    tracemalloc.start()
    t0 = time.time()
    r = detectar(img_bgr)
    tempo = time.time() - t0
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return r, tempo, pico_bytes / 1024 / 1024


def analisar(img_bgr: np.ndarray) -> dict[str, Any]:
    """Detecta embaúbas medindo tempo e memória de pico do processamento.

    Saída autocontida por imagem: os bounds de cada copa mais as métricas de
    diagnóstico. Usada pelo `anotar.py` para montar o estado inicial da validação
    manual.

    Args:
        img_bgr: Imagem carregada pelo OpenCV no formato BGR.

    Returns:
        Dicionário com `embaubas` (lista de bounds), `n_candidatos`,
        `n_deteccoes`, `coverage_pct`, `tempo_s` e `memoria_mb`.
    """
    r, tempo, memoria = detectar_cronometrado(img_bgr)
    embaubas = []
    for c in r["contornos"]:
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)
        embaubas.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": round(float(cv2.contourArea(c)), 1),
            "poligono": hull.reshape(-1, 2).tolist(),
        })
    return {
        "embaubas":     embaubas,
        "n_candidatos": r["n_candidatos"],
        "n_deteccoes":  r["count"],
        "coverage_pct": r["coverage_pct"],
        "tempo_s":      tempo,
        "memoria_mb":   memoria,
    }


def _paineis_passo_a_passo(r: dict[str, Any]) -> list[tuple[np.ndarray, str, str | None]]:
    """Lista ``(imagem, titulo, cmap)`` das etapas do pipeline, na ordem do grid.

    Fonte única das etapas: consumida tanto pelo grid combinado quanto pelas
    imagens individuais salvas por tile em `salvar_figuras_etapas`.
    """
    return [
        (r["img_rgb"],        "1. Original", None),
        (r["suavizado"],      "2. Gaussian Blur 9x9", None),
        (r["hsv"][:, :, 0],   "3. Canal H", "hsv"),
        (r["hsv"][:, :, 1],   "4. Canal S", "gray"),
        (r["hsv"][:, :, 2],   "5. Canal V", "gray"),
        (r["mask_bruta"],     "6. Mascara HSV bruta", "gray"),
        (r["mask_fecha"],     "7. CLOSE 25x25", "gray"),
        (r["mask_limpa"],     "8. OPEN 9x9", "gray"),
        (_desenhar_candidatos(r["img_rgb"], r["candidatos"]), "9. Filtro forma/tamanho", None),
        (r["resultado"],      f"10. Convex hull final ({r['count']})", None),
    ]


def _slug_painel(titulo: str) -> str:
    """Nome de arquivo estável a partir do título do painel.

    ``"9. Convex hull final (12)"`` → ``"9_convex_hull_final"``; o parêntese com
    a contagem (que varia por tile) é removido para o nome não mudar entre tiles.
    """
    numero, resto = titulo.split(". ", 1)
    resto = re.sub(r"\s*\(.*\)", "", resto)
    slug = re.sub(r"[^a-z0-9]+", "_", resto.lower()).strip("_")
    return f"{numero}_{slug}"


def salvar_figuras_etapas(r: dict[str, Any], nome_tile: str, saida_dir: str) -> None:
    """Salva as etapas do pipeline de um tile numa subpasta própria.

    Cria ``saida_dir/<base>/`` com cada etapa como PNG separado
    (``1_original.png`` … ``10_convex_hull_final.png``, prática para escolher uma
    imagem específica no relatório) mais o ``grid.png`` com os 10 painéis juntos.
    O JSON de detecções do tile é gravado na mesma pasta por quem chama.

    Args:
        r: Dicionário retornado por `detectar`.
        nome_tile: Nome do arquivo do tile (ex.: ``tile_0681.jpg``); a extensão
            define o nome da subpasta (``tile_0681``).
        saida_dir: Diretório onde a subpasta do tile é criada.
    """
    paineis = _paineis_passo_a_passo(r)
    imagens = [_preparar_painel(img, titulo, cmap) for img, titulo, cmap in paineis]

    base = os.path.splitext(nome_tile)[0]
    etapas_dir = os.path.join(saida_dir, base)
    os.makedirs(etapas_dir, exist_ok=True)
    for (_, titulo, _), painel in zip(paineis, imagens):
        cv2.imwrite(os.path.join(etapas_dir, _slug_painel(titulo) + ".png"), painel)

    # Grid: completa a última linha com painéis brancos para fechar em blocos de 3.
    branco = np.full_like(imagens[0], 255)
    celulas = imagens + [branco] * (-len(imagens) % 3)
    linhas = [np.hstack(celulas[i:i + 3]) for i in range(0, len(celulas), 3)]
    grid = np.vstack(linhas)
    cv2.imwrite(os.path.join(etapas_dir, "grid.png"), grid)
