"""Algoritmos de visão computacional clássica para detecção de copas de Cecropia.

Pipeline: suavização → segmentação HSV → morfologia (close/open) → extração de
contornos → classificação por área/circularidade/densidade. Sem deep learning;
apenas `cv2` e `numpy`.
"""

import os
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


def _pipeline_base(img_bgr: np.ndarray) -> dict[str, Any]:
    """Executa as etapas comuns até candidatos com área mínima."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(img_bgr, BLUR_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_bruta = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
    mask_fecha = cv2.morphologyEx(mask_bruta, cv2.MORPH_CLOSE, k_close)
    mask_limpa = cv2.morphologyEx(mask_fecha, cv2.MORPH_OPEN, k_open)

    todos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in todos:
        area = cv2.contourArea(c)
        if area <= AREA_MIN:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(c, True)
        harea = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        canvas = np.zeros(mask_bruta.shape, np.uint8)
        cv2.drawContours(canvas, [hull], -1, 255, -1)
        dentro = cv2.countNonZero(canvas)
        fill = cv2.countNonZero(cv2.bitwise_and(mask_bruta, canvas)) / dentro if dentro else 0
        candidatos.append({
            "contorno": c,
            "area": round(float(area), 1),
            "circular": round(4 * np.pi * area / (peri * peri), 4) if peri else 0.0,
            "fill": round(fill, 4),
            "solidity": round(float(area / harea), 4) if harea else 0.0,
            "bbox": [int(x), int(y), int(w), int(h)],
            "hull": hull,
        })
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


def candidato_eh_embauba(candidato: dict[str, Any]) -> bool:
    """Decide se um candidato extraído pelo pipeline base é embaúba.

    Aceita quando o candidato tem a forma de copa (grande **ou** pouco circular)
    e está dentro da faixa de tamanho/solidez plausível: abaixo de `AREA_MAX`
    (descarta blobs fundidos gigantes) e com solidez ≥ `SOL_MIN` (descarta copas
    ralas). O ``get('solidity', 1.0)`` mantém compatível qualquer dict antigo sem
    a feature, tratando-o como sólido.
    """
    forma = candidato["area"] >= AREA_GRANDE or candidato["circular"] <= CIRC_MAX
    return (forma
            and candidato["area"] <= AREA_MAX
            and candidato.get("solidity", 1.0) >= SOL_MIN)


def detectar_para_validacao(img_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Detecções já rotuladas (embauba/lixo) para a UI de anotação manual.

    Usa a mesma extração e classificação do pipeline canônico (`detectar`), mas
    preserva os candidatos rejeitados como 'lixo' para que possam ser revistos e
    corrigidos na UI. Cada entrada traz `id`, `label` e as features de validação
    (`bbox`, `area`, `circular`, `fill`, `solidity`). É a única fonte das
    detecções iniciais da validação — `anotar.py` não reimplementa essa lógica.
    """
    return [
        {
            "id":       i,
            "label":    "embauba" if candidato_eh_embauba(c) else "lixo",
            "bbox":     c["bbox"],
            "area":     c["area"],
            "circular": c["circular"],
            "fill":     c["fill"],
            "solidity": c["solidity"],
        }
        for i, c in enumerate(_pipeline_base(img_bgr)["candidatos"], 1)
    ]


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
    aceitos = [c for c in candidatos if candidato_eh_embauba(c)]
    contornos = [c["contorno"] for c in aceitos]

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
        "contornos":    contornos,
        "resultado":    resultado,
        "count":        len(contornos),
        "n_candidatos": len(candidatos),
        "areas_px":     [c["area"] for c in aceitos],
        "coverage_pct": base["coverage_pct"],
    }


def detectar_embaubas(img_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Detecta embaúbas numa imagem e devolve os bounds, serializáveis em JSON.

    É a saída canônica do pipeline: uma entrada por copa detectada (já depois do
    filtro `candidato_eh_embauba`). Consumida tanto pela avaliação (comparar com
    o esperado) quanto pela execução em produção.

    Args:
        img_bgr: Imagem carregada pelo OpenCV no formato BGR.

    Returns:
        Lista de detecções, cada uma com `bbox` ``[x, y, w, h]``, `area` (px²) e
        `poligono` (vértices ``[x, y]`` do convex hull).
    """
    saida = []
    for c in detectar(img_bgr)["contornos"]:
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)
        saida.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": round(float(cv2.contourArea(c)), 1),
            "poligono": hull.reshape(-1, 2).tolist(),
        })
    return saida


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
    diagnóstico. Reaproveitada pela execução em lote (`execucao.processar_tile`)
    e pela detecção avulsa (`main.py <imagem_ou_pasta>`).

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


def salvar_passo_a_passo(r: dict[str, Any], nome_tile: str, passo_a_passo_dir: str) -> None:
    """Salva um PNG com as 8 etapas do pipeline para um tile.

    Args:
        r: Dicionário retornado por `detectar`.
        nome_tile: Nome do arquivo do tile (ex.: ``tile_0681.jpg``); a extensão
            é trocada por ``.png`` na saída.
        passo_a_passo_dir: Diretório onde a figura é gravada.
    """
    paineis = [
        (r["img_rgb"],        "1. Original",                   None),
        (r["suavizado"],      "2. Gaussian Blur (9×9)",         None),
        (r["hsv"][:, :, 0],  "3. Canal H (Matiz)",             "hsv"),
        (r["hsv"][:, :, 1],  "4. Canal S (Saturação)",         "gray"),
        (r["hsv"][:, :, 2],  "5. Canal V (Valor/Brilho)",      "gray"),
        (r["mask_bruta"],     "6. Máscara Bruta (inRange)",     "gray"),
        (r["mask_limpa"],     "7. Máscara Limpa (CLOSE+OPEN)",  "gray"),
        (r["resultado"],      f"8. Resultado — {r['count']} detecções", None),
    ]
    _, axs = plt.subplots(2, 4, figsize=(24, 12))
    plt.suptitle(f"Pipeline — Cecropia / Embaúba ({nome_tile})",
                 fontsize=14, fontweight="bold", y=1.01)
    for ax, (img, titulo, cmap) in zip(axs.flat, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=9, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(passo_a_passo_dir, nome_tile.replace(".jpg", ".png")),
                dpi=80, bbox_inches="tight")
    plt.close()
