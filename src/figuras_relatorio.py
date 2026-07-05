"""Gera figuras qualitativas usadas no relatório e na apresentação."""

import json
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from core.deteccao import _pipeline_base, candidato_eh_embauba, detectar, salvar_passo_a_passo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data", "output")
VALIDACAO_DIR = os.path.join(ROOT, "data", "validacao")
TILES_DIR = os.path.join(ROOT, "data", "tiles")

VERDE = (0, 180, 0)
VERMELHO = (30, 30, 220)
AZUL = (220, 90, 30)
AMARELO = (240, 180, 0)


def _rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _desenhar_candidatos(img_rgb: np.ndarray, candidatos: list[dict[str, Any]], finais: bool) -> np.ndarray:
    vis = img_rgb.copy()
    for c in candidatos:
        if finais and not candidato_eh_embauba(c):
            continue
        cor = (255, 190, 0) if finais else (255, 255, 0)
        hull = c["hull"]
        cv2.drawContours(vis, [hull], -1, cor, 4)
    return vis


def _paineis_pipeline_completo(tile: str) -> list[tuple[np.ndarray, str, str | None]]:
    img_path = os.path.join(TILES_DIR, f"{tile}.jpg")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    base = _pipeline_base(img_bgr)
    resultado = detectar(img_bgr)
    img_rgb = base["img_rgb"]
    candidatos = base["candidatos"]

    return [
        (img_rgb, "1. Original", None),
        (base["suavizado"], "2. Gaussian Blur 9x9", None),
        (base["hsv"][:, :, 0], "3. Canal H", "hsv"),
        (base["mask_bruta"], "4. Mascara HSV bruta", "gray"),
        (base["mask_fecha"], "5. CLOSE 25x25", "gray"),
        (base["mask_limpa"], "6. OPEN 9x9", "gray"),
        (_desenhar_candidatos(img_rgb, candidatos, finais=False), "7. Contornos area > 10.000", None),
        (_desenhar_candidatos(img_rgb, candidatos, finais=True), "8. Filtro eh_embauba", None),
        (resultado["resultado"], f"9. Convex hull final ({resultado['count']})", None),
    ]


def gerar_pipeline_completo(tile: str = "tile_0634", nome_saida: str = "pipeline_completo_classico.png") -> str:
    paineis = _paineis_pipeline_completo(tile)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (img, titulo, cmap) in zip(axs.flat, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Pipeline classico completo - {tile}.jpg", fontsize=16, fontweight="bold")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, nome_saida)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def gerar_etapas_pipeline(tile: str = "tile_0681") -> list[str]:
    paineis = _paineis_pipeline_completo(tile)
    etapas_dir = os.path.join(OUTPUT_DIR, "etapas_pipeline")
    os.makedirs(etapas_dir, exist_ok=True)

    saidas = []
    for i, (img, titulo, cmap) in enumerate(paineis, 1):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.axis("off")
        out = os.path.join(etapas_dir, f"{tile}_{i:02d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        saidas.append(out)
    return saidas


def gerar_passo_a_passo(tile: str = "tile_0681") -> str:
    img_path = os.path.join(TILES_DIR, f"{tile}.jpg")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    passo_dir = os.path.join(OUTPUT_DIR, "passo_a_passo")
    os.makedirs(passo_dir, exist_ok=True)
    salvar_passo_a_passo(detectar(img_bgr), f"{tile}.jpg", passo_dir)
    return os.path.join(passo_dir, f"{tile}.png")


def _expandir_bbox(bbox: list[int], largura: int, altura: int, margem: int = 180) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x0 = max(0, x - margem)
    y0 = max(0, y - margem)
    x1 = min(largura, x + w + margem)
    y1 = min(altura, y + h + margem)
    return x0, y0, x1, y1


def _crop_anotado(img_bgr: np.ndarray, bbox: list[int], cor: tuple[int, int, int], rotulo: str) -> np.ndarray:
    altura, largura = img_bgr.shape[:2]
    x0, y0, x1, y1 = _expandir_bbox(bbox, largura, altura)
    crop = img_bgr[y0:y1, x0:x1].copy()
    x, y, w, h = bbox
    p1 = (x - x0, y - y0)
    p2 = (x + w - x0, y + h - y0)
    cv2.rectangle(crop, p1, p2, cor, 5)
    cv2.putText(crop, rotulo, (max(8, p1[0]), max(34, p1[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 3)
    return _rgb(crop)


def _frac_preto(img_bgr: np.ndarray, bbox: list[int]) -> float:
    altura, largura = img_bgr.shape[:2]
    x0, y0, x1, y1 = _expandir_bbox(bbox, largura, altura)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 1.0
    return float(np.mean(np.all(crop < 12, axis=2)))


def _bbox_area(bbox: list[int]) -> int:
    return int(bbox[2]) * int(bbox[3])


def _encontrar_exemplos() -> dict[str, tuple[str, list[int]]]:
    candidatos: dict[str, list[tuple[float, str, list[int]]]] = {"tp": [], "fp": [], "fn": []}
    for nome in sorted(os.listdir(VALIDACAO_DIR)):
        pasta = os.path.join(VALIDACAO_DIR, nome)
        json_path = os.path.join(pasta, f"{nome}.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        img_path = os.path.join(pasta, f"{nome}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        fps = set(meta.get("falsos_positivos", []))
        deteccoes = meta.get("embaubas", [])
        for i, det in enumerate(deteccoes):
            chave = "fp" if i in fps else "tp"
            bbox = det["bbox"]
            score = _frac_preto(img_bgr, bbox) + 0.00000001 * abs(_bbox_area(bbox) - 90_000)
            candidatos[chave].append((score, nome, bbox))
        for fa in meta.get("faltantes", []):
            bbox = fa["bbox"]
            score = _frac_preto(img_bgr, bbox) + 0.00000001 * abs(_bbox_area(bbox) - 90_000)
            candidatos["fn"].append((score, nome, bbox))

    exemplos: dict[str, tuple[str, list[int]]] = {}
    for chave, opcoes in candidatos.items():
        boas = [op for op in opcoes if op[0] < 0.15]
        fonte = boas if boas else opcoes
        if fonte:
            _, nome, bbox = sorted(fonte, key=lambda item: item[0])[0]
            exemplos[chave] = (nome, bbox)
    return exemplos


def gerar_exemplos_validacao() -> str:
    exemplos = _encontrar_exemplos()
    faltando = {"tp", "fp", "fn"} - exemplos.keys()
    if faltando:
        raise RuntimeError(f"Exemplos ausentes: {sorted(faltando)}")

    tile_final = "tile_0681"
    img_final = cv2.imread(os.path.join(TILES_DIR, f"{tile_final}.jpg"))
    if img_final is None:
        raise FileNotFoundError(os.path.join(TILES_DIR, f"{tile_final}.jpg"))

    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    axs[0, 0].imshow(detectar(img_final)["resultado"])
    axs[0, 0].set_title(f"Saida final do detector\n{tile_final}.jpg", fontsize=12, fontweight="bold")
    axs[0, 0].axis("off")

    configs = [
        (axs[0, 1], "tp", "Deteccao correta (TP)", VERDE, "TP"),
        (axs[1, 0], "fp", "Falso positivo (FP)", VERMELHO, "FP"),
        (axs[1, 1], "fn", "Falso negativo / faltante (FN)", AZUL, "FN"),
    ]

    for ax, chave, titulo, cor, rotulo in configs:
        tile, bbox = exemplos[chave]
        img_path = os.path.join(VALIDACAO_DIR, tile, f"{tile}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        ax.imshow(_crop_anotado(img_bgr, bbox, cor, rotulo))
        ax.set_title(f"{titulo}\n{tile}.jpg", fontsize=12, fontweight="bold")
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=np.array((255, 200, 0)) / 255, lw=4, label="Saida: hull final do detector"),
        plt.Line2D([0], [0], color=np.array(VERDE[::-1]) / 255, lw=4, label="TP: deteccao correta"),
        plt.Line2D([0], [0], color=np.array(VERMELHO[::-1]) / 255, lw=4, label="FP: deteccao incorreta"),
        plt.Line2D([0], [0], color=np.array(AZUL[::-1]) / 255, lw=4, label="FN: copa faltante"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "exemplos_validacao_tp_fp_fn.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(gerar_pipeline_completo())
    print(gerar_pipeline_completo("tile_0681", "pipeline_exemplo_filtro_classico.png"))
    print("\n".join(gerar_etapas_pipeline("tile_0681")))
    print(gerar_passo_a_passo("tile_0681"))
    print(gerar_exemplos_validacao())
