"""Anota tiles para validação: corrige a saída do detector e salva o resultado.

Você seleciona uma ou mais entradas (tiles, imagens, globs ou uma pasta) e o
`anotar.py` abre, uma por uma, só as que **ainda não foram validadas**, com as
detecções do algoritmo já desenhadas para corrigir:

    - clicar numa detecção  → marca FALSO POSITIVO (vermelho); clicar de novo desfaz.
    - arrastar em área vazia → desenha uma embaúba FALTANTE (azul) que o algoritmo
      não pegou.

Ao salvar, grava em `data/validacao/<tile>/<tile>.json` a saída do algoritmo
(`embaubas`) mais as correções (`falsos_positivos`, `faltantes`). **A existência
desse JSON é o que marca a entrada como validada** — não há flag de revisão. De
`embaubas`/`falsos_positivos`/`faltantes` recupera-se TP/FP/FN:
    TP = len(embaubas) - len(falsos_positivos)
    FP = len(falsos_positivos)
    FN = len(faltantes)

Uso::

    python3 src/anotar.py tile_0053                     # uma entrada
    python3 src/anotar.py tile_0053 tile_0120 img.jpg   # várias
    python3 src/anotar.py "tile_01*"                    # glob
    python3 src/anotar.py data/tiles --amostra 20       # amostra aleatória de 20
    python3 src/anotar.py data/validacao --resumo       # métricas, sem abrir janela
    python3 src/anotar.py tile_0053 --refazer           # reanota do zero (descarta)
    python3 src/anotar.py data/validacao --incluir-validadas   # reabre as já feitas

Controles:
    clique numa detecção     alterna falso positivo (verde ↔ vermelho)
    arrastar em área vazia   desenha caixa de embaúba faltante (azul)
    clique direito           remove a caixa faltante sob o cursor
    u                        desfaz a última faltante
    s                        salva e fecha
    n                        pula o tile atual sem salvar
    q / ESC                  encerra sem salvar
"""

import argparse
import glob as _glob
import json
import os
import random
import shutil
import sys
from typing import Any

import cv2

from core.deteccao import analisar

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDACAO_DIR: str = os.path.join(_ROOT, "data", "validacao")
TILES_DIRS: tuple[str, ...] = (os.path.join(_ROOT, "tiles"), os.path.join(_ROOT, "data", "tiles"))
EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MAX_LADO: int = 1100  # maior lado da janela (o tile 2048² é reduzido para caber)

VERDE = (0, 200, 0); VERMELHO = (0, 0, 255); AZUL = (255, 0, 0); AMARELO = (0, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Caminhos e seleção ────────────────────────────────────────────────────────

def _base_de(caminho: str) -> str:
    return os.path.splitext(os.path.basename(caminho))[0]


def _json_de(base: str) -> str:
    return os.path.join(VALIDACAO_DIR, base, base + ".json")


def ja_validado(caminho_img: str) -> bool:
    """Uma entrada está validada quando já existe o JSON de resultado dela."""
    return os.path.exists(_json_de(_base_de(caminho_img)))


def resolver_imagem(ref: str) -> str:
    """Resolve `ref` (caminho de imagem ou nome de tile) para o caminho da imagem."""
    if os.path.isfile(ref):
        return ref
    nome = ref if ref.lower().endswith(EXTS) else ref + ".jpg"
    for d in TILES_DIRS:
        p = os.path.join(d, nome)
        if os.path.isfile(p):
            return p
    print(f"[ERRO] imagem não encontrada para '{ref}'.")
    sys.exit(1)


def listar_imagens(alvo: str) -> list[str]:
    """Lista imagens de uma pasta (recursivo), ignorando overlays `_vis`."""
    imagens: list[str] = []
    for raiz, _, arquivos in os.walk(alvo):
        for nome in arquivos:
            stem, ext = os.path.splitext(nome)
            if ext.lower() in EXTS and not stem.endswith("_vis"):
                imagens.append(os.path.join(raiz, nome))
    return sorted(imagens)


def _expandir_alvo(alvo: str) -> list[str]:
    """Expande um alvo (pasta, glob, caminho ou nome de tile) em caminhos de imagem."""
    if os.path.isdir(alvo):
        return listar_imagens(alvo)
    if any(c in alvo for c in "*?["):
        matches = _glob.glob(alvo)
        if not matches:  # glob por nome de tile: tenta nas pastas de tiles
            padrao = alvo if os.path.splitext(alvo)[1] else alvo + ".jpg"
            for d in TILES_DIRS:
                matches += _glob.glob(os.path.join(d, padrao))
        return sorted(p for p in matches
                      if os.path.isfile(p) and os.path.splitext(p)[1].lower() in EXTS)
    if os.path.isfile(alvo):
        return [alvo]
    return [resolver_imagem(alvo)]  # nome de tile (sai com erro se não achar)


def resolver_selecao(alvos: list[str], amostra: int | None) -> list[str]:
    """Junta todos os alvos em uma lista de imagens única, opcionalmente amostrada."""
    imagens: list[str] = []
    vistos: set[str] = set()
    for alvo in alvos:
        for p in _expandir_alvo(alvo):
            ap = os.path.abspath(p)
            if ap not in vistos:
                vistos.add(ap); imagens.append(p)
    if amostra is not None and amostra < len(imagens):
        imagens = sorted(random.sample(imagens, amostra))
    return imagens


def carregar_json(json_path: str) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ── Resumo / métricas ─────────────────────────────────────────────────────────

def imprimir_resumo(jsons: list[str]) -> None:
    """Agrega TP/FP/FN e precisão/recall/F1 sobre os JSONs validados."""
    total = deteccoes = falsos = faltantes = 0
    for json_path in jsons:
        try:
            meta = carregar_json(json_path)
        except (OSError, json.JSONDecodeError):
            print(f"[AVISO] ignorando JSON inválido: {json_path}")
            continue
        total += 1
        deteccoes += len(meta.get("embaubas", []))
        falsos += len(meta.get("falsos_positivos", []))
        faltantes += len(meta.get("faltantes", []))

    tp = deteccoes - falsos
    precisao = tp / deteccoes if deteccoes else 0.0
    recall = tp / (tp + faltantes) if (tp + faltantes) else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) else 0.0

    print(f"Validados: {total}")
    print(f"Detecções: {deteccoes}")
    print(f"Falsos positivos: {falsos}")
    print(f"Faltantes: {faltantes}")
    print(f"TP: {tp}")
    print(f"Precisão: {precisao:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")


# ── Overlay e persistência ────────────────────────────────────────────────────

def salvar_overlay(img_bgr: Any, json_path: str, meta: dict[str, Any]) -> None:
    """Grava `<tile>_vis.png`: detecções (verde / vermelho=falso+) e faltantes (azul)."""
    vis = img_bgr.copy()
    fp = set(meta.get("falsos_positivos", []))
    for i, d in enumerate(meta["embaubas"]):
        x, y, w, h = d["bbox"]
        cor = VERMELHO if i in fp else VERDE
        cv2.rectangle(vis, (x, y), (x + w, y + h), cor, 4)
        cv2.putText(vis, f"#{i}", (x, max(0, y - 12)), FONT, 1.1, cor, 3)
    for fa in meta.get("faltantes", []):
        x, y, w, h = fa["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), AZUL, 4)
        cv2.putText(vis, "faltante", (x, max(0, y - 12)), FONT, 1.1, AZUL, 3)
    base = _base_de(json_path)
    cv2.imwrite(os.path.join(os.path.dirname(json_path), f"{base}_vis.png"), vis)


def salvar(json_path: str, meta: dict[str, Any], fp: set[int],
           faltantes: list[list[int]], img_bgr: Any, src: str | None) -> None:
    """Grava o resultado da validação: saída do algoritmo + correções + overlay.

    A pasta e a cópia da imagem são criadas aqui (na primeira vez que se salva),
    então um JSON só existe para entradas de fato validadas.
    """
    meta["falsos_positivos"] = sorted(fp)
    meta["faltantes"] = [{"bbox": [int(v) for v in fb]} for fb in faltantes]
    tile_dir = os.path.dirname(json_path)
    os.makedirs(tile_dir, exist_ok=True)
    dst_img = os.path.join(tile_dir, meta["tile"])
    if src and os.path.abspath(src) != os.path.abspath(dst_img):
        shutil.copy(src, dst_img)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    salvar_overlay(img_bgr, json_path, meta)


def preparar(caminho_img: str, refazer: bool) -> tuple[dict[str, Any] | None, Any, str, str | None]:
    """Monta ``(meta, img, json_path, src)`` para anotar uma entrada.

    Se já validada e sem `--refazer`, carrega o resultado salvo (com as correções).
    Caso contrário, semeia `meta` com a saída atual do detector (`analisar`); o JSON
    só será escrito quando o usuário salvar.
    """
    base = _base_de(caminho_img)
    json_path = _json_de(base)
    tile_dir = os.path.join(VALIDACAO_DIR, base)

    if os.path.exists(json_path) and not refazer:
        meta = carregar_json(json_path)
        img = cv2.imread(os.path.join(tile_dir, meta["tile"]))
        return meta, img, json_path, None

    if os.path.exists(json_path):  # --refazer: reusa a imagem já copiada
        nome = carregar_json(json_path)["tile"]
        img_src = os.path.join(tile_dir, nome)
    else:
        nome = os.path.basename(caminho_img)
        img_src = caminho_img
    img = cv2.imread(img_src)
    if img is None:
        return None, None, json_path, None
    meta = {"tile": nome, **analisar(img)}
    return meta, img, json_path, img_src


# ── Janela de anotação ────────────────────────────────────────────────────────

def ponto_em_bbox(px: int, py: int, bbox: list[int]) -> bool:
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def anotar(meta: dict[str, Any], img: Any, json_path: str,
           src: str | None, progresso: str = "", lado: int = MAX_LADO) -> str:
    """Abre a janela de anotação; salva no `json_path` ao pressionar `s`."""
    altura, largura = img.shape[:2]
    escala = lado / max(altura, largura)
    base = cv2.resize(img, (int(largura * escala), int(altura * escala)))

    deteccoes = [d["bbox"] for d in meta["embaubas"]]
    estado: dict[str, Any] = {
        "fp": set(meta.get("falsos_positivos", [])),
        "faltantes": [list(fa["bbox"]) for fa in meta.get("faltantes", [])],
        "down": None,   # ponto inicial do clique/arrasto (coords originais)
        "drag": None,   # retângulo em arrasto (coords originais)
    }
    win = f"anotar - {meta['tile']}"

    def desenhar() -> None:
        vis = base.copy()
        for i, bb in enumerate(deteccoes):
            x, y, w, h = (int(v * escala) for v in bb)
            cor = VERMELHO if i in estado["fp"] else VERDE
            cv2.rectangle(vis, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(vis, f"#{i}", (x, max(10, y - 4)), FONT, 0.45, cor, 1)
        for fb in estado["faltantes"]:
            x, y, w, h = (int(v * escala) for v in fb)
            cv2.rectangle(vis, (x, y), (x + w, y + h), AZUL, 2)
        if estado["drag"]:
            x0, y0, x1, y1 = (int(v * escala) for v in estado["drag"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), AMARELO, 1)
        tp = len(deteccoes) - len(estado["fp"])
        prefixo = f"{progresso}  " if progresso else ""
        ajuda = (f"{prefixo}embauba:{tp}  falso+:{len(estado['fp'])}  "
                 f"faltantes:{len(estado['faltantes'])}   "
                 "[clique]=falso+  [arrasta]=faltante  [dir]=remove  u s n q")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(vis, ajuda, (6, 16), FONT, 0.42, (255, 255, 255), 1)
        cv2.imshow(win, vis)

    def on_mouse(event: int, mx: int, my: int, flags: int, param: Any) -> None:
        ox, oy = int(mx / escala), int(my / escala)
        if event == cv2.EVENT_LBUTTONDOWN:
            estado["down"] = (ox, oy); estado["drag"] = None
        elif event == cv2.EVENT_MOUSEMOVE and estado["down"]:
            dx, dy = estado["down"]
            if abs(ox - dx) > 5 or abs(oy - dy) > 5:
                estado["drag"] = (dx, dy, ox, oy); desenhar()
        elif event == cv2.EVENT_LBUTTONUP:
            if estado["drag"]:
                x0, y0, x1, y1 = estado["drag"]
                bb = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
                if bb[2] > 5 and bb[3] > 5:
                    estado["faltantes"].append(bb)
            elif estado["down"]:
                dx, dy = estado["down"]
                dentro = [i for i, bb in enumerate(deteccoes) if ponto_em_bbox(dx, dy, bb)]
                if dentro:  # menor caixa que contém o ponto (permite caixas aninhadas)
                    alvo = min(dentro, key=lambda i: deteccoes[i][2] * deteccoes[i][3])
                    estado["fp"].discard(alvo) if alvo in estado["fp"] else estado["fp"].add(alvo)
            estado["down"] = None; estado["drag"] = None; desenhar()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, fb in enumerate(estado["faltantes"]):
                if ponto_em_bbox(ox, oy, fb):
                    del estado["faltantes"][i]; break
            desenhar()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, base.shape[1], base.shape[0])
    desenhar()
    cv2.setMouseCallback(win, on_mouse)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return "sair"
        if k == ord("n"):
            cv2.destroyAllWindows()
            return "pular"
        if k == ord("u") and estado["faltantes"]:
            estado["faltantes"].pop(); desenhar()
        if k == ord("s"):
            salvar(json_path, meta, estado["fp"], estado["faltantes"], img, src)
            print(f"[OK] salvo: {json_path}  "
                  f"({len(estado['fp'])} falso+, {len(estado['faltantes'])} faltantes)")
            cv2.destroyAllWindows()
            return "salvo"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Anota tiles para validação (corrige a saída do detector).")
    ap.add_argument("alvo", nargs="+",
                    help="tiles, imagens, globs ou pasta a validar")
    ap.add_argument("--amostra", type=int, default=None,
                    help="valida uma amostra aleatória de N entradas da seleção")
    ap.add_argument("--refazer", action="store_true",
                    help="reanota do zero, descartando o resultado salvo")
    ap.add_argument("--incluir-validadas", action="store_true",
                    help="também abre entradas que já têm resultado salvo")
    ap.add_argument("--resumo", action="store_true",
                    help="imprime as métricas das entradas validadas e sai")
    ap.add_argument("--lado", type=int, default=MAX_LADO,
                    help=f"maior lado da janela em px (padrão {MAX_LADO})")
    args = ap.parse_args()

    selecao = resolver_selecao(args.alvo, args.amostra)
    if not selecao:
        print("[ERRO] nenhuma entrada encontrada na seleção.")
        sys.exit(1)

    if args.resumo:
        imprimir_resumo([_json_de(_base_de(p)) for p in selecao if ja_validado(p)])
        return

    fila = selecao if (args.refazer or args.incluir_validadas) \
        else [p for p in selecao if not ja_validado(p)]
    pulados = len(selecao) - len(fila)
    if pulados:
        print(f"[INFO] {pulados} já validada(s) na seleção — puladas "
              f"(use --refazer ou --incluir-validadas para reabrir).")
    if not fila:
        print("[OK] nada a validar.")
        return

    total = len(fila)
    for i, img_path in enumerate(fila, start=1):
        meta, img, json_path, src = preparar(img_path, args.refazer)
        if meta is None or img is None:
            print(f"[ERRO] não pôde ler '{img_path}', pulando.")
            continue
        if img.mean() < 15:  # tile preto (fora da área mapeada): nada a validar
            print(f"[INFO] {i}/{total} {os.path.basename(img_path)}: tile preto, pulando.")
            continue
        print(f"[INFO] abrindo {i}/{total}: {img_path}")
        if anotar(meta, img, json_path, src, f"{i}/{total}", args.lado) == "sair":
            print("[INFO] sessão encerrada.")
            break


if __name__ == "__main__":
    main()
