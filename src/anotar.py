"""Adiciona e anota um tile na revisão de validação.

`adicionar_revisao` roda o algoritmo no tile e grava, em `data/validacao/<tile>/`,
um JSON praticamente igual à saída do algoritmo (`core.deteccao.analisar`) com a
flag `revisado`. A janela de anotação (OpenCV) deixa o usuário **corrigir** essa
saída sobre o esquema "só detecções":

    - clicar numa detecção  → marca como FALSO POSITIVO (vermelho); clicar de novo
      desfaz. Vira o índice em `falsos_positivos`.
    - arrastar em área vazia → desenha uma embaúba FALTANTE (azul) que o algoritmo
      não pegou. Vira uma entrada em `faltantes`.

O `embaubas` (saída crua do algoritmo) não é alterado; as correções ficam em
`falsos_positivos` e `faltantes`, então dá pra recuperar TP/FP/FN:
    TP = len(embaubas) - len(falsos_positivos)
    FP = len(falsos_positivos)
    FN = len(faltantes)

Uso::

    python3 src/anotar.py tile_0053            # cria se faltar e abre a janela
    python3 src/anotar.py caminho/img.jpg
    python3 src/anotar.py tile_0053 --refazer  # recria o JSON do algoritmo e abre

Controles:
    clique numa detecção     alterna falso positivo (verde ↔ vermelho)
    arrastar em área vazia   desenha caixa de embaúba faltante (azul)
    clique direito           remove a caixa faltante sob o cursor
    u                        desfaz a última faltante
    s                        salva (marca revisado) e fecha
    q / ESC                  encerra sem salvar
"""

import argparse
import json
import os
import shutil
import sys
from typing import Any

import cv2

from core.deteccao import analisar

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDACAO_DIR: str = os.path.join(_ROOT, "data", "validacao")
TILES_DIRS: tuple[str, ...] = (os.path.join(_ROOT, "tiles"), os.path.join(_ROOT, "data", "tiles"))
EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MAX_LADO: int = 900  # maior lado da janela (o tile 2048² é reduzido para caber)

VERDE = (0, 200, 0); VERMELHO = (0, 0, 255); AZUL = (255, 0, 0); AMARELO = (0, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def adicionar_revisao(caminho_img: str) -> str:
    """Cria o JSON de revisão do tile: saída do algoritmo + flag `revisado`."""
    img = cv2.imread(caminho_img)
    if img is None:
        print(f"[ERRO] imagem não pôde ser lida: '{caminho_img}'.")
        sys.exit(1)

    nome = os.path.basename(caminho_img)
    base = os.path.splitext(nome)[0]
    tile_dir = os.path.join(VALIDACAO_DIR, base)
    os.makedirs(tile_dir, exist_ok=True)

    dst_img = os.path.join(tile_dir, nome)
    if os.path.abspath(caminho_img) != os.path.abspath(dst_img):
        shutil.copy(caminho_img, dst_img)

    meta = {"tile": nome, "revisado": False, **analisar(img)}
    json_path = os.path.join(tile_dir, base + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    salvar_overlay(img, json_path, meta)
    print(f"[OK] revisão criada: {json_path} ({meta['n_deteccoes']} detecções)")
    return json_path


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
    base = os.path.splitext(os.path.basename(json_path))[0]
    cv2.imwrite(os.path.join(os.path.dirname(json_path), f"{base}_vis.png"), vis)


def ponto_em_bbox(px: int, py: int, bbox: list[int]) -> bool:
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def salvar(json_path: str, meta: dict[str, Any], fp: set[int], faltantes: list[list[int]]) -> None:
    """Grava as correções no JSON e marca o tile como revisado."""
    meta["falsos_positivos"] = sorted(fp)
    meta["faltantes"] = [{"bbox": [int(v) for v in fb]} for fb in faltantes]
    meta["revisado"] = True
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    img = cv2.imread(os.path.join(os.path.dirname(json_path), meta["tile"]))
    if img is not None:
        salvar_overlay(img, json_path, meta)


def anotar(json_path: str) -> str:
    """Abre a janela de anotação para o tile do JSON dado."""
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    img = cv2.imread(os.path.join(os.path.dirname(json_path), meta["tile"]))
    assert img is not None, f"imagem do tile não encontrada: {meta['tile']}"
    altura, largura = img.shape[:2]
    escala = MAX_LADO / max(altura, largura)
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
        rev = "rev" if meta.get("revisado", False) else "novo"
        ajuda = (f"{rev}  embauba:{tp}  falso+:{len(estado['fp'])}  "
                 f"faltantes:{len(estado['faltantes'])}   "
                 "[clique]=falso+  [arrasta]=faltante  [dir]=remove  u s q")
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
    desenhar()
    cv2.setMouseCallback(win, on_mouse)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return "sair"
        if k == ord("u") and estado["faltantes"]:
            estado["faltantes"].pop(); desenhar()
        if k == ord("s"):
            salvar(json_path, meta, estado["fp"], estado["faltantes"])
            print(f"[OK] salvo: {json_path}  (revisado, "
                  f"{len(estado['fp'])} falso+, {len(estado['faltantes'])} faltantes)")
            cv2.destroyAllWindows()
            return "salvo"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adiciona e anota um tile na revisão de validação.")
    ap.add_argument("alvo", help="nome do tile (ex.: tile_0053) ou caminho de imagem")
    ap.add_argument("--refazer", action="store_true",
                    help="recria o JSON a partir do algoritmo (descarta anotações) antes de abrir")
    args = ap.parse_args()

    img_path = resolver_imagem(args.alvo)
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(VALIDACAO_DIR, base, base + ".json")

    precisa_criar = args.refazer or not os.path.exists(json_path)
    if not precisa_criar:
        with open(json_path, encoding="utf-8") as f:
            if "embaubas" not in json.load(f):  # JSON em esquema antigo
                print(f"[INFO] '{json_path}' está no esquema antigo; recriando pelo algoritmo.")
                precisa_criar = True
    if precisa_criar:
        json_path = adicionar_revisao(img_path)
    anotar(json_path)
