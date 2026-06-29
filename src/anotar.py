"""UI simples (OpenCV) para revisar a anotação da validação.

Mostra as detecções do algoritmo sobre o tile e deixa o usuário **corrigir os
rótulos** (embaúba/lixo) e **desenhar caixas** sobre embaúbas que o algoritmo não
detectou. Salva no próprio JSON da validação (alimentando a avaliação espacial).

Uso::

    python3 src/anotar.py tile_0905
    python3 src/anotar.py data/validacao/tile_0905/tile_0905.json
    python3 src/anotar.py caminho/nova_imagem.jpg
    python3 src/anotar.py data/validacao --pendentes

Controles:
    clique numa caixa        alterna embaúba (verde) / lixo (vermelho)
    arrastar em área vazia   desenha caixa de embaúba faltante (azul)
    clique direito           remove a caixa faltante sob o cursor
    u                        desfaz a última faltante
    s                        salva (marca revisado) e avança
    n                        pula sem salvar
    q / ESC                  encerra sem salvar o tile atual
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from typing import Any

import cv2

from core.deteccao import candidato_eh_embauba, extrair_candidatos

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDACAO_DIR: str = os.path.join(_ROOT, "data", "validacao")
MAX_LADO: int = 900  # maior lado da janela (o tile 2048² é reduzido para caber)

VERDE = (0, 200, 0); VERMELHO = (0, 0, 255); AZUL = (255, 0, 0); AMARELO = (0, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX
EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def salvar_overlay(img_bgr: Any, json_path: str, meta: dict[str, Any],
                   candidatos: list[dict[str, Any]] | None = None) -> None:
    """Salva o overlay `_vis.png` da validação."""
    vis = img_bgr.copy()
    por_id = {i + 1: d for i, d in enumerate(candidatos or [])}
    for d in meta["deteccoes"]:
        cor = VERDE if d["label"] == "embauba" else VERMELHO
        x, y = d["bbox"][0], d["bbox"][1]
        if d["id"] in por_id:
            cv2.drawContours(vis, [por_id[d["id"]]["hull"]], -1, cor, 4)
        else:
            bx, by, bw, bh = d["bbox"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), cor, 4)
        cv2.putText(vis, f"#{d['id']} {d['label']}", (x, max(0, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 3)
    for fa in meta.get("faltantes", []):
        fx, fy, fw, fh = fa["bbox"]
        cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), AZUL, 4)
        cv2.putText(vis, "faltante", (fx, max(0, fy - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, AZUL, 3)
    base = os.path.splitext(os.path.basename(json_path))[0]
    cv2.imwrite(os.path.join(os.path.dirname(json_path), f"{base}_vis.png"), vis)


def criar_validacao_imagem(caminho_img: str, sobrescrever: bool = False) -> str:
    """Copia uma imagem para `data/validacao/<tile>/` e cria seu JSON inicial."""
    img = cv2.imread(caminho_img)
    if img is None:
        print(f"[ERRO] imagem não pôde ser lida: '{caminho_img}'.")
        sys.exit(1)

    os.makedirs(VALIDACAO_DIR, exist_ok=True)
    nome = os.path.basename(caminho_img)
    base, ext = os.path.splitext(nome)
    if ext.lower() not in EXTS:
        print(f"[ERRO] extensão de imagem não suportada: '{ext}'.")
        sys.exit(1)

    tile_dir = os.path.join(VALIDACAO_DIR, base)
    os.makedirs(tile_dir, exist_ok=True)
    dst_img = os.path.join(tile_dir, nome)
    json_path = os.path.join(tile_dir, base + ".json")
    if os.path.exists(json_path) and not sobrescrever:
        return json_path

    if os.path.abspath(caminho_img) != os.path.abspath(dst_img):
        shutil.copy(caminho_img, dst_img)

    candidatos = extrair_candidatos(img)
    deteccoes = []
    for i, d in enumerate(candidatos, 1):
        label = "embauba" if candidato_eh_embauba(d) else "lixo"
        deteccoes.append({
            "id": i,
            "label": label,
            "bbox": d["bbox"],
            "area": d["area"],
            "circular": d["circular"],
            "fill": d["fill"],
        })
    meta = {
        "tile": nome,
        "revisado": False,
        "deteccoes": deteccoes,
        "faltantes": [],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    salvar_overlay(img, json_path, meta, candidatos)
    print(f"[OK] validação criada: {json_path} ({len(deteccoes)} candidatos)")
    return json_path


def resolver_json(ref: str, sobrescrever: bool = False) -> str:
    """Resolve a referência de um tile (.json, .jpg, nome ou caminho) para o JSON."""
    if os.path.isfile(ref) and ref.lower().endswith(EXTS):
        json_path = os.path.splitext(ref)[0] + ".json"
        if os.path.exists(json_path):
            return json_path
        return criar_validacao_imagem(ref, sobrescrever)

    base = ref
    if base.lower().endswith(EXTS):
        base = os.path.splitext(base)[0] + ".json"
    elif not base.endswith(".json"):
        base = base + ".json"
    nome = os.path.splitext(os.path.basename(base))[0]
    for p in (
        base,
        os.path.join(VALIDACAO_DIR, base),
        os.path.join(VALIDACAO_DIR, nome, os.path.basename(base)),
    ):
        if os.path.exists(p):
            return p
    print(f"[ERRO] JSON do tile não encontrado para '{ref}'.")
    sys.exit(1)


def listar_jsons(ref: str, pendentes: bool, sobrescrever: bool) -> list[str]:
    """Resolve uma referência para uma lista de JSONs anotáveis."""
    candidatos = [ref, os.path.join(VALIDACAO_DIR, ref)]
    pasta = next((p for p in candidatos if os.path.isdir(p)), None)
    if pasta is None:
        return [resolver_json(ref, sobrescrever)]

    jsons = sorted(
        os.path.join(raiz, nome)
        for raiz, _, nomes in os.walk(pasta)
        for nome in nomes
        if nome.endswith(".json")
    )
    if pendentes:
        filtrados = []
        for p in jsons:
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            if not meta.get("revisado", False):
                filtrados.append(p)
        jsons = filtrados
    if not jsons:
        print(f"[ERRO] Nenhum JSON encontrado em '{ref}'.")
        sys.exit(1)
    return jsons


def refazer_validacao_pasta(ref: str) -> list[str]:
    """Recria os JSONs de validação de uma pasta usando o algoritmo atual."""
    candidatos = [ref, os.path.join(VALIDACAO_DIR, ref)]
    pasta = next((p for p in candidatos if os.path.isdir(p)), None)
    if pasta is None:
        print(f"[ERRO] pasta não encontrada: '{ref}'.")
        sys.exit(1)

    imagens = sorted(
        os.path.join(raiz, nome)
        for raiz, _, nomes in os.walk(pasta)
        for nome in nomes
        if nome.lower().endswith(EXTS)
        and not os.path.splitext(nome)[0].endswith("_vis")
    )
    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em '{ref}'.")
        sys.exit(1)

    existentes = sorted(
        os.path.join(raiz, nome)
        for raiz, _, nomes in os.walk(pasta)
        for nome in nomes
        if nome.endswith(".json") or nome.endswith("_vis.png")
    )
    if existentes:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(_ROOT, "data", f"validacao_backup_{stamp}")
        for caminho in existentes:
            rel = os.path.relpath(caminho, pasta)
            destino = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            shutil.copy2(caminho, destino)
        print(f"[OK] backup criado: {backup_dir}")

    jsons = [criar_validacao_imagem(img, sobrescrever=True) for img in imagens]
    print(f"[OK] validação refeita: {len(jsons)} JSON(s)")
    return jsons


def resumo(jsons: list[str]) -> None:
    """Imprime um resumo do conjunto de anotação."""
    n_rev = n_det = n_emb = n_lixo = n_falt = 0
    for p in jsons:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        n_rev += int(meta.get("revisado", False))
        n_det += len(meta["deteccoes"])
        n_emb += sum(1 for d in meta["deteccoes"] if d["label"] == "embauba")
        n_lixo += sum(1 for d in meta["deteccoes"] if d["label"] == "lixo")
        n_falt += len(meta.get("faltantes", []))
    print(f"[INFO] {len(jsons)} tiles de validação")
    print(f"[INFO] revisados: {n_rev}/{len(jsons)}")
    print(f"[INFO] detecções: {n_det} ({n_emb} embaúba / {n_lixo} lixo), faltantes: {n_falt}")


def ponto_em_bbox(px: int, py: int, bbox: list[int]) -> bool:
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def salvar(json_path: str, meta: dict[str, Any], faltantes: list[list[int]]) -> None:
    """Grava as anotações de volta no JSON e marca o tile como revisado."""
    meta["faltantes"] = [{"bbox": [int(v) for v in fb]} for fb in faltantes]
    meta["revisado"] = True
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    img = cv2.imread(os.path.join(os.path.dirname(json_path), meta["tile"]))
    if img is not None:
        salvar_overlay(img, json_path, meta)


def anotar(json_path: str, pos: int = 1, total: int = 1) -> str:
    """Abre a janela de anotação para o tile do JSON dado."""
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    img = cv2.imread(os.path.join(os.path.dirname(json_path), meta["tile"]))
    assert img is not None, f"imagem do tile não encontrada: {meta['tile']}"
    altura, largura = img.shape[:2]
    escala = MAX_LADO / max(altura, largura)
    base = cv2.resize(img, (int(largura * escala), int(altura * escala)))

    estado: dict[str, Any] = {
        "faltantes": [list(fa["bbox"]) for fa in meta.get("faltantes", [])],
        "down": None,   # ponto inicial do clique/arrasto (coords originais)
        "drag": None,   # retângulo em arrasto (coords originais)
    }
    win = f"anotar - {meta['tile']}"

    def desenhar() -> None:
        vis = base.copy()
        for d in meta["deteccoes"]:
            x, y, w, h = (int(v * escala) for v in d["bbox"])
            cor = VERDE if d["label"] == "embauba" else VERMELHO
            cv2.rectangle(vis, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(vis, f"#{d['id']}", (x, max(10, y - 4)), FONT, 0.45, cor, 1)
        for fb in estado["faltantes"]:
            x, y, w, h = (int(v * escala) for v in fb)
            cv2.rectangle(vis, (x, y), (x + w, y + h), AZUL, 2)
        if estado["drag"]:
            x0, y0, x1, y1 = (int(v * escala) for v in estado["drag"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), AMARELO, 1)
        n_emb = sum(1 for d in meta["deteccoes"] if d["label"] == "embauba")
        prefixo = f"{pos}/{total}  " if total > 1 else ""
        rev = "rev" if meta.get("revisado", False) else "novo"
        ajuda = (f"{prefixo}{rev}  embauba:{n_emb}  lixo:{len(meta['deteccoes'])-n_emb}  "
                 f"faltantes:{len(estado['faltantes'])}   "
                 "[clique]=alterna  [arrasta]=faltante  [dir]=remove  u s n q")
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
                dentro = [d for d in meta["deteccoes"] if ponto_em_bbox(dx, dy, d["bbox"])]
                if dentro:  # menor caixa que contém o ponto (permite caixas aninhadas)
                    alvo = min(dentro, key=lambda d: d["bbox"][2] * d["bbox"][3])
                    alvo["label"] = "lixo" if alvo["label"] == "embauba" else "embauba"
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
        if k == ord("n"):
            cv2.destroyAllWindows()
            return "pular"
        if k == ord("u") and estado["faltantes"]:
            estado["faltantes"].pop(); desenhar()
        if k == ord("s"):
            salvar(json_path, meta, estado["faltantes"])
            print(f"[OK] salvo: {json_path}  (revisado, {len(estado['faltantes'])} faltantes)")
            cv2.destroyAllWindows()
            return "salvo"
    cv2.destroyAllWindows()
    return "sair"


def anotar_lista(jsons: list[str]) -> None:
    """Percorre uma lista de JSONs para revisão manual."""
    resumo(jsons)
    salvos = pulados = 0
    for i, p in enumerate(jsons, 1):
        print(f"[INFO] abrindo {i}/{len(jsons)}: {p}")
        r = anotar(p, i, len(jsons))
        if r == "salvo":
            salvos += 1
        elif r == "pular":
            pulados += 1
        else:
            break
    print(f"[INFO] sessão encerrada: {salvos} salvo(s), {pulados} pulado(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Revisa labels e embaúbas faltantes da validação.")
    ap.add_argument("alvo", help="tile, JSON ou pasta de JSONs")
    ap.add_argument("--pendentes", action="store_true",
                    help="ao receber pasta, abre apenas JSONs com revisado=false")
    ap.add_argument("--resumo", action="store_true",
                    help="só imprime o resumo do conjunto, sem abrir janela")
    ap.add_argument("--refazer", action="store_true",
                    help="recria JSON inicial de uma imagem ou de todas as imagens de uma pasta")
    args = ap.parse_args()
    if args.refazer and os.path.isdir(args.alvo):
        jsons = refazer_validacao_pasta(args.alvo)
    else:
        jsons = listar_jsons(args.alvo, args.pendentes, args.refazer)
    if args.resumo:
        resumo(jsons)
    else:
        anotar_lista(jsons)
