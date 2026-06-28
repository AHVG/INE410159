import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

HSV_LOWER   = np.array([44, 144, 104])
HSV_UPPER   = np.array([58, 231, 203])
BLUR_KERNEL = (9, 9)
CLOSE_SIZE  = 25
OPEN_SIZE   = 9
AREA_MIN    = 10_000


def detectar(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(img_bgr, BLUR_KERNEL, 0)

    hsv        = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_bruta = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    k_close    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    k_open     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
    mask_fecha = cv2.morphologyEx(mask_bruta, cv2.MORPH_CLOSE, k_close)
    mask_limpa = cv2.morphologyEx(mask_fecha, cv2.MORPH_OPEN,  k_open)

    todos, _  = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in todos if cv2.contourArea(c) > AREA_MIN]

    resultado = img_rgb.copy()
    for c in contornos:
        hull = cv2.convexHull(c)
        x, y, _, _ = cv2.boundingRect(hull)
        cv2.drawContours(resultado, [hull], -1, (0, 200, 255), 3)
        cv2.putText(resultado, 'Cecropia', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 3)

    return {
        "img_rgb":      img_rgb,
        "suavizado":    cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
        "hsv":          hsv,
        "mask_bruta":   mask_bruta,
        "mask_fecha":   mask_fecha,
        "mask_limpa":   mask_limpa,
        "contornos":    contornos,
        "resultado":    resultado,
        "count":        len(contornos),
        "areas_px":     [cv2.contourArea(c) for c in contornos],
        "coverage_pct": float(np.count_nonzero(mask_limpa)) / mask_limpa.size * 100,
    }


def salvar_passo_a_passo(r, nome_tile, passo_a_passo_dir):
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
