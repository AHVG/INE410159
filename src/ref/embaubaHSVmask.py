import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_cecropia_hsv(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_cecropia = np.array([44, 144, 104])
    upper_cecropia = np.array([58, 231, 203])

    mask = cv2.inRange(hsv, lower_cecropia, upper_cecropia)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img_rgb.copy()
    count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            hull = cv2.convexHull(contour)
            x, y, _, _ = cv2.boundingRect(hull)
            cv2.drawContours(result_img, [hull], -1, (0, 200, 255), 3)
            cv2.putText(result_img, 'Cecropia', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 3)
            count += 1

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].imshow(img_rgb);                axs[0].set_title('1. Tile Original');                     axs[0].axis('off')
    axs[1].imshow(mask_cleaned, cmap='gray'); axs[1].set_title('2. Máscara HSV');                   axs[1].axis('off')
    axs[2].imshow(result_img);             axs[2].set_title(f'3. Resultado Final: {count} Detecções'); axs[2].axis('off')

    plt.tight_layout()
    plt.show()

detect_cecropia_hsv('tile_0053.jpg')
