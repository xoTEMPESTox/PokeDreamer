"""Crop one real/imagined pair from the rollout video frame, back to 160x144, quantise to the 4 Game Boy shades."""
import sys
import cv2
import numpy as np

src, pair, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
y0 = 28 + pair * 294            # panels at rows 28-316, 322-610, ... (6-px separators)
for name, x0 in (("real", 0), ("imag", 326)):
    p = f[y0:y0 + 288, x0:x0 + 320].astype(np.float32)
    g = cv2.resize(p, (160, 144), interpolation=cv2.INTER_AREA)
    print(name, "grey levels (5/25/50/75/95 pct):", np.percentile(g, [5, 25, 50, 75, 95]).round(1))
    q = np.digitize(g, [42, 127, 212])          # 0 = black .. 3 = white
    np.save(f"{out}_{name}.npy", q.astype(np.uint8))
    cv2.imwrite(f"{out}_{name}.png", cv2.resize((q * 85).astype(np.uint8), (640, 576), interpolation=cv2.INTER_NEAREST))
    print(name, "shade counts:", np.bincount(q.ravel(), minlength=4))
    if name == "real":                          # tile id per 8x8 cell (ids = order of np.unique), plus a contact sheet
        cells = q.reshape(18, 8, 20, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        u, inv = np.unique(cells, axis=0, return_inverse=True)
        np.save(f"{out}_cellid.npy", inv.reshape(18, 20))
        sheet = np.full((5 * 94, 9 * 76), 128, np.uint8)
        for k, t in enumerate(u):
            yy, xx = divmod(k, 9)
            ty, tx = yy * 94 + 24, xx * 76 + 6
            sheet[ty:ty + 64, tx:tx + 64] = cv2.resize((t.reshape(8, 8) * 85).astype(np.uint8), (64, 64),
                                                       interpolation=cv2.INTER_NEAREST)
            cv2.putText(sheet, str(k), (tx, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        cv2.imwrite(f"{out}_tiles.png", sheet)
        print("distinct tiles:", len(u))
