"""Turn one quantised 160x144 Game Boy frame into coloured voxel boxes (look test, pair 6 of main_test_id_k64_up.mp4).

Tile labels are hand-assigned for this frame's 45 distinct tiles (Pallet Town). In the automated version they come from
the game's own tile map (RAM 0xC3A0) for real frames and a nearest-tile match for imagined frames.
Coordinates: 1 unit = 1 Game Boy pixel; x = column, y = 143 - row (north is +y), z up. Output: boxes (N,6), rgb (N,3).
"""
import sys
import numpy as np

stem = sys.argv[1]                                  # e.g. work/viz/p5_real
q = np.load(stem + ".npy").astype(int)              # (144,160), 0 = black .. 3 = white
cid = np.load(stem.rsplit("_", 1)[0] + "_cellid.npy")  # (18,20) tile id per 8x8 cell, from prep_pair.py
rng = np.random.default_rng(0)

TREE, TALL, GRASS, PATH, TUFT = {36, 3, 17, 33}, {39}, {24}, {44}, {41}
PLAYER, SIGN = {38, 37, 31, 20, 30, 16}, {25, 1, 6, 34}
WINDOW, DOOR = {22}, {12, 21, 14, 2}
HOUSES = [(0, 4), (12, 20)]                         # cell columns [c0, c1); both span cell rows 12-17
ROOF_ROWS, WALL_ROWS = (96, 112), (112, 144)        # pixel rows
TALL_PAT = None

# colour ramps: index = shade 0 (dark) .. 3 (light), sRGB 0-255
C = {
    "grass": [(52, 102, 46), (70, 128, 56), (112, 170, 76), (140, 196, 92)],
    "tall": [(30, 78, 38), (48, 108, 46), (88, 150, 64), (120, 178, 74)],
    "path": [(150, 128, 96), (180, 158, 120), (208, 186, 146), (230, 210, 168)],
    "tree": [(22, 62, 36), (40, 96, 50), (78, 140, 70), (120, 178, 96)],
    "trunk": [(60, 40, 26), (92, 64, 40), (120, 86, 56), (140, 104, 70)],
    "roof": [(78, 32, 28), (146, 58, 44), (198, 92, 66), (228, 134, 96)],
    "wall": [(70, 62, 58), (150, 140, 126), (208, 198, 178), (246, 238, 218)],
    "window": [(46, 50, 62), (64, 104, 150), (132, 176, 214), (238, 240, 236)],
    "door": [(50, 34, 24), (108, 70, 40), (150, 100, 60), (182, 132, 82)],
    "sign": [(60, 40, 25), (128, 88, 50), (168, 120, 70), (236, 226, 196)],
    "player": [(44, 46, 72), (214, 52, 44), (240, 196, 160), (248, 248, 248)],
    "earth": [(92, 66, 46)] * 4,
}
boxes, cols = [], []


def add(x0, y0, z0, x1, y1, z1, mat, shade=3, jitter=0.04):
    boxes.append((x0, y0, z0, x1, y1, z1))
    cols.append(np.array(C[mat][shade]) / 255 * (1 + rng.uniform(-jitter, jitter)))


label = np.empty((144, 160), object)
for r in range(144):
    for c in range(160):
        t = cid[r // 8, c // 8]
        label[r, c] = ("tree" if t in TREE else "tall" if t in TALL | PLAYER else "grass" if t in GRASS
                       else "tuft" if t in TUFT else "path")
tall_pat = None
for (R, Cc) in zip(*np.nonzero(cid == 39)):
    tall_pat = q[R * 8:R * 8 + 8, Cc * 8:Cc * 8 + 8]
    break

# base slab and ground
add(0, 0, -10, 160, 144, 0, "earth", jitter=0)
for r in range(144):
    for c in range(160):
        x, y, s, L = c, 143 - r, q[r, c], label[r, c]
        if L == "tall":
            s = tall_pat[r % 8, c % 8]               # ground under the player = the tall-grass pattern
            add(x, y, 0, x + 1, y + 1, {0: 7, 1: 5, 2: 3.5, 3: 2}[s], "tall", s)
        elif L == "grass":
            add(x, y, 0, x + 1, y + 1, 2.5 if s < 3 else 2, "grass", s)
        elif L == "tuft" and s < 3:
            add(x, y, 0, x + 1, y + 1, 3, "grass", 1)
        elif L == "tree":
            add(x, y, 0, x + 1, y + 1, 2, "grass", 2)
        else:
            add(x, y, 0, x + 1, y + 1, 1, "path", 3 if s == 3 else 2)

# trees: 2x2-cell blocks whose top-left tile is 36; dome canopy coloured by the sprite's own shading, plus a trunk
for (R, Cc) in zip(*np.nonzero(cid == 36)):
    r0, c0 = R * 8, Cc * 8
    cx, cy = c0 + 8, 143 - (r0 + 8) + 1
    add(cx - 2, cy - 2, 2, cx + 2, cy + 2, 9, "trunk", 1)
    for i in range(16):
        for j in range(16):
            dx, dy = j + 0.5 - 8, -(i + 0.5 - 8)
            d2 = (dx * dx + dy * dy) / 7.6 ** 2
            if d2 > 1:
                continue
            h = np.sqrt(1 - d2)
            s = q[r0 + i, c0 + j]
            add(c0 + j, 143 - (r0 + i), 15 - 7 * h, c0 + j + 1, 143 - (r0 + i) + 1, 15 + 8 * h, "tree", s)

# houses: wall art stands up on the front edge (y = 0), body behind it, roof art on the front slope of a gable
D, H, RISE = 30, 32, 14
for c0, c1 in HOUSES:
    x0, x1 = c0 * 8, c1 * 8
    add(x0 + 0.5, 0.5, 0, x1 - 0.5, D, H, "wall", 3, jitter=0)
    for r in range(*WALL_ROWS):
        for c in range(x0, x1):
            t, s, k = cid[r // 8, c // 8], q[r, c], 143 - r
            mat = "window" if t in WINDOW else "door" if t in DOOR else "wall"
            add(c, -0.6, k, c + 1, 0.5, k + 1, mat, s, jitter=0.02)
    ridge_y = D / 2
    for c in range(x0, x1):
        col = q[ROOF_ROWS[0]:ROOF_ROWS[1], c]
        start = int(np.argmax(col < 3)) if (col < 3).any() else 16   # skip sky-white above the roof outline
        for u in range(start, 16):
            f0, f1 = (15 - u) / 16, (16 - u) / 16                     # 0 at the eave .. 1 at the ridge
            ya, yb = -3 + f0 * (ridge_y + 3), -3 + f1 * (ridge_y + 3)
            add(c, ya, H, c + 1, yb + 0.01, H + f1 * RISE, "roof", col[u], jitter=0.02)
        for v in range(8):                                           # plain back slope
            g0, g1 = v / 8, (v + 1) / 8
            add(c, ridge_y + g0 * (D + 3 - ridge_y), H, c + 1, ridge_y + g1 * (D + 3 - ridge_y),
                H + (1 - g0) * RISE, "roof", 1, jitter=0.02)


def standup(r0, r1, c0, c1, mask, mat, y_front, thick, z_off=0):
    """Stand a 2D sprite up: pixel row r goes to height (r_bottom - r), on a plane at y_front."""
    rb = max(r for r in range(r0, r1) if mask[r - r0].any())
    for r in range(r0, r1):
        for c in range(c0, c1):
            if mask[r - r0, c - c0]:
                k = rb - r + z_off
                add(c, y_front, k, c + 1, y_front + thick, k + 1, mat, q[r, c], jitter=0.02)


# player: pixels that differ from the tall-grass pattern in its 2x3-cell box
pr0, pr1, pc0, pc1 = 56, 80, 64, 80
pm = np.array([[q[r, c] != tall_pat[r % 8, c % 8] for c in range(pc0, pc1)] for r in range(pr0, pr1)])
rb = max(r for r in range(pr0, pr1) if pm[r - pr0].any())
standup(pr0, pr1, pc0, pc1, pm, "player", 143 - rb - 1, 4, z_off=1)

# sign: everything in its 2x2 cells except the white background reachable from the box edge
sr0, sr1, sc0, sc1 = 128, 144, 80, 96
bg = np.zeros((16, 16), bool)
stack = [(i, j) for i in range(16) for j in range(16) if (i in (0, 15) or j in (0, 15)) and q[sr0 + i, sc0 + j] == 3]
while stack:
    i, j = stack.pop()
    if 0 <= i < 16 and 0 <= j < 16 and not bg[i, j] and q[sr0 + i, sc0 + j] == 3:
        bg[i, j] = True
        stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
standup(sr0, sr1, sc0, sc1, ~bg, "sign", 2, 2, z_off=1)

b, rgb = np.array(boxes, np.float32), np.clip(np.array(cols, np.float32), 0, 1)
np.savez(stem + "_boxes.npz", boxes=b, rgb=rgb)
print(stem, "boxes", len(b))
