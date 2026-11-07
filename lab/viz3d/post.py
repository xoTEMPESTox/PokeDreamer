"""Post-processing and video assembly for Hook v2 per DESIGN §2.10, §2.11, and STORYBOARD_hook.md."""

from pathlib import Path
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lab.video.compose import FontManager

BG_COLOR = (15, 20, 25)
BORDER_COLOR = (60, 75, 95)


def create_vignette_mask(height: int = 1080, width: int = 1920) -> np.ndarray:
    """Precompute normalized radial vignette multiplier: 1.0 - 0.22 * r^2.2."""
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]
    r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    vignette = (1.0 - 0.22 * (r ** 2.2)).astype(np.float32)[:, :, None]
    return np.clip(vignette, 0.0, 1.0)


def create_chromatic_maps(height: int = 1080, width: int = 1920, max_shift: float = 1.5):
    """Precompute coordinate remap arrays for 1.5 px radial chromatic aberration."""
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]
    u = (x - cx) / cx
    v = (y - cy) / cy
    r = np.sqrt(u ** 2 + v ** 2)

    map_x_r = (x + u * max_shift * r).astype(np.float32)
    map_y_r = (y + v * max_shift * r).astype(np.float32)
    map_x_b = (x - u * max_shift * r).astype(np.float32)
    map_y_b = (y - v * max_shift * r).astype(np.float32)
    return map_x_r, map_y_r, map_x_b, map_y_b


def apply_chromatic_aberration(
    img: np.ndarray,
    map_x_r: np.ndarray,
    map_y_r: np.ndarray,
    map_x_b: np.ndarray,
    map_y_b: np.ndarray,
) -> np.ndarray:
    """Applies radial shift to R and B channels using cv2.remap (BGR order: 0=B, 1=G, 2=R)."""
    b = cv2.remap(img[:, :, 0], map_x_b, map_y_b, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g = img[:, :, 1]
    r = cv2.remap(img[:, :, 2], map_x_r, map_y_r, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.dstack([b, g, r])


def apply_grain(img: np.ndarray, frame_idx: int, sigma: float = 2.0) -> np.ndarray:
    """Adds subtle Gaussian film grain (sigma ≈ 2.0/255, luminance only) seeded by frame index."""
    rng = np.random.RandomState(frame_idx * 1007 + 42)
    noise = rng.normal(0.0, sigma, size=img.shape[:2]).astype(np.float32)[:, :, None]
    img_f = img.astype(np.float32) + noise
    return np.clip(img_f, 0.0, 255.0).astype(np.uint8)


def create_battle_play_frame(color_f111: np.ndarray, bg_image: np.ndarray | None = None) -> np.ndarray:
    """Renders step-111 colour frame (144x160x3 RGB) in composer's play layout (1920x1080)."""
    if bg_image is not None:
        if bg_image.shape[:2] != (1080, 1920):
            canvas = cv2.resize(bg_image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = bg_image.copy()
    else:
        canvas = np.full((1080, 1920, 3), BG_COLOR, dtype=np.uint8)

    # 4x scale -> (576, 640)
    scaled = cv2.resize(color_f111, (640, 576), interpolation=cv2.INTER_NEAREST)

    # Game frame at (640, 190)
    fx, fy = 640, 190
    fw, fh = 640, 576
    canvas[fy : fy + fh, fx : fx + fw] = scaled

    # Border
    cv2.rectangle(canvas, (fx - 3, fy - 3), (fx + fw + 2, fy + fh + 2), BORDER_COLOR, thickness=3)
    return canvas


def draw_caption(
    img_pil: Image.Image,
    cap: dict,
    fonts: FontManager,
    alpha_fac: float = 1.0,
) -> None:
    """Draws a caption with white bold text and soft dark drop shadow directly over scene."""
    if alpha_fac <= 0.01:
        return

    headline = cap.get("headline", cap.get("text", ""))
    sub = cap.get("sub", "")
    if not headline and not sub:
        return

    anchor = cap.get("anchor", "bottom_center")
    dx = int(cap.get("dx", 0))
    dy = int(cap.get("dy", 0))
    size = int(cap.get("size", 24))

    font_head = fonts.get(size, bold=True)
    sub_size = max(13, int(size * 0.72))
    font_sub = fonts.get(sub_size, bold=False)

    w_h = int(font_head.getlength(headline)) if headline else 0
    bbox_h = font_head.getbbox(headline) if headline else (0, 0, 0, 0)
    h_h = (bbox_h[3] - bbox_h[1]) if headline else 0

    if sub:
        w_s = int(font_sub.getlength(sub))
        bbox_s = font_sub.getbbox(sub)
        h_s = bbox_s[3] - bbox_s[1]
        gap = max(4, int(size * 0.25))
        total_w = max(w_h, w_s)
        total_h = h_h + gap + h_s
    else:
        w_s = 0
        h_s = 0
        gap = 0
        total_w = w_h
        total_h = h_h

    W, H = img_pil.size
    pad_x = 80
    pad_y = 60

    if anchor == "top_center":
        bx = (W - total_w) // 2
        by = pad_y
    elif anchor == "bottom_center":
        bx = (W - total_w) // 2
        by = H - pad_y - total_h
    elif anchor == "top_left":
        bx = pad_x
        by = pad_y
    elif anchor == "bottom_left":
        bx = pad_x
        by = H - pad_y - total_h
    elif anchor == "center":
        bx = (W - total_w) // 2
        by = (H - total_h) // 2
    else:
        bx = (W - total_w) // 2
        by = H - pad_y - total_h

    x = bx + dx
    y = by + dy

    if "center" in anchor:
        x_head = x + (total_w - w_h) // 2 if headline else x
        x_sub = x + (total_w - w_s) // 2 if sub else x
    else:
        x_head = x
        x_sub = x
    y_head = y
    y_sub = y + h_h + gap

    # 1. Soft dark drop shadow (blurred 60%-black copy offset 2px)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    shadow_alpha = int(153 * alpha_fac)
    offset = 2

    if headline:
        sdraw.text((x_head + offset, y_head + offset), headline, font=font_head, fill=(0, 0, 0, shadow_alpha))
    if sub:
        sdraw.text((x_sub + offset, y_sub + offset), sub, font=font_sub, fill=(0, 0, 0, shadow_alpha))

    shadow_blurred = shadow.filter(ImageFilter.GaussianBlur(1.8))
    img_pil.alpha_composite(shadow_blurred)

    # 2. White text directly over scene
    draw = ImageDraw.Draw(img_pil)
    head_alpha = int(255 * alpha_fac)
    sub_alpha = int(235 * alpha_fac)

    if headline:
        draw.text((x_head, y_head), headline, font=font_head, fill=(255, 255, 255, head_alpha))
    if sub:
        draw.text((x_sub, y_sub), sub, font=font_sub, fill=(235, 240, 248, sub_alpha))


def draw_two_tier_card(
    draw: ImageDraw.Draw,
    fonts: FontManager,
    xy: tuple[int, int],
    headline: str,
    sub: str = "",
    tag: str = "",
    min_width: int = 560,
    align: str = "left",
) -> None:
    """Legacy helper: renders via draw_caption with soft shadow."""
    pass


def draw_centered_title_taglines(
    draw: ImageDraw.Draw,
    fonts: FontManager,
    headline: str,
    sub: str,
    third_line: str = "",
    alpha_fac: float = 1.0,
    center_y: int = 680,
) -> None:
    """Legacy helper for title tagline."""
    pass


def compose_single_frame(
    f: int,
    frame_bgr: np.ndarray,
    fonts: FontManager,
    captions: dict,
    vignette_mask: np.ndarray,
    chroma_maps: tuple,
    battle_frame: np.ndarray | None = None,
    insert_2d_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Post-processes one frame with vignette, grain, aberration, text overlays, and transitions."""
    # 1. 2D Hard-Cut Insert (frames 351..440, shot 2d)
    is_2d_insert = 351 <= f <= 440
    if is_2d_insert and insert_2d_frame is not None:
        img = insert_2d_frame.copy()
        img = apply_grain(img, f, sigma=1.5)
    else:
        img = frame_bgr.copy()

        # 2. Chromatic aberration (Shot 1 only, frames 0..149)
        if f < 150:
            map_x_r, map_y_r, map_x_b, map_y_b = chroma_maps
            img = apply_chromatic_aberration(img, map_x_r, map_y_r, map_x_b, map_y_b)

        # 3. Vignette
        img = (img.astype(np.float32) * vignette_mask).astype(np.uint8)

        # 4. Film Grain (luminance only)
        img = apply_grain(img, f, sigma=2.0)

        # 5. Fade from black (frames 0..12, 0.4 s)
        if f <= 12:
            fade = float(f) / 12.0
            img = (img.astype(np.float32) * fade).astype(np.uint8)

        # 6. Final Battle Wipe (Shot 3d, frames 728..740; the old 578..590 is mid-walk now)
        is_wipe = f >= 728
        if is_wipe and battle_frame is not None:
            f_start_wipe = 728
            t_wipe = min(1.0, max(0.0, (f - f_start_wipe) / 12.0))
            split_x = int(1920 * t_wipe)
            wipe_composite = img.copy()
            wipe_composite[:, :split_x] = battle_frame[:, :split_x]
            if 0 < split_x < 1920:
                x_line_0 = max(0, split_x - 2)
                x_line_1 = min(1920, split_x + 2)
                wipe_composite[:, x_line_0:x_line_1] = (255, 220, 125)
            img = wipe_composite

    # 7. 2D Text Overlays (Pillow RGBA composite from hook_captions.json)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")

    t_sec = f / 30.0
    for cap_key, cap in captions.items():
        if cap_key == "title":
            continue  # 3D title rendered in Blender
        st = float(cap.get("start_s", 0.0))
        en = float(cap.get("end_s", 999.0))
        if st <= t_sec <= en:
            alpha_fac = 1.0
            if t_sec - st < 0.15:
                alpha_fac = (t_sec - st) / 0.15
            elif en - t_sec < 0.20:
                alpha_fac = (en - t_sec) / 0.20

            if cap_key == "subtitle" and f > 115:
                alpha_fac = max(0.0, (128 - f) / 13.0)

            draw_caption(pil_img, cap, fonts, alpha_fac=alpha_fac)

    res_bgr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return res_bgr


def assemble_hook(
    frames_dir: Path,
    frames_2d_dir: Path,
    captions_json_path: Path,
    traj_color_npz_path: Path,
    out_mp4_path: Path,
    out_sheet_path: Path,
    total_frames: int = 741,
    fps: int = 30,
) -> dict:
    """Assembles hook_v2.mp4 and contact sheets from 3D rendered frames and 2D inserts."""
    t_start = time.perf_counter()

    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920, max_shift=1.5)

    with open(captions_json_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    tracks_npz = captions_json_path.parent / "hook_tracks.npz"
    if tracks_npz.exists():
        t_data = np.load(tracks_npz)
        total_frames = len(t_data["cam_pos"])

    # Load Step 111 battle frame with blurred 3D background if available
    color_npz = np.load(traj_color_npz_path)
    f111_rgb = color_npz["color"][min(111, len(color_npz["color"]) - 1)]
    f111_bgr = cv2.cvtColor(f111_rgb, cv2.COLOR_RGB2BGR)

    bg_cand = None
    for cand in [frames_dir / f"frame_{total_frames - 1:04d}.png", frames_dir / "frame_0590.png", frames_dir / "frame_0250.png"]:
        if cand.exists():
            bg_cand = cv2.imread(str(cand))
            break
    darkened_bg = None
    if bg_cand is not None:
        blurred = cv2.GaussianBlur(bg_cand, (51, 51), 25)
        darkened_bg = np.clip(blurred.astype(np.float32) * 0.60, 0, 255).astype(np.uint8)

    battle_frame_bgr = create_battle_play_frame(f111_bgr, bg_image=darkened_bg)

    # Video writer
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (1920, 1080))

    saved_sheet_frames = {}
    sheet_targets = [0, 60, 128, 174, 200, 280, 340, 395, 480, 520, 600, 740]

    print(f"Assembling {total_frames} frames into {out_mp4_path.name}...")

    for f_idx in range(total_frames):
        # 1. Load source frame
        insert_2d = None
        if 351 <= f_idx <= 440:  # shot 2d; the old 255..290 range is the 3D sweep now
            p_2d = frames_2d_dir / f"frame_{f_idx:04d}.png"
            if p_2d.exists():
                insert_2d = cv2.imread(str(p_2d))

        # 3D source frame
        p_3d = frames_dir / f"frame_{f_idx:04d}.png"
        if not p_3d.exists():
            # Fallback if frame was not rendered individually
            frame_3d = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            frame_3d = cv2.imread(str(p_3d))

        # 2. Composite post pass
        out_frame = compose_single_frame(
            f_idx,
            frame_3d,
            fonts=fonts,
            captions=captions,
            vignette_mask=vignette,
            chroma_maps=chroma_maps,
            battle_frame=battle_frame_bgr,
            insert_2d_frame=insert_2d,
        )

        writer.write(out_frame)

        if f_idx in sheet_targets:
            saved_sheet_frames[f_idx] = out_frame

        if (f_idx + 1) % 60 == 0 or f_idx == total_frames - 1:
            print(f"  Processed frame {f_idx + 1:03d} / {total_frames}")

    writer.release()
    dur_total = time.perf_counter() - t_start
    print(f"Assembled {total_frames} frames in {dur_total:.2f} s -> {out_mp4_path}")

    # Build 12-frame contact sheet (4 columns x 3 rows)
    # Thumbnail size: 480 x 270 px
    th_w, th_h = 480, 270
    sheet = np.zeros((th_h * 3, th_w * 4, 3), dtype=np.uint8)

    for idx, f_target in enumerate(sheet_targets):
        r = idx // 4
        c = idx % 4
        if f_target in saved_sheet_frames:
            th = cv2.resize(saved_sheet_frames[f_target], (th_w, th_h), interpolation=cv2.INTER_AREA)
        else:
            th = np.zeros((th_h, th_w, 3), dtype=np.uint8)

        # Draw frame number and timestamp overlay
        t_s = f_target / 30.0
        label = f"#{f_target:03d} ({t_s:.2f}s)"
        cv2.rectangle(th, (8, 8), (140, 32), (10, 15, 20), -1)
        cv2.putText(th, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 240, 255), 1, cv2.LINE_AA)

        sheet[r * th_h : (r + 1) * th_h, c * th_w : (c + 1) * th_w] = th

    cv2.imwrite(str(out_sheet_path), sheet)
    print(f"Generated 12-frame contact sheet -> {out_sheet_path}")

    return {
        "total_frames": total_frames,
        "duration_s": dur_total,
        "video_path": str(out_mp4_path),
        "sheet_path": str(out_sheet_path),
    }


if __name__ == "__main__":
    frames_dir = Path("work/viz3d/v2/frames")
    frames_2d_dir = Path("work/viz3d/v2/frames_2d")
    captions_json = Path("work/viz3d/v2/hook_captions.json")
    traj_color = Path("work/video/traj/run_seed_100_color.npz")
    out_mp4 = Path("work/viz3d/v2/hook_v2.mp4")
    out_sheet = Path("work/viz3d/v2/hook_v2_sheet.png")

    assemble_hook(frames_dir, frames_2d_dir, captions_json, traj_color, out_mp4, out_sheet)
