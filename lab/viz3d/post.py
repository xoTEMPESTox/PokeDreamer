"""Post-processing and video assembly for Hook v2 per DESIGN §2.10, §2.11, and STORYBOARD_hook.md."""

from pathlib import Path
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

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


def create_battle_play_frame(color_f111: np.ndarray) -> np.ndarray:
    """Renders step-111 colour frame (144x160x3 RGB) in composer's play layout (1920x1080)."""
    # Canvas 1920x1080, bg (15, 20, 25)
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
    """Draws a clean, rounded semi-transparent caption card per COPY_REVIEW §1 and §4."""
    x, y = xy
    font_head = fonts.get(22, bold=True)
    font_sub = fonts.get(15, bold=False)
    font_tag = fonts.get(12, bold=True)

    w_head = int(draw.textlength(headline, font=font_head))
    w_sub = int(draw.textlength(sub, font=font_sub)) if sub else 0
    w_tag = int(draw.textlength(tag, font=font_tag)) if tag else 0

    pad_x = 24
    pad_y = 16
    content_w = max(w_head, w_sub, w_tag, min_width - 2 * pad_x)
    box_w = content_w + 2 * pad_x

    box_h = pad_y * 2 + 26
    if sub:
        box_h += 24
    if tag:
        box_h += 18

    if align == "center":
        x = x - box_w // 2

    # Card background: dark rounded rectangle with border
    x0, y0 = x, y
    x1, y1 = x + box_w, y + box_h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(12, 16, 22, 215), outline=(50, 65, 85, 230), width=1)

    cur_y = y0 + pad_y
    if tag:
        draw.text((x0 + pad_x, cur_y), tag.upper(), font=font_tag, fill=(100, 160, 240))
        cur_y += 18

    draw.text((x0 + pad_x, cur_y), headline, font=font_head, fill=(245, 247, 250))
    if sub:
        cur_y += 28
        draw.text((x0 + pad_x, cur_y), sub, font=font_sub, fill=(155, 170, 190))


def draw_centered_title_taglines(
    draw: ImageDraw.Draw,
    fonts: FontManager,
    headline: str,
    sub: str,
    third_line: str = "",
    alpha_fac: float = 1.0,
    center_y: int = 680,
) -> None:
    """Draws Shot 1b subtitle and third line centered under title pond reflection."""
    if alpha_fac <= 0.01:
        return

    cx = 960
    f_head = fonts.get(25, bold=True)
    f_sub = fonts.get(16, bold=False)
    f_third = fonts.get(14, bold=False)

    w_head = int(draw.textlength(headline, font=f_head))
    w_sub = int(draw.textlength(sub, font=f_sub)) if sub else 0
    w_third = int(draw.textlength(third_line, font=f_third)) if third_line else 0
    max_w = max(w_head, w_sub, w_third)

    box_pad_x = 28
    box_w = max_w + 2 * box_pad_x
    box_h = 100 if third_line else 72
    bx0 = cx - box_w // 2
    by0 = center_y - 20
    bx1 = cx + box_w // 2
    by1 = by0 + box_h

    # Semi-transparent pill backing
    bg_alpha = int(190 * alpha_fac)
    border_alpha = int(220 * alpha_fac)
    text_alpha = int(255 * alpha_fac)
    sub_alpha = int(180 * alpha_fac)

    draw.rounded_rectangle(
        [bx0, by0, bx1, by1],
        radius=12,
        fill=(10, 14, 20, bg_alpha),
        outline=(55, 70, 90, border_alpha),
        width=1,
    )

    y_pos = center_y
    draw.text((cx, y_pos), headline, font=f_head, fill=(245, 248, 252, text_alpha), anchor="mm")

    if sub:
        y_pos += 30
        draw.text((cx, y_pos), sub, font=f_sub, fill=(160, 175, 195, sub_alpha), anchor="mm")

    if third_line:
        y_pos += 26
        draw.text((cx, y_pos), third_line, font=f_third, fill=(135, 150, 170, sub_alpha), anchor="mm")


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
    # 1. 2D Hard-Cut Insert (frames 255..290)
    if 255 <= f <= 290 and insert_2d_frame is not None:
        # The 2D swipe insert is already complete at 1920x1080 with layout and labels
        img = insert_2d_frame.copy()
        # Apply subtle grain to match live video texture
        return apply_grain(img, f, sigma=1.5)

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

    # 6. Final Battle Wipe (Shot 3d, frames 578..590 = last 0.4 s)
    if f >= 578 and battle_frame is not None:
        t_wipe = (f - 578) / 12.0  # 0.0 -> 1.0
        # Horizontal sweep wipe from left to right
        split_x = int(1920 * t_wipe)
        wipe_composite = img.copy()
        wipe_composite[:, :split_x] = battle_frame[:, :split_x]
        # 3-px glowing wipe hairline
        if 0 < split_x < 1920:
            x_line_0 = max(0, split_x - 2)
            x_line_1 = min(1920, split_x + 2)
            wipe_composite[:, x_line_0:x_line_1] = (255, 220, 125)  # Soft amber-cyan hairline
        img = wipe_composite

    # 7. 2D Text Overlays (Pillow RGBA composite)
    # Convert BGR -> RGBA
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Time in seconds
    t_sec = f / 30.0

    # 7a. Subtitle & Third line (Shot 1b: 1.6 s to 4.8 s = frames 48..144)
    if 48 <= f <= 144:
        cap_sub = captions.get("subtitle", {})
        cap_third = captions.get("third_line", {})
        head = cap_sub.get("headline", "Dreams the game. Learns to walk in the dream. Plays for real.")
        sub = cap_sub.get("sub", "") if t_sec >= 2.0 else ""
        third = cap_third.get("sub", "") if t_sec >= 2.4 else ""

        # Fade out during frames 129..144
        alpha_fac = 1.0
        if f > 128:
            alpha_fac = max(0.0, (144 - f) / 16.0)

        draw_centered_title_taglines(draw, fonts, head, sub, third, alpha_fac=alpha_fac, center_y=685)

    # 7b. Card 2b: "This is all the AI sees" (frames 181..196 = 6.05 s to 6.55 s)
    elif 181 <= f <= 196:
        card_raw = captions.get("card_raw", {})
        draw_two_tier_card(
            draw, fonts, xy=(80, 890),
            headline=card_raw.get("headline", "This is all the AI sees"),
            sub=card_raw.get("sub", "72 x 80 pixels, greyscale"),
            tag="AGENT VIEW",
        )

    # 7c. Reveal 2c Captions (frames 197..254 = 6.55 s to 8.5 s)
    elif 197 <= f <= 254:
        rev_a = captions.get("reveal_a", {})
        rev_b = captions.get("reveal_b", {})
        rev_note = captions.get("reveal_note", {})

        if t_sec < 6.95:
            # Stage A: Detail restored
            draw_two_tier_card(
                draw, fonts, xy=(80, 875),
                headline=rev_a.get("headline", "Detail restored"),
                sub=rev_a.get("sub", "neural upscaler: 144 x 160, the Game Boy's 4 shades"),
                tag="STAGE 1",
            )
        else:
            # Stage B: Colour restored
            sub_text = rev_b.get("sub", "neural colorizer: a colour per pixel")
            note_text = rev_note.get("sub", "3D: illustration. The real screen is next.")
            full_sub = f"{sub_text}  ·  {note_text}"
            draw_two_tier_card(
                draw, fonts, xy=(80, 875),
                headline=rev_b.get("headline", "Colour restored"),
                sub=full_sub,
                tag="STAGE 2",
            )

    # 7d. Shot 3b: "Now it walks" (frames 330..404 = 11.0 s to 13.5 s)
    elif 330 <= f <= 404:
        walk = captions.get("agent_walk", {})
        draw_two_tier_card(
            draw, fonts, xy=(80, 885),
            headline=walk.get("headline", "Now it walks. It learned this in the dream, never in the real game."),
            sub=walk.get("sub", "navigation policy trained only inside the world model, from frames"),
            tag="POLICY WALK",
            min_width=640,
        )

    # 7e. Shot 3c: "Route 1" (frames 420..495 = 14.0 s to 16.5 s)
    elif 420 <= f <= 495:
        r1 = captions.get("route1_label", {})
        draw_two_tier_card(
            draw, fonts, xy=(80, 915),
            headline=r1.get("headline", "Route 1"),
            sub="",
            tag="LOCATION",
            min_width=220,
        )

    # Convert back RGBA -> BGR
    res_bgr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return res_bgr


def assemble_hook(
    frames_dir: Path,
    frames_2d_dir: Path,
    captions_json_path: Path,
    traj_color_npz_path: Path,
    out_mp4_path: Path,
    out_sheet_path: Path,
    total_frames: int = 591,
    fps: int = 30,
) -> dict:
    """Assembles hook_v2.mp4 and contact sheets from 3D rendered frames and 2D inserts."""
    t_start = time.perf_counter()

    fonts = FontManager()
    vignette = create_vignette_mask(1080, 1920)
    chroma_maps = create_chromatic_maps(1080, 1920, max_shift=1.5)

    with open(captions_json_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # Load Step 111 battle frame
    color_npz = np.load(traj_color_npz_path)
    f111_rgb = color_npz["color"][111]
    # In BGR for OpenCV
    f111_bgr = cv2.cvtColor(f111_rgb, cv2.COLOR_RGB2BGR)
    battle_frame_bgr = create_battle_play_frame(f111_bgr)

    # Video writer
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (1920, 1080))

    saved_sheet_frames = {}
    sheet_targets = [0, 45, 110, 160, 185, 215, 250, 270, 330, 450, 520, 590]

    print(f"Assembling {total_frames} frames into {out_mp4_path.name}...")

    for f_idx in range(total_frames):
        # 1. Load source frame
        insert_2d = None
        if 255 <= f_idx <= 290:
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
