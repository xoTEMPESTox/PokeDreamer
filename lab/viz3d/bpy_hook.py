"""Blender 4.5 procedural 3D diorama scene builder and renderer for Hook v2.

Usage (headless Blender):
    blender.exe -b --factory-startup -P lab/viz3d/bpy_hook.py -- world.npz hook_tracks.npz out_dir [options]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# Ensure lab.viz3d modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bpy_build import (
    CLASS_NAMES,
    PALETTE_SRGB,
    srgb_to_linear,
    build_ground_mesh,
    build_fences,
    build_voxel_player,
    configure_engine,
    check_gpu_temp_and_throttle,
)

S = 0.01  # 1 cm per Game Boy pixel


def clear_default_startup_scene():
    """Remove any factory startup default objects (Cube, Light, Camera)."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block, do_unlink=True)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block, do_unlink=True)


def setup_color_management():
    """Setup AgX color management with Punchy look and rich golden-hour exposure."""
    scene = bpy.context.scene
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = -0.40


def setup_nishita_world_and_sun():
    """Setup golden hour Nishita atmospheric sky texture and matching warm Sun lamp."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    # World mist settings for aerial perspective haze
    world.mist_settings.use_mist = True
    world.mist_settings.start = 4.0
    world.mist_settings.depth = 120.0
    world.mist_settings.falloff = "QUADRATIC"

    for vl in bpy.context.scene.view_layers:
        vl.use_pass_mist = True

    # Nishita Sky Texture node
    sky_node = tree.nodes.new("ShaderNodeTexSky")
    sky_node.sky_type = "NISHITA"
    sun_elev_deg = 14.0
    sun_rot_deg = -135.0  # North-West

    sky_node.sun_elevation = math.radians(sun_elev_deg)
    sky_node.sun_rotation = math.radians(sun_rot_deg)
    sky_node.sun_disc = False
    sky_node.altitude = 0.0
    sky_node.air_density = 1.0
    sky_node.dust_density = 2.5
    sky_node.ozone_density = 1.0

    bg_node = tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Strength"].default_value = 0.40
    tree.links.new(sky_node.outputs["Color"], bg_node.inputs["Color"])

    out_node = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # Sun lamp matching Nishita sun direction (Warm ≈3500K tint)
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 8.5
    sun_data.angle = math.radians(1.2)
    sun_data.color = (1.0, 0.76, 0.46)
    sun_obj = bpy.data.objects.new("Sun", sun_data)

    rot_x = math.radians(90.0 - sun_elev_deg)
    rot_z = math.radians(sun_rot_deg)
    sun_obj.rotation_euler = (rot_x, 0.0, rot_z)
    bpy.context.scene.collection.objects.link(sun_obj)
    bpy.context.view_layer.update()

    lamp_dir = (sun_obj.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    to_sun_lamp = -lamp_dir
    sky_to_sun = Vector((
        math.cos(sky_node.sun_elevation) * math.sin(sky_node.sun_rotation),
        -math.cos(sky_node.sun_elevation) * math.cos(sky_node.sun_rotation),
        math.sin(sky_node.sun_elevation),
    )).normalized()

    dot_val = max(-1.0, min(1.0, to_sun_lamp.dot(sky_to_sun)))
    angle_diff_deg = math.degrees(math.acos(dot_val))
    print(f"[Lighting Check] Sun lamp direction: {to_sun_lamp}")
    print(f"[Lighting Check] Nishita sky sun:    {sky_to_sun}")
    print(f"[Lighting Check] Angular difference: {angle_diff_deg:.4f} deg")
    assert angle_diff_deg < 1.0, f"Sun lamp and Nishita sky mismatch: {angle_diff_deg} deg > 1.0 deg"

    return sun_obj, sky_node


def build_hook_water_surface(cls_map: np.ndarray) -> bpy.types.Object | None:
    """Build glassy, refractive teal water surface with Volume Absorption and 4D ripple bump."""
    H_px, W_px = cls_map.shape
    H_tiles, W_tiles = H_px // 8, W_px // 8
    water_cls_idx = CLASS_NAMES.index("water")

    water_mask = np.zeros((H_tiles, W_tiles), dtype=bool)
    for r in range(H_tiles):
        for c in range(W_tiles):
            c_patch = cls_map[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            if np.mean(c_patch == water_cls_idx) > 0.35:
                water_mask[r, c] = True

    if not np.any(water_mask):
        return None

    r_coords, c_coords = np.where(water_mask)
    min_r, max_r = r_coords.min(), r_coords.max() + 1
    min_c, max_c = c_coords.min(), c_coords.max() + 1

    x0 = (min_c - 0.25) * 8 * S
    x1 = (max_c + 0.25) * 8 * S
    y0 = -(min_r - 0.25) * 8 * S
    y1 = -(max_r + 0.25) * 8 * S
    h = -0.005

    verts = [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
    faces = [(0, 1, 2, 3)]

    me = bpy.data.meshes.new("WaterSurfaceMesh")
    me.from_pydata(verts, [], faces)
    me.update()

    mat = bpy.data.materials.new("WaterMat")
    mat.use_nodes = True
    mat.use_raytrace_refraction = True
    nt = mat.node_tree
    out_node = nt.nodes["Material Output"]
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.015, 0.14, 0.24, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.025
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 0.60
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.50
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0

    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    vol_abs = nt.nodes.new("ShaderNodeVolumeAbsorption")
    vol_abs.inputs["Color"].default_value = (0.50, 0.82, 0.88, 1.0)
    vol_abs.inputs["Density"].default_value = 35.0
    nt.links.new(vol_abs.outputs["Volume"], out_node.inputs["Volume"])

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    noise_broad = nt.nodes.new("ShaderNodeTexNoise")
    noise_broad.noise_dimensions = "4D"
    noise_broad.inputs["Scale"].default_value = 6.0
    drv_b = noise_broad.inputs["W"].driver_add("default_value")
    drv_b.driver.expression = "(frame / 30.0) * 0.25"
    nt.links.new(tex_coord.outputs["Object"], noise_broad.inputs["Vector"])

    noise_fine = nt.nodes.new("ShaderNodeTexNoise")
    noise_fine.noise_dimensions = "4D"
    noise_fine.inputs["Scale"].default_value = 48.0
    drv_f = noise_fine.inputs["W"].driver_add("default_value")
    drv_f.driver.expression = "(frame / 30.0) * 1.20"
    nt.links.new(tex_coord.outputs["Object"], noise_fine.inputs["Vector"])

    add_ripples = nt.nodes.new("ShaderNodeMath")
    add_ripples.operation = "ADD"
    nt.links.new(noise_broad.outputs["Fac"], add_ripples.inputs[0])
    nt.links.new(noise_fine.outputs["Fac"], add_ripples.inputs[1])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.015
    nt.links.new(add_ripples.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    me.materials.append(mat)
    water_obj = bpy.data.objects.new("WaterSurface", me)
    bpy.context.scene.collection.objects.link(water_obj)
    return water_obj


def setup_lookstate_ground_material(
    ground_mat: bpy.types.Material,
    world_dir: Path,
    edge_a_track: np.ndarray,
    edge_b_track: np.ndarray,
) -> bpy.types.Object:
    """Setup LookState node network for GroundMat: mixes raw -> crisp -> color + cyan laser hairlines."""
    g_nt = ground_mat.node_tree
    g_nt.nodes.clear()

    out_node = g_nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = g_nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    g_nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # 1. Texture Coordinate & Mapping
    tex_coord = g_nt.nodes.new("ShaderNodeTexCoord")
    sep_pos = g_nt.nodes.new("ShaderNodeSeparateXYZ")
    g_nt.links.new(tex_coord.outputs["Object"], sep_pos.inputs["Vector"])

    # 2. Driver Empty for Edge A and Edge B keyframes
    driver_obj = bpy.data.objects.new("LookStateDriver", None)
    driver_obj.empty_display_size = 0.01
    bpy.context.scene.collection.objects.link(driver_obj)

    # Keyframe edge positions onto driver object location (X = edge_a, Y = edge_b)
    F = len(edge_a_track)
    for f in range(F):
        driver_obj.location.x = float(edge_a_track[f])
        driver_obj.location.y = float(edge_b_track[f])
        driver_obj.keyframe_insert(data_path="location", frame=f)

    val_a = g_nt.nodes.new("ShaderNodeValue")
    val_a.name = "EdgeA"
    drv_a = val_a.outputs["Value"].driver_add("default_value")
    drv_a.driver.type = "AVERAGE"
    var_a = drv_a.driver.variables.new()
    var_a.name = "ea"
    var_a.type = "TRANSFORMS"
    var_a.targets[0].id = driver_obj
    var_a.targets[0].transform_type = "LOC_X"

    val_b = g_nt.nodes.new("ShaderNodeValue")
    val_b.name = "EdgeB"
    drv_b = val_b.outputs["Value"].driver_add("default_value")
    drv_b.driver.type = "AVERAGE"
    var_b = drv_b.driver.variables.new()
    var_b.name = "eb"
    var_b.type = "TRANSFORMS"
    var_b.targets[0].id = driver_obj
    var_b.targets[0].transform_type = "LOC_Y"

    # 3. Three Ground Textures
    # 3a. Color texture (with HSV boost: saturation 1.45, value 0.85)
    img_col = bpy.data.images.load(str((world_dir / "ground_color.png").resolve()))
    tex_col = g_nt.nodes.new("ShaderNodeTexImage")
    tex_col.image = img_col
    tex_col.interpolation = "Closest"
    g_nt.links.new(tex_coord.outputs["UV"], tex_col.inputs["Vector"])

    hsv = g_nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Hue"].default_value = 0.50
    hsv.inputs["Saturation"].default_value = 1.45
    hsv.inputs["Value"].default_value = 0.85
    g_nt.links.new(tex_col.outputs["Color"], hsv.inputs["Color"])

    # 3b. Crisp texture (Game Boy 4-shade grey, Closest)
    img_crisp = bpy.data.images.load(str((world_dir / "ground_crisp.png").resolve()))
    tex_crisp = g_nt.nodes.new("ShaderNodeTexImage")
    tex_crisp.image = img_crisp
    tex_crisp.interpolation = "Closest"
    g_nt.links.new(tex_coord.outputs["UV"], tex_crisp.inputs["Vector"])

    # 3c. Raw texture (Game Boy 2x downsampled grey, Linear)
    img_raw = bpy.data.images.load(str((world_dir / "ground_raw.png").resolve()))
    tex_raw = g_nt.nodes.new("ShaderNodeTexImage")
    tex_raw.image = img_raw
    tex_raw.interpolation = "Linear"
    g_nt.links.new(tex_coord.outputs["UV"], tex_raw.inputs["Vector"])

    # 4. Transitions
    # Step01: (world_x > edge_a) -> raw
    sub_a = g_nt.nodes.new("ShaderNodeMath")
    sub_a.operation = "SUBTRACT"
    g_nt.links.new(sep_pos.outputs["X"], sub_a.inputs[0])
    g_nt.links.new(val_a.outputs["Value"], sub_a.inputs[1])

    cmp_a = g_nt.nodes.new("ShaderNodeMath")
    cmp_a.operation = "GREATER_THAN"
    cmp_a.inputs[1].default_value = 0.0
    g_nt.links.new(sub_a.outputs["Value"], cmp_a.inputs[0])

    mix_01 = g_nt.nodes.new("ShaderNodeMixRGB")
    g_nt.links.new(cmp_a.outputs["Value"], mix_01.inputs["Fac"])
    g_nt.links.new(tex_crisp.outputs["Color"], mix_01.inputs[1])
    g_nt.links.new(tex_raw.outputs["Color"], mix_01.inputs[2])

    # Step12: (world_x > edge_b) -> col_01, else ground_color
    sub_b = g_nt.nodes.new("ShaderNodeMath")
    sub_b.operation = "SUBTRACT"
    g_nt.links.new(sep_pos.outputs["X"], sub_b.inputs[0])
    g_nt.links.new(val_b.outputs["Value"], sub_b.inputs[1])

    cmp_b = g_nt.nodes.new("ShaderNodeMath")
    cmp_b.operation = "GREATER_THAN"
    cmp_b.inputs[1].default_value = 0.0
    g_nt.links.new(sub_b.outputs["Value"], cmp_b.inputs[0])

    mix_final = g_nt.nodes.new("ShaderNodeMixRGB")
    g_nt.links.new(cmp_b.outputs["Value"], mix_final.inputs["Fac"])
    g_nt.links.new(hsv.outputs["Color"], mix_final.inputs[1])
    g_nt.links.new(mix_01.outputs["Color"], mix_final.inputs[2])

    g_nt.links.new(mix_final.outputs["Color"], bsdf.inputs["Base Color"])

    # 5. Cyan Hairline Emission (1-px bright laser line at edge_a and edge_b)
    # line_a = exp(-((x - edge_a) / 0.008)^2)
    div_a = g_nt.nodes.new("ShaderNodeMath")
    div_a.operation = "DIVIDE"
    div_a.inputs[1].default_value = 0.008
    g_nt.links.new(sub_a.outputs["Value"], div_a.inputs[0])

    sq_a = g_nt.nodes.new("ShaderNodeMath")
    sq_a.operation = "MULTIPLY"
    g_nt.links.new(div_a.outputs["Value"], sq_a.inputs[0])
    g_nt.links.new(div_a.outputs["Value"], sq_a.inputs[1])

    exp_a = g_nt.nodes.new("ShaderNodeMath")
    exp_a.operation = "EXPONENT"
    neg_sq_a = g_nt.nodes.new("ShaderNodeMath")
    neg_sq_a.operation = "MULTIPLY"
    neg_sq_a.inputs[1].default_value = -1.0
    g_nt.links.new(sq_a.outputs["Value"], neg_sq_a.inputs[0])
    g_nt.links.new(neg_sq_a.outputs["Value"], exp_a.inputs[0])

    div_b = g_nt.nodes.new("ShaderNodeMath")
    div_b.operation = "DIVIDE"
    div_b.inputs[1].default_value = 0.008
    g_nt.links.new(sub_b.outputs["Value"], div_b.inputs[0])

    sq_b = g_nt.nodes.new("ShaderNodeMath")
    sq_b.operation = "MULTIPLY"
    g_nt.links.new(div_b.outputs["Value"], sq_b.inputs[0])
    g_nt.links.new(div_b.outputs["Value"], sq_b.inputs[1])

    exp_b = g_nt.nodes.new("ShaderNodeMath")
    exp_b.operation = "EXPONENT"
    neg_sq_b = g_nt.nodes.new("ShaderNodeMath")
    neg_sq_b.operation = "MULTIPLY"
    neg_sq_b.inputs[1].default_value = -1.0
    g_nt.links.new(sq_b.outputs["Value"], neg_sq_b.inputs[0])
    g_nt.links.new(neg_sq_b.outputs["Value"], exp_b.inputs[0])

    # Gate emission: only active when edge_a is within sweep range (< 10m)
    cmp_gate = g_nt.nodes.new("ShaderNodeMath")
    cmp_gate.operation = "LESS_THAN"
    cmp_gate.inputs[1].default_value = 10.0
    g_nt.links.new(val_a.outputs["Value"], cmp_gate.inputs[0])

    add_lines = g_nt.nodes.new("ShaderNodeMath")
    add_lines.operation = "ADD"
    g_nt.links.new(exp_a.outputs["Value"], add_lines.inputs[0])
    g_nt.links.new(exp_b.outputs["Value"], add_lines.inputs[1])

    gated_lines = g_nt.nodes.new("ShaderNodeMath")
    gated_lines.operation = "MULTIPLY"
    g_nt.links.new(add_lines.outputs["Value"], gated_lines.inputs[0])
    g_nt.links.new(cmp_gate.outputs["Value"], gated_lines.inputs[1])

    mult_emis = g_nt.nodes.new("ShaderNodeMath")
    mult_emis.operation = "MULTIPLY"
    mult_emis.inputs[1].default_value = 3.5  # Vibrant cyan glow
    g_nt.links.new(gated_lines.outputs["Value"], mult_emis.inputs[0])

    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.49, 0.83, 0.99, 1.0)  # Light cyan
        g_nt.links.new(mult_emis.outputs["Value"], bsdf.inputs["Emission Strength"])

    return driver_obj


def setup_lookstate_solid_color(
    mat: bpy.types.Material,
    base_color: tuple,
    driver_obj: bpy.types.Object,
    roughness: float = 0.80,
) -> None:
    """Setup LookState desaturation on a solid color material (greyscale when world X > EdgeB)."""
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep_pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])

    val_b = nt.nodes.new("ShaderNodeValue")
    val_b.name = "EdgeB"
    drv_b = val_b.outputs["Value"].driver_add("default_value")
    drv_b.driver.type = "AVERAGE"
    var_b = drv_b.driver.variables.new()
    var_b.name = "eb"
    var_b.type = "TRANSFORMS"
    var_b.targets[0].id = driver_obj
    var_b.targets[0].transform_type = "LOC_Y"

    sub_b = nt.nodes.new("ShaderNodeMath")
    sub_b.operation = "SUBTRACT"
    nt.links.new(sep_pos.outputs["X"], sub_b.inputs[0])
    nt.links.new(val_b.outputs["Value"], sub_b.inputs[1])

    cmp_b = nt.nodes.new("ShaderNodeMath")
    cmp_b.operation = "GREATER_THAN"
    cmp_b.inputs[1].default_value = 0.0
    nt.links.new(sub_b.outputs["Value"], cmp_b.inputs[0])

    col_node = nt.nodes.new("ShaderNodeRGB")
    col_node.outputs["Color"].default_value = (base_color[0], base_color[1], base_color[2], 1.0)

    # Luminance L = 0.299 R + 0.587 G + 0.114 B
    lum = 0.299 * base_color[0] + 0.587 * base_color[1] + 0.114 * base_color[2]
    grey_node = nt.nodes.new("ShaderNodeRGB")
    grey_node.outputs["Color"].default_value = (lum, lum, lum, 1.0)

    mix_col = nt.nodes.new("ShaderNodeMixRGB")
    nt.links.new(cmp_b.outputs["Value"], mix_col.inputs["Fac"])
    nt.links.new(col_node.outputs["Color"], mix_col.inputs[1])
    nt.links.new(grey_node.outputs["Color"], mix_col.inputs[2])

    nt.links.new(mix_col.outputs["Color"], bsdf.inputs["Base Color"])


def setup_lookstate_sign_face_material(
    sign_face_mat: bpy.types.Material,
    world_dir: Path,
    driver_obj: bpy.types.Object,
) -> None:
    """Setup LookState 3-texture mix (raw -> crisp -> color + cyan hairline) for sign face."""
    s_nt = sign_face_mat.node_tree
    s_nt.nodes.clear()

    out_node = s_nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = s_nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.70
    s_nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    tex_coord = s_nt.nodes.new("ShaderNodeTexCoord")
    geom = s_nt.nodes.new("ShaderNodeNewGeometry")
    sep_pos = s_nt.nodes.new("ShaderNodeSeparateXYZ")
    s_nt.links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])

    val_a = s_nt.nodes.new("ShaderNodeValue")
    val_a.name = "EdgeA"
    drv_a = val_a.outputs["Value"].driver_add("default_value")
    drv_a.driver.type = "AVERAGE"
    var_a = drv_a.driver.variables.new()
    var_a.name = "ea"
    var_a.type = "TRANSFORMS"
    var_a.targets[0].id = driver_obj
    var_a.targets[0].transform_type = "LOC_X"

    val_b = s_nt.nodes.new("ShaderNodeValue")
    val_b.name = "EdgeB"
    drv_b = val_b.outputs["Value"].driver_add("default_value")
    drv_b.driver.type = "AVERAGE"
    var_b = drv_b.driver.variables.new()
    var_b.name = "eb"
    var_b.type = "TRANSFORMS"
    var_b.targets[0].id = driver_obj
    var_b.targets[0].transform_type = "LOC_Y"

    col_path = world_dir / "sign_texture.png"
    crisp_path = world_dir / "sign_crisp.png"
    raw_path = world_dir / "sign_raw.png"

    img_col = bpy.data.images.load(str(col_path.resolve()))
    tex_col = s_nt.nodes.new("ShaderNodeTexImage")
    tex_col.image = img_col
    tex_col.interpolation = "Closest"
    s_nt.links.new(tex_coord.outputs["UV"], tex_col.inputs["Vector"])

    img_crisp = bpy.data.images.load(str(crisp_path.resolve()))
    tex_crisp = s_nt.nodes.new("ShaderNodeTexImage")
    tex_crisp.image = img_crisp
    tex_crisp.interpolation = "Closest"
    s_nt.links.new(tex_coord.outputs["UV"], tex_crisp.inputs["Vector"])

    img_raw = bpy.data.images.load(str(raw_path.resolve()))
    tex_raw = s_nt.nodes.new("ShaderNodeTexImage")
    tex_raw.image = img_raw
    tex_raw.interpolation = "Linear"
    s_nt.links.new(tex_coord.outputs["UV"], tex_raw.inputs["Vector"])

    # Step 01: world X > Edge A -> raw, else crisp
    sub_a = s_nt.nodes.new("ShaderNodeMath")
    sub_a.operation = "SUBTRACT"
    s_nt.links.new(sep_pos.outputs["X"], sub_a.inputs[0])
    s_nt.links.new(val_a.outputs["Value"], sub_a.inputs[1])

    cmp_a = s_nt.nodes.new("ShaderNodeMath")
    cmp_a.operation = "GREATER_THAN"
    cmp_a.inputs[1].default_value = 0.0
    s_nt.links.new(sub_a.outputs["Value"], cmp_a.inputs[0])

    mix_01 = s_nt.nodes.new("ShaderNodeMixRGB")
    s_nt.links.new(cmp_a.outputs["Value"], mix_01.inputs["Fac"])
    s_nt.links.new(tex_crisp.outputs["Color"], mix_01.inputs[1])
    s_nt.links.new(tex_raw.outputs["Color"], mix_01.inputs[2])

    # Step 12: world X > Edge B -> mix_01, else tex_col
    sub_b = s_nt.nodes.new("ShaderNodeMath")
    sub_b.operation = "SUBTRACT"
    s_nt.links.new(sep_pos.outputs["X"], sub_b.inputs[0])
    s_nt.links.new(val_b.outputs["Value"], sub_b.inputs[1])

    cmp_b = s_nt.nodes.new("ShaderNodeMath")
    cmp_b.operation = "GREATER_THAN"
    cmp_b.inputs[1].default_value = 0.0
    s_nt.links.new(sub_b.outputs["Value"], cmp_b.inputs[0])

    mix_final = s_nt.nodes.new("ShaderNodeMixRGB")
    s_nt.links.new(cmp_b.outputs["Value"], mix_final.inputs["Fac"])
    s_nt.links.new(tex_col.outputs["Color"], mix_final.inputs[1])
    s_nt.links.new(mix_01.outputs["Color"], mix_final.inputs[2])

    s_nt.links.new(mix_final.outputs["Color"], bsdf.inputs["Base Color"])

    # Cyan laser hairlines emission
    div_a = s_nt.nodes.new("ShaderNodeMath")
    div_a.operation = "DIVIDE"
    div_a.inputs[1].default_value = 0.008
    s_nt.links.new(sub_a.outputs["Value"], div_a.inputs[0])
    sq_a = s_nt.nodes.new("ShaderNodeMath")
    sq_a.operation = "MULTIPLY"
    s_nt.links.new(div_a.outputs["Value"], sq_a.inputs[0])
    s_nt.links.new(div_a.outputs["Value"], sq_a.inputs[1])
    exp_a = s_nt.nodes.new("ShaderNodeMath")
    exp_a.operation = "EXPONENT"
    neg_sq_a = s_nt.nodes.new("ShaderNodeMath")
    neg_sq_a.operation = "MULTIPLY"
    neg_sq_a.inputs[1].default_value = -1.0
    s_nt.links.new(sq_a.outputs["Value"], neg_sq_a.inputs[0])
    s_nt.links.new(neg_sq_a.outputs["Value"], exp_a.inputs[0])

    div_b = s_nt.nodes.new("ShaderNodeMath")
    div_b.operation = "DIVIDE"
    div_b.inputs[1].default_value = 0.008
    s_nt.links.new(sub_b.outputs["Value"], div_b.inputs[0])
    sq_b = s_nt.nodes.new("ShaderNodeMath")
    sq_b.operation = "MULTIPLY"
    s_nt.links.new(div_b.outputs["Value"], sq_b.inputs[0])
    s_nt.links.new(div_b.outputs["Value"], sq_b.inputs[1])
    exp_b = s_nt.nodes.new("ShaderNodeMath")
    exp_b.operation = "EXPONENT"
    neg_sq_b = s_nt.nodes.new("ShaderNodeMath")
    neg_sq_b.operation = "MULTIPLY"
    neg_sq_b.inputs[1].default_value = -1.0
    s_nt.links.new(sq_b.outputs["Value"], neg_sq_b.inputs[0])
    s_nt.links.new(neg_sq_b.outputs["Value"], exp_b.inputs[0])

    cmp_gate = s_nt.nodes.new("ShaderNodeMath")
    cmp_gate.operation = "LESS_THAN"
    cmp_gate.inputs[1].default_value = 10.0
    s_nt.links.new(val_a.outputs["Value"], cmp_gate.inputs[0])

    add_lines = s_nt.nodes.new("ShaderNodeMath")
    add_lines.operation = "ADD"
    s_nt.links.new(exp_a.outputs["Value"], add_lines.inputs[0])
    s_nt.links.new(exp_b.outputs["Value"], add_lines.inputs[1])

    gated_lines = s_nt.nodes.new("ShaderNodeMath")
    gated_lines.operation = "MULTIPLY"
    s_nt.links.new(add_lines.outputs["Value"], gated_lines.inputs[0])
    s_nt.links.new(cmp_gate.outputs["Value"], gated_lines.inputs[1])

    mult_emis = s_nt.nodes.new("ShaderNodeMath")
    mult_emis.operation = "MULTIPLY"
    mult_emis.inputs[1].default_value = 3.5
    s_nt.links.new(gated_lines.outputs["Value"], mult_emis.inputs[0])

    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.49, 0.83, 0.99, 1.0)
        s_nt.links.new(mult_emis.outputs["Value"], bsdf.inputs["Emission Strength"])


def build_solid_buildings(cls_map: np.ndarray, ground_mat: bpy.types.Material) -> bpy.types.Object | None:
    """Build fully enclosed solid buildings with capped roofs, capped walls, and textured facades."""
    H_px, W_px = cls_map.shape
    H_tiles, W_tiles = H_px // 8, W_px // 8

    # "sign" tiles are standing boards now (build_sign_posts); as walls they became building-height pillars
    wall_names = {"building_wall", "door"}
    roof_names = {"roof_red", "roof_blue"}

    is_wall = np.zeros((H_tiles, W_tiles), dtype=bool)
    is_roof = np.zeros((H_tiles, W_tiles), dtype=bool)

    for r in range(H_tiles):
        for c in range(W_tiles):
            patch = cls_map[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            dom_cls = int(np.bincount(patch.flatten()).argmax())
            dom_name = CLASS_NAMES[dom_cls] if dom_cls < len(CLASS_NAMES) else "other"
            if dom_name in wall_names:
                is_wall[r, c] = True
            elif dom_name in roof_names:
                is_roof[r, c] = True

    is_bldg = is_wall | is_roof
    visited = np.zeros((H_tiles, W_tiles), dtype=bool)
    clusters = []

    for r0 in range(H_tiles):
        for c0 in range(W_tiles):
            if is_bldg[r0, c0] and not visited[r0, c0]:
                queue = [(r0, c0)]
                visited[r0, c0] = True
                comp = []
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H_tiles and 0 <= nc < W_tiles:
                            if is_bldg[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                clusters.append(comp)

    bldg_mask = np.zeros((H_tiles, W_tiles), dtype=bool)
    low_posts = []
    for comp in clusters:
        if len(comp) >= 4:
            for r, c in comp:
                bldg_mask[r, c] = True
        else:
            for r, c in comp:
                low_posts.append((r, c))

    verts = []
    uvs = []
    faces = []
    v_idx = 0

    z_wall = 0.28
    z_roof = 0.42

    for r in range(H_tiles):
        for c in range(W_tiles):
            if not bldg_mask[r, c]:
                continue

            x0 = c * 8 * S
            x1 = (c + 1) * 8 * S
            y0 = -r * 8 * S
            y1 = -(r + 1) * 8 * S

            u0 = (c * 8) / float(W_px)
            u1 = ((c + 1) * 8) / float(W_px)
            v0 = 1.0 - (r * 8) / float(H_px)
            v1 = 1.0 - ((r + 1) * 8) / float(H_px)

            if is_wall[r, c]:
                # 1. Solid Wall Box (z: 0.0 -> z_wall)
                verts.extend([(x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, z_wall), (x0, y1, z_wall)])
                uvs.extend([(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

                if r == 0 or not bldg_mask[r - 1, c]:
                    verts.extend([(x1, y0, 0.0), (x0, y0, 0.0), (x0, y0, z_wall), (x1, y0, z_wall)])
                    uvs.extend([(u1, v0), (u0, v0), (u0, v1), (u1, v1)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

                if c == 0 or not bldg_mask[r, c - 1]:
                    verts.extend([(x0, y0, 0.0), (x0, y1, 0.0), (x0, y1, z_wall), (x0, y0, z_wall)])
                    uvs.extend([(u0, v0), (u0, v1), (u0, v1), (u0, v0)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

                if c + 1 == W_tiles or not bldg_mask[r, c + 1]:
                    verts.extend([(x1, y1, 0.0), (x1, y0, 0.0), (x1, y0, z_wall), (x1, y1, z_wall)])
                    uvs.extend([(u1, v1), (u1, v0), (u1, v0), (u1, v1)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

                # TOP CAP (at z_wall)
                verts.extend([(x0, y0, z_wall), (x1, y0, z_wall), (x1, y1, z_wall), (x0, y1, z_wall)])
                uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            elif is_roof[r, c]:
                # 2. Solid Roof Slab (z: z_wall -> z_roof, or 0 -> z_roof)
                z_base = z_wall if (r + 1 < H_tiles and is_wall[r + 1, c]) else 0.0

                verts.extend([(x0, y0, z_roof), (x1, y0, z_roof), (x1, y1, z_roof), (x0, y1, z_roof)])
                uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

                verts.extend([(x0, y1, z_base), (x1, y1, z_base), (x1, y1, z_roof), (x0, y1, z_roof)])
                uvs.extend([(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

                if r == 0 or not is_roof[r - 1, c]:
                    verts.extend([(x1, y0, 0.0), (x0, y0, 0.0), (x0, y0, z_roof), (x1, y0, z_roof)])
                    uvs.extend([(u1, v0), (u0, v0), (u0, v1), (u1, v1)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

                if c == 0 or not is_roof[r, c - 1]:
                    verts.extend([(x0, y0, 0.0), (x0, y1, 0.0), (x0, y1, z_roof), (x0, y0, z_roof)])
                    uvs.extend([(u0, v0), (u0, v1), (u0, v1), (u0, v0)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

                if c + 1 == W_tiles or not is_roof[r, c + 1]:
                    verts.extend([(x1, y1, 0.0), (x1, y0, 0.0), (x1, y0, z_roof), (x1, y1, z_roof)])
                    uvs.extend([(u1, v1), (u1, v0), (u1, v0), (u1, v1)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

    me = bpy.data.meshes.new("BuildingsMesh")
    me.from_pydata(verts, [], faces)
    me.update()

    uv_layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for loop_idx in poly.loop_indices:
            v_i = me.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = uvs[v_i]

    me.materials.append(ground_mat)
    bldg_obj = bpy.data.objects.new("Buildings", me)
    bpy.context.scene.collection.objects.link(bldg_obj)

    # 3. Build low capped posts (~0.35 tile = 0.045 m tall) for ledge components < 4 tiles
    if low_posts:
        post_mat = bpy.data.materials.new("LedgePostMat")
        post_mat.use_nodes = True
        post_bsdf = post_mat.node_tree.nodes["Principled BSDF"]
        post_bsdf.inputs["Base Color"].default_value = (0.55, 0.45, 0.35, 1.0)
        post_bsdf.inputs["Roughness"].default_value = 0.75

        for idx, (pr, pc) in enumerate(low_posts):
            cx = (pc + 0.5) * 8 * S
            cy = -(pr + 0.5) * 8 * S
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.016, depth=0.045, location=(cx, cy, 0.0225)
            )
            p_obj = bpy.context.active_object
            p_obj.name = f"LedgePost_{idx}"
            p_obj.data.materials.append(post_mat)

    return bldg_obj


def build_voxel_player_hook(
    player_track: np.ndarray,
    player_sprites_cls: np.ndarray,
    player_sprites_shade: np.ndarray,
    distinct_steps: np.ndarray,
    player_hop: np.ndarray | None = None,
) -> bpy.types.Object:
    """Build standup voxel character with stage-1 red palette and continuous walk/hop animation."""
    player_cls_idx = CLASS_NAMES.index("sprite_player")
    ramp = PALETTE_SRGB["sprite_player"]

    mat = bpy.data.materials.new("PlayerMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "col"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65

    player_obj = bpy.data.objects.new("Player", None)
    player_obj.empty_display_type = "PLAIN_AXES"
    player_obj.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(player_obj)

    unique_sprites = {}
    step_to_sprite = {}

    for st_idx, step_val in enumerate(distinct_steps):
        cls_crop = player_sprites_cls[st_idx]
        shade_crop = player_sprites_shade[st_idx]
        key = (cls_crop.tobytes(), shade_crop.tobytes())

        if key not in unique_sprites:
            active_voxels = np.where(cls_crop == player_cls_idx)
            verts = []
            faces = []
            cols = []
            v_idx = 0

            for r_px, c_px in zip(*active_voxels):
                sh = shade_crop[r_px, c_px]
                lin_col = srgb_to_linear(ramp[sh])
                rgba = [lin_col[0], lin_col[1], lin_col[2], 1.0]

                z0 = (15 - r_px) * S
                z1 = (16 - r_px) * S
                x0 = (c_px - 8) * S
                x1 = (c_px - 7) * S

                for d in range(4):
                    y0 = (d - 2) * S
                    y1 = (d - 1) * S

                    verts.extend([
                        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
                    ])
                    quad = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
                    for q in quad:
                        faces.append((v_idx + q[0], v_idx + q[1], v_idx + q[2], v_idx + q[3]))
                    v_idx += 8
                    cols.extend([rgba] * 24)

            me = bpy.data.meshes.new(f"PlayerVoxelMesh_{len(unique_sprites)}")
            me.from_pydata(verts, [], faces)
            me.update(calc_edges=True)

            col_attr = me.color_attributes.new("col", "FLOAT_COLOR", "CORNER")
            col_attr.data.foreach_set("color", np.array(cols, dtype=np.float32).reshape(-1))
            me.materials.append(mat)

            s_obj = bpy.data.objects.new(f"PlayerSprite_{len(unique_sprites)}", me)
            s_obj.parent = player_obj
            bpy.context.scene.collection.objects.link(s_obj)
            unique_sprites[key] = s_obj

        step_to_sprite[int(step_val)] = unique_sprites[key]

    F = len(player_track)
    for f in range(F):
        px, py, facing_step = player_track[f][:3]
        hop = float(player_hop[f]) if player_hop is not None else 0.0

        if abs(px) < 20.0:
            x_m, y_m = float(px), float(py)
        else:
            x_m, y_m = float(px * S), float(-py * S)

        player_obj.location = (x_m, y_m, hop + 0.005)
        player_obj.keyframe_insert(data_path="location", frame=f)

        cur_step = int(facing_step)
        active_s = step_to_sprite.get(cur_step, next(iter(unique_sprites.values())) if unique_sprites else None)
        for s_obj in unique_sprites.values():
            hide = (s_obj != active_s)
            s_obj.hide_render = hide
            s_obj.hide_viewport = hide
            s_obj.keyframe_insert(data_path="hide_render", frame=f)
            s_obj.keyframe_insert(data_path="hide_viewport", frame=f)

    return player_obj


def build_sign_posts(
    sign_tiles: np.ndarray,
    sign_boards: np.ndarray | None = None,
    world_dir: Path | None = None,
    ground_mat: bpy.types.Material | None = None,
    driver_obj: bpy.types.Object | None = None,
) -> bpy.types.Collection | None:
    """Build low wooden posts for ledge markers and standing 3D sign boards for real signs."""
    has_tiles = sign_tiles is not None and len(sign_tiles) > 0
    has_boards = sign_boards is not None and len(sign_boards) > 0
    if not has_tiles and not has_boards:
        return None

    sign_coll = bpy.data.collections.new("SignPosts")
    bpy.context.scene.collection.children.link(sign_coll)

    post_mat = bpy.data.materials.new("SignPostMat")
    post_mat.use_nodes = True
    if driver_obj:
        setup_lookstate_solid_color(post_mat, (0.35, 0.22, 0.12, 1.0), driver_obj, roughness=0.80)
    else:
        bsdf = post_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.35, 0.22, 0.12, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.80

    # 1. Low wooden posts for ledge markers (24 instances)
    if has_tiles:
        for idx, (cx, cy, r_tile, c_tile) in enumerate(sign_tiles):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.006, depth=0.035, location=(float(cx), float(cy), 0.0175)
            )
            post_obj = bpy.context.active_object
            post_obj.name = f"LedgePost_{idx}"
            post_obj.data.materials.append(post_mat)

            if post_obj.name not in sign_coll.objects:
                sign_coll.objects.link(post_obj)
            if post_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(post_obj)

    # 2. Standing 3D sign boards on wooden post (5 instances)
    if has_boards:
        sign_face_mat = bpy.data.materials.new("SignFaceMat")
        sign_face_mat.use_nodes = True
        w_dir = world_dir or Path("work/viz3d/v2/world")

        if driver_obj and (w_dir / "sign_texture.png").exists():
            setup_lookstate_sign_face_material(sign_face_mat, w_dir, driver_obj)
        else:
            s_nt = sign_face_mat.node_tree
            s_bsdf = s_nt.nodes["Principled BSDF"]
            s_bsdf.inputs["Roughness"].default_value = 0.70
            sign_tex_path = (w_dir / "sign_texture.png").resolve()
            if sign_tex_path.exists():
                tex_img = bpy.data.images.load(str(sign_tex_path))
                img_node = s_nt.nodes.new("ShaderNodeTexImage")
                img_node.image = tex_img
                img_node.interpolation = "Closest"
                tex_coord = s_nt.nodes.new("ShaderNodeTexCoord")
                s_nt.links.new(tex_coord.outputs["UV"], img_node.inputs["Vector"])
                s_nt.links.new(img_node.outputs["Color"], s_bsdf.inputs["Base Color"])
            else:
                s_bsdf.inputs["Base Color"].default_value = (0.78, 0.68, 0.52, 1.0)

        for idx, (cx, cy, r_top, c_left) in enumerate(sign_boards):
            # Cylinder post: radius 0.008, depth 0.060, location z = 0.030
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.008, depth=0.060, location=(float(cx), float(cy), 0.030)
            )
            post_obj = bpy.context.active_object
            post_obj.name = f"SignBoardPost_{idx}"
            post_obj.data.materials.append(post_mat)

            # Thin box board: 0.16 x 0.012 x 0.16 m, location z = 0.110
            bpy.ops.mesh.primitive_cube_add(
                size=1.0, location=(float(cx), float(cy), 0.110)
            )
            board_obj = bpy.context.active_object
            board_obj.name = f"SignBoard_{idx}"
            board_obj.scale = (0.16, 0.012, 0.16)
            bpy.ops.object.transform_apply(scale=True)

            board_obj.data.materials.append(post_mat)       # slot 0: wood
            board_obj.data.materials.append(sign_face_mat)  # slot 1: sign face

            me = board_obj.data
            if not me.uv_layers:
                uv_layer = me.uv_layers.new(name="UVMap")
            else:
                uv_layer = me.uv_layers.active

            for poly in me.polygons:
                if poly.normal.y < -0.5:
                    poly.material_index = 1
                    for loop_idx in poly.loop_indices:
                        v = me.vertices[me.loops[loop_idx].vertex_index]
                        # v.co is in world space: transform_apply(scale=True) also applies location (its default)
                        u = (v.co.x - (float(cx) - 0.08)) / 0.16
                        v_coord = (v.co.z - (0.110 - 0.08)) / 0.16
                        uv_layer.data[loop_idx].uv = (float(np.clip(u, 0.0, 1.0)), float(np.clip(v_coord, 0.0, 1.0)))
                else:
                    poly.material_index = 0

            for obj in (post_obj, board_obj):
                if obj.name not in sign_coll.objects:
                    sign_coll.objects.link(obj)
                if obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(obj)

    return sign_coll



def build_tree_prototypes(driver_obj: bpy.types.Object | None = None) -> bpy.types.Object:
    """Build low-poly tree prototype mesh in hidden Prototypes collection."""
    proto_coll = bpy.data.collections.get("Prototypes")
    if proto_coll is None:
        proto_coll = bpy.data.collections.new("Prototypes")
        bpy.context.scene.collection.children.link(proto_coll)
        proto_coll.hide_viewport = True
        proto_coll.hide_render = True

    trunk_mat = bpy.data.materials.new("TreeTrunkMat")
    trunk_mat.use_nodes = True
    if driver_obj:
        setup_lookstate_solid_color(trunk_mat, (0.32, 0.20, 0.10, 1.0), driver_obj, roughness=0.85)
    else:
        t_bsdf = trunk_mat.node_tree.nodes["Principled BSDF"]
        t_bsdf.inputs["Base Color"].default_value = (0.32, 0.20, 0.10, 1.0)
        t_bsdf.inputs["Roughness"].default_value = 0.85

    canopy_mat = bpy.data.materials.new("TreeCanopyMat")
    canopy_mat.use_nodes = True
    c_nt = canopy_mat.node_tree
    c_bsdf = c_nt.nodes["Principled BSDF"]

    c_tex_coord = c_nt.nodes.new("ShaderNodeTexCoord")
    c_noise = c_nt.nodes.new("ShaderNodeTexNoise")
    c_noise.noise_dimensions = "4D"
    c_noise.inputs["Scale"].default_value = 8.0
    drv_w = c_noise.inputs["W"].driver_add("default_value")
    drv_w.driver.expression = "(frame / 30.0) * 0.80"
    c_nt.links.new(c_tex_coord.outputs["Object"], c_noise.inputs["Vector"])

    c_mix = c_nt.nodes.new("ShaderNodeMixRGB")
    c_mix.inputs["Color1"].default_value = (0.16, 0.44, 0.22, 1.0)
    c_mix.inputs["Color2"].default_value = (0.34, 0.68, 0.36, 1.0)
    c_nt.links.new(c_noise.outputs["Fac"], c_mix.inputs["Fac"])

    if driver_obj:
        geom = c_nt.nodes.new("ShaderNodeNewGeometry")
        sep_pos = c_nt.nodes.new("ShaderNodeSeparateXYZ")
        c_nt.links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])

        val_b = c_nt.nodes.new("ShaderNodeValue")
        val_b.name = "EdgeB"
        drv_b = val_b.outputs["Value"].driver_add("default_value")
        drv_b.driver.type = "AVERAGE"
        var_b = drv_b.driver.variables.new()
        var_b.name = "eb"
        var_b.type = "TRANSFORMS"
        var_b.targets[0].id = driver_obj
        var_b.targets[0].transform_type = "LOC_Y"

        sub_b = c_nt.nodes.new("ShaderNodeMath")
        sub_b.operation = "SUBTRACT"
        c_nt.links.new(sep_pos.outputs["X"], sub_b.inputs[0])
        c_nt.links.new(val_b.outputs["Value"], sub_b.inputs[1])

        cmp_b = c_nt.nodes.new("ShaderNodeMath")
        cmp_b.operation = "GREATER_THAN"
        cmp_b.inputs[1].default_value = 0.0
        c_nt.links.new(sub_b.outputs["Value"], cmp_b.inputs[0])

        hsv_c = c_nt.nodes.new("ShaderNodeHueSaturation")
        hsv_c.inputs["Saturation"].default_value = 0.0
        c_nt.links.new(c_mix.outputs["Color"], hsv_c.inputs["Color"])

        mix_final = c_nt.nodes.new("ShaderNodeMixRGB")
        c_nt.links.new(cmp_b.outputs["Value"], mix_final.inputs["Fac"])
        c_nt.links.new(c_mix.outputs["Color"], mix_final.inputs[1])
        c_nt.links.new(hsv_c.outputs["Color"], mix_final.inputs[2])
        c_nt.links.new(mix_final.outputs["Color"], c_bsdf.inputs["Base Color"])
    else:
        c_nt.links.new(c_mix.outputs["Color"], c_bsdf.inputs["Base Color"])
    c_bsdf.inputs["Roughness"].default_value = 0.50

    c_bump = c_nt.nodes.new("ShaderNodeBump")
    c_bump.inputs["Strength"].default_value = 0.25
    c_bump.inputs["Distance"].default_value = 0.006
    c_nt.links.new(c_noise.outputs["Fac"], c_bump.inputs["Height"])
    c_nt.links.new(c_bump.outputs["Normal"], c_bsdf.inputs["Normal"])

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12, radius=0.014, depth=0.08, location=(0, 0, 0.04)
    )
    trunk_obj = bpy.context.active_object
    trunk_obj.data.materials.append(trunk_mat)

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2, radius=0.085, location=(0, 0, 0.13)
    )
    canopy_obj = bpy.context.active_object
    canopy_obj.scale = (1.0, 1.0, 0.85)
    bpy.ops.object.transform_apply(scale=True)
    canopy_obj.data.materials.append(canopy_mat)

    bpy.ops.object.select_all(action="DESELECT")
    trunk_obj.select_set(True)
    canopy_obj.select_set(True)
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.join()

    tree_proto = bpy.context.active_object
    tree_proto.name = "TreePrototype"
    proto_coll.objects.link(tree_proto)
    if tree_proto.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(tree_proto)

    return tree_proto


def build_tree_instances(tree_blocks: np.ndarray, tree_proto: bpy.types.Object) -> bpy.types.Object | None:
    """Instance tree prototype with position jitter (<=0.08 tile), random scale (±15%), yaw, and wind sway."""
    if len(tree_blocks) == 0:
        return None

    verts = [tuple(b[:3]) for b in tree_blocks]
    me = bpy.data.meshes.new("TreePointCloudMesh")
    me.from_pydata(verts, [], [])
    me.update()

    tree_pc_obj = bpy.data.objects.new("TreePointCloud", me)
    bpy.context.scene.collection.objects.link(tree_pc_obj)

    mod = tree_pc_obj.modifiers.new("TreeScatter", "NODES")
    ng = bpy.data.node_groups.new("TreeScatterTree", "GeometryNodeTree")
    mod.node_group = ng

    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    outp = ng.nodes.new("NodeGroupOutput")

    set_pos = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(inp.outputs["Geometry"], set_pos.inputs["Geometry"])

    rand_offset = ng.nodes.new("FunctionNodeRandomValue")
    rand_offset.data_type = "FLOAT_VECTOR"
    rand_offset.inputs["Min"].default_value = (-0.012, -0.012, 0.0)
    rand_offset.inputs["Max"].default_value = (0.012, 0.012, 0.0)
    ng.links.new(rand_offset.outputs["Value"], set_pos.inputs["Offset"])

    inst_node = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(set_pos.outputs["Geometry"], inst_node.inputs["Points"])

    obj_info = ng.nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = tree_proto
    obj_info.transform_space = "RELATIVE"
    ng.links.new(obj_info.outputs["Geometry"], inst_node.inputs["Instance"])

    rand_scale = ng.nodes.new("FunctionNodeRandomValue")
    rand_scale.data_type = "FLOAT_VECTOR"
    rand_scale.inputs["Min"].default_value = (0.85, 0.85, 0.85)
    rand_scale.inputs["Max"].default_value = (1.15, 1.15, 1.15)
    ng.links.new(rand_scale.outputs["Value"], inst_node.inputs["Scale"])

    rot_inst = ng.nodes.new("GeometryNodeRotateInstances")
    ng.links.new(inst_node.outputs["Instances"], rot_inst.inputs["Instances"])

    rand_yaw = ng.nodes.new("FunctionNodeRandomValue")
    rand_yaw.data_type = "FLOAT"
    rand_yaw.inputs["Min"].default_value = 0.0
    rand_yaw.inputs["Max"].default_value = 6.283

    time_node = ng.nodes.new("GeometryNodeInputSceneTime")
    pos_node = ng.nodes.new("GeometryNodeInputPosition")
    sep_xyz = ng.nodes.new("ShaderNodeSeparateXYZ")
    ng.links.new(pos_node.outputs["Position"], sep_xyz.inputs["Vector"])

    math_time = ng.nodes.new("ShaderNodeMath")
    math_time.operation = "MULTIPLY"
    math_time.inputs[1].default_value = 1.80
    ng.links.new(time_node.outputs["Seconds"], math_time.inputs[0])

    math_phase = ng.nodes.new("ShaderNodeMath")
    math_phase.operation = "ADD"
    ng.links.new(math_time.outputs["Value"], math_phase.inputs[0])
    ng.links.new(sep_xyz.outputs["X"], math_phase.inputs[1])

    math_sin = ng.nodes.new("ShaderNodeMath")
    math_sin.operation = "SINE"
    ng.links.new(math_phase.outputs["Value"], math_sin.inputs[0])

    math_amp = ng.nodes.new("ShaderNodeMath")
    math_amp.operation = "MULTIPLY"
    math_amp.inputs[1].default_value = 0.0436
    ng.links.new(math_sin.outputs["Value"], math_amp.inputs[0])

    comb_rot = ng.nodes.new("ShaderNodeCombineXYZ")
    ng.links.new(math_amp.outputs["Value"], comb_rot.inputs["X"])
    ng.links.new(rand_yaw.outputs["Value"], comb_rot.inputs["Z"])
    ng.links.new(comb_rot.outputs["Vector"], rot_inst.inputs["Rotation"])

    ng.links.new(rot_inst.outputs["Instances"], outp.inputs["Geometry"])
    return tree_pc_obj


def build_blade_prototypes() -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """Build wider, curved grass blade tufts, dense tall-grass tufts, and flower prototypes."""
    proto_coll = bpy.data.collections.get("Prototypes")
    if proto_coll is None:
        proto_coll = bpy.data.collections.new("Prototypes")
        bpy.context.scene.collection.children.link(proto_coll)
        proto_coll.hide_viewport = True
        proto_coll.hide_render = True

    blade_mat = bpy.data.materials.new("GrassBladeMat")
    blade_mat.use_nodes = True
    b_nt = blade_mat.node_tree
    b_bsdf = b_nt.nodes["Principled BSDF"]
    b_bsdf.inputs["Base Color"].default_value = (0.24, 0.62, 0.14, 1.0)
    b_bsdf.inputs["Roughness"].default_value = 0.45
    if "Subsurface Weight" in b_bsdf.inputs:
        b_bsdf.inputs["Subsurface Weight"].default_value = 0.18

    # 1. Lawn Grass: 3-blade curved tuft (width 5.0 mm at base, height 2.2 cm)
    wb = 0.0050 / 2.0
    hb1, hb2, hb3 = 0.008, 0.016, 0.024
    curve_y = 0.0035

    verts_blade = [
        (-wb, 0, 0), (wb, 0, 0), (wb * 0.75, curve_y * 0.5, hb1), (-wb * 0.75, curve_y * 0.5, hb1),
        (wb * 0.45, curve_y, hb2), (-wb * 0.45, curve_y, hb2), (0.0, curve_y * 1.2, hb3),
        (-wb * 0.8, -wb * 0.4, 0), (0, -wb * 0.4, 0), (-wb * 1.2, -curve_y * 0.5, hb1 * 0.8),
        (-wb * 0.6, -curve_y * 0.5, hb1 * 0.8), (-wb * 1.5, -curve_y * 0.8, hb2 * 0.8),
    ]
    faces_blade = [
        (0, 1, 2, 3), (3, 2, 4, 5), (5, 4, 6),
        (7, 8, 10, 9), (9, 10, 11),
    ]
    me_blade = bpy.data.meshes.new("BladeMesh")
    me_blade.from_pydata(verts_blade, [], faces_blade)
    me_blade.update()
    me_blade.materials.append(blade_mat)
    blade_obj = bpy.data.objects.new("BladeProto", me_blade)
    proto_coll.objects.link(blade_obj)

    # 2. Tall Grass: 6-blade crossed dense tuft (width 6.0 mm, height 5.2 cm)
    tg_mat = bpy.data.materials.new("TallGrassBladeMat")
    tg_mat.use_nodes = True
    tg_bsdf = tg_mat.node_tree.nodes["Principled BSDF"]
    tg_bsdf.inputs["Base Color"].default_value = (0.16, 0.48, 0.12, 1.0)
    tg_bsdf.inputs["Roughness"].default_value = 0.50
    if "Subsurface Weight" in tg_bsdf.inputs:
        tg_bsdf.inputs["Subsurface Weight"].default_value = 0.20

    wt = 0.0060 / 2.0
    ht = 0.052
    cy = 0.008

    v_tg = [
        (-wt, 0, 0), (wt, 0, 0), (wt * 0.7, cy * 0.5, ht * 0.5), (-wt * 0.7, cy * 0.5, ht * 0.5), (0, cy, ht),
        (0, -wt, 0), (0, wt, 0), (-cy * 0.5, wt * 0.7, ht * 0.5), (-cy * 0.5, -wt * 0.7, ht * 0.5), (-cy, 0, ht),
        (-wt * 0.8, -wt * 0.8, 0), (wt * 0.8, wt * 0.8, 0), (cy * 0.6, cy * 0.6, ht * 0.7),
        (-wt * 0.8, wt * 0.8, 0), (wt * 0.8, -wt * 0.8, 0), (-cy * 0.6, cy * 0.6, ht * 0.7),
    ]
    f_tg = [
        (0, 1, 2, 3), (3, 2, 4),
        (5, 6, 7, 8), (8, 7, 9),
        (10, 11, 12),
        (13, 14, 15),
    ]
    me_tg = bpy.data.meshes.new("TallGrassMesh")
    me_tg.from_pydata(v_tg, [], f_tg)
    me_tg.update()
    me_tg.materials.append(tg_mat)
    tg_obj = bpy.data.objects.new("TallGrassProto", me_tg)
    proto_coll.objects.link(tg_obj)

    # 3. Flower prototype
    fl_mat = bpy.data.materials.new("FlowerHeadMat")
    fl_mat.use_nodes = True
    fl_bsdf = fl_mat.node_tree.nodes["Principled BSDF"]
    fl_bsdf.inputs["Base Color"].default_value = (0.94, 0.22, 0.52, 1.0)
    fl_bsdf.inputs["Roughness"].default_value = 0.35

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=6, radius=0.002, depth=0.024, location=(0, 0, 0.012)
    )
    stalk = bpy.context.active_object
    stalk.data.materials.append(blade_mat)

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=1, radius=0.008, location=(0, 0, 0.026)
    )
    head = bpy.context.active_object
    head.data.materials.append(fl_mat)

    bpy.ops.object.select_all(action="DESELECT")
    stalk.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()

    flower_obj = bpy.context.active_object
    flower_obj.name = "FlowerProto"
    proto_coll.objects.link(flower_obj)
    if flower_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(flower_obj)

    return blade_obj, tg_obj, flower_obj


def add_grass_scatter_modifier(
    ground_obj: bpy.types.Object,
    mask_png_path: str | Path,
    blade_proto: bpy.types.Object,
    tg_proto: bpy.types.Object,
    flower_proto: bpy.types.Object,
    driver_obj: bpy.types.Object | None = None,
    player_obj: bpy.types.Object | None = None,
    shake_obj: bpy.types.Object | None = None,
    map_w: float = 6.40,
    map_h: float = 11.52,
) -> None:
    """Add Geometry Nodes 'Scatter' modifier to ground mesh with blade/flower instancing & scale randomisation."""
    mask_img = bpy.data.images.load(str(Path(mask_png_path).resolve()))

    mod = ground_obj.modifiers.new("Scatter", "NODES")
    ng = bpy.data.node_groups.new("ScatterTree", "GeometryNodeTree")
    mod.node_group = ng

    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    inp = ng.nodes.new("NodeGroupInput")
    outp = ng.nodes.new("NodeGroupOutput")

    join_all = ng.nodes.new("GeometryNodeJoinGeometry")
    ng.links.new(inp.outputs["Geometry"], join_all.inputs["Geometry"])

    def create_mask_sampler():
        tex = ng.nodes.new("GeometryNodeImageTexture")
        tex.inputs["Image"].default_value = mask_img
        tex.interpolation = "Closest"

        pos = ng.nodes.new("GeometryNodeInputPosition")
        sep = ng.nodes.new("ShaderNodeSeparateXYZ")
        ng.links.new(pos.outputs["Position"], sep.inputs["Vector"])

        math_x = ng.nodes.new("ShaderNodeMath")
        math_x.operation = "DIVIDE"
        math_x.inputs[1].default_value = map_w
        ng.links.new(sep.outputs["X"], math_x.inputs[0])

        math_y = ng.nodes.new("ShaderNodeMath")
        math_y.operation = "DIVIDE"
        math_y.inputs[1].default_value = map_h
        ng.links.new(sep.outputs["Y"], math_y.inputs[0])

        math_v = ng.nodes.new("ShaderNodeMath")
        math_v.operation = "ADD"
        math_v.inputs[0].default_value = 1.0
        ng.links.new(math_y.outputs["Value"], math_v.inputs[1])

        comb = ng.nodes.new("ShaderNodeCombineXYZ")
        ng.links.new(math_x.outputs["Value"], comb.inputs["X"])
        ng.links.new(math_v.outputs["Value"], comb.inputs["Y"])
        ng.links.new(comb.outputs["Vector"], tex.inputs["Vector"])

        sep_col = ng.nodes.new("FunctionNodeSeparateColor")
        ng.links.new(tex.outputs["Color"], sep_col.inputs["Color"])
        return sep_col

    def create_reveal_scaler():
        if driver_obj is None:
            return None
        obj_drv = ng.nodes.new("GeometryNodeObjectInfo")
        obj_drv.inputs["Object"].default_value = driver_obj
        obj_drv.transform_space = "RELATIVE"
        sep_drv = ng.nodes.new("ShaderNodeSeparateXYZ")
        ng.links.new(obj_drv.outputs["Location"], sep_drv.inputs["Vector"])

        pos_in = ng.nodes.new("GeometryNodeInputPosition")
        sep_p = ng.nodes.new("ShaderNodeSeparateXYZ")
        ng.links.new(pos_in.outputs["Position"], sep_p.inputs["Vector"])

        sub_eb = ng.nodes.new("ShaderNodeMath")
        sub_eb.operation = "SUBTRACT"
        ng.links.new(sep_drv.outputs["Y"], sub_eb.inputs[0])
        ng.links.new(sep_p.outputs["X"], sub_eb.inputs[1])

        div_eb = ng.nodes.new("ShaderNodeMath")
        div_eb.operation = "DIVIDE"
        div_eb.inputs[1].default_value = 0.15
        ng.links.new(sub_eb.outputs["Value"], div_eb.inputs[0])

        clamp_eb = ng.nodes.new("ShaderNodeClamp")
        clamp_eb.inputs["Min"].default_value = 0.0
        clamp_eb.inputs["Max"].default_value = 1.0
        ng.links.new(div_eb.outputs["Value"], clamp_eb.inputs["Value"])
        return clamp_eb.outputs["Result"]

    # 1. Lawn Grass branch (Density ~2800/m2)
    dist_grass = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist_grass.distribute_method = "RANDOM"
    dist_grass.inputs["Density"].default_value = 2800.0
    ng.links.new(inp.outputs["Geometry"], dist_grass.inputs["Mesh"])

    grass_col = create_mask_sampler()
    cmp_grass = ng.nodes.new("FunctionNodeCompare")
    cmp_grass.data_type = "FLOAT"
    cmp_grass.operation = "LESS_THAN"
    cmp_grass.inputs["B"].default_value = 0.15
    ng.links.new(grass_col.outputs["Red"], cmp_grass.inputs["A"])

    del_grass = ng.nodes.new("GeometryNodeDeleteGeometry")
    ng.links.new(dist_grass.outputs["Points"], del_grass.inputs["Geometry"])
    ng.links.new(cmp_grass.outputs["Result"], del_grass.inputs["Selection"])

    inst_grass = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(del_grass.outputs["Geometry"], inst_grass.inputs["Points"])

    obj_grass = ng.nodes.new("GeometryNodeObjectInfo")
    obj_grass.inputs["Object"].default_value = blade_proto
    obj_grass.transform_space = "RELATIVE"
    ng.links.new(obj_grass.outputs["Geometry"], inst_grass.inputs["Instance"])

    rand_sc_g = ng.nodes.new("FunctionNodeRandomValue")
    rand_sc_g.data_type = "FLOAT_VECTOR"
    rand_sc_g.inputs["Min"].default_value = (0.75, 0.75, 0.65)
    rand_sc_g.inputs["Max"].default_value = (1.25, 1.25, 1.35)

    growth_fac_g = create_reveal_scaler()
    if growth_fac_g:
        scale_g = ng.nodes.new("ShaderNodeVectorMath")
        scale_g.operation = "SCALE"
        ng.links.new(rand_sc_g.outputs["Value"], scale_g.inputs[0])
        ng.links.new(growth_fac_g, scale_g.inputs["Scale"])
        ng.links.new(scale_g.outputs["Vector"], inst_grass.inputs["Scale"])
    else:
        ng.links.new(rand_sc_g.outputs["Value"], inst_grass.inputs["Scale"])

    rot_grass = ng.nodes.new("GeometryNodeRotateInstances")
    ng.links.new(inst_grass.outputs["Instances"], rot_grass.inputs["Instances"])
    rand_yaw = ng.nodes.new("FunctionNodeRandomValue")
    rand_yaw.data_type = "FLOAT_VECTOR"
    rand_yaw.inputs["Min"].default_value = (0.0, 0.0, 0.0)
    rand_yaw.inputs["Max"].default_value = (0.0, 0.0, 6.283)
    ng.links.new(rand_yaw.outputs["Value"], rot_grass.inputs["Rotation"])
    ng.links.new(rot_grass.outputs["Instances"], join_all.inputs["Geometry"])

    # 2. Tall Grass branch (Density ~4500/m2)
    dist_tg = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist_tg.distribute_method = "RANDOM"
    dist_tg.inputs["Density"].default_value = 4500.0
    ng.links.new(inp.outputs["Geometry"], dist_tg.inputs["Mesh"])

    tg_col = create_mask_sampler()
    cmp_tg = ng.nodes.new("FunctionNodeCompare")
    cmp_tg.data_type = "FLOAT"
    cmp_tg.operation = "LESS_THAN"
    cmp_tg.inputs["B"].default_value = 0.50
    ng.links.new(tg_col.outputs["Green"], cmp_tg.inputs["A"])

    del_tg = ng.nodes.new("GeometryNodeDeleteGeometry")
    ng.links.new(dist_tg.outputs["Points"], del_tg.inputs["Geometry"])
    ng.links.new(cmp_tg.outputs["Result"], del_tg.inputs["Selection"])

    inst_tg = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(del_tg.outputs["Geometry"], inst_tg.inputs["Points"])

    obj_tg = ng.nodes.new("GeometryNodeObjectInfo")
    obj_tg.inputs["Object"].default_value = tg_proto
    obj_tg.transform_space = "RELATIVE"
    ng.links.new(obj_tg.outputs["Geometry"], inst_tg.inputs["Instance"])

    rand_sc_tg = ng.nodes.new("FunctionNodeRandomValue")
    rand_sc_tg.data_type = "FLOAT_VECTOR"
    rand_sc_tg.inputs["Min"].default_value = (0.80, 0.80, 0.70)
    rand_sc_tg.inputs["Max"].default_value = (1.20, 1.20, 1.40)

    growth_fac_tg = create_reveal_scaler()
    if growth_fac_tg:
        scale_tg = ng.nodes.new("ShaderNodeVectorMath")
        scale_tg.operation = "SCALE"
        ng.links.new(rand_sc_tg.outputs["Value"], scale_tg.inputs[0])
        ng.links.new(growth_fac_tg, scale_tg.inputs["Scale"])
        ng.links.new(scale_tg.outputs["Vector"], inst_tg.inputs["Scale"])
    else:
        ng.links.new(rand_sc_tg.outputs["Value"], inst_tg.inputs["Scale"])

    rot_tg = ng.nodes.new("GeometryNodeRotateInstances")
    ng.links.new(inst_tg.outputs["Instances"], rot_tg.inputs["Instances"])
    rand_tgyaw = ng.nodes.new("FunctionNodeRandomValue")
    rand_tgyaw.data_type = "FLOAT_VECTOR"
    rand_tgyaw.inputs["Min"].default_value = (0.0, 0.0, 0.0)
    rand_tgyaw.inputs["Max"].default_value = (0.0, 0.0, 6.283)

    if player_obj:
        obj_p = ng.nodes.new("GeometryNodeObjectInfo")
        obj_p.inputs["Object"].default_value = player_obj
        obj_p.transform_space = "RELATIVE"

        pos_tg = ng.nodes.new("GeometryNodeInputPosition")
        dist_p = ng.nodes.new("ShaderNodeVectorMath")
        dist_p.operation = "DISTANCE"
        ng.links.new(pos_tg.outputs["Position"], dist_p.inputs[0])
        ng.links.new(obj_p.outputs["Location"], dist_p.inputs[1])

        map_rustle = ng.nodes.new("ShaderNodeMapRange")
        map_rustle.interpolation_type = "SMOOTHSTEP"
        map_rustle.inputs["From Min"].default_value = 0.04
        map_rustle.inputs["From Max"].default_value = 0.14
        map_rustle.inputs["To Min"].default_value = 1.0
        map_rustle.inputs["To Max"].default_value = 0.0
        ng.links.new(dist_p.outputs["Value"], map_rustle.inputs["Value"])

        mult_angle = ng.nodes.new("ShaderNodeMath")
        mult_angle.operation = "MULTIPLY"
        mult_angle.inputs[1].default_value = math.radians(25.0)  # 25 deg rustle
        ng.links.new(map_rustle.outputs["Result"], mult_angle.inputs[0])

        add_rot = ng.nodes.new("ShaderNodeVectorMath")
        add_rot.operation = "ADD"
        ng.links.new(rand_tgyaw.outputs["Value"], add_rot.inputs[0])
        comb_rustle = ng.nodes.new("ShaderNodeCombineXYZ")
        ng.links.new(mult_angle.outputs["Value"], comb_rustle.inputs["X"])
        ng.links.new(mult_angle.outputs["Value"], comb_rustle.inputs["Y"])
        ng.links.new(comb_rustle.outputs["Vector"], add_rot.inputs[1])

        if shake_obj:
            obj_shk = ng.nodes.new("GeometryNodeObjectInfo")
            obj_shk.inputs["Object"].default_value = shake_obj
            obj_shk.transform_space = "RELATIVE"

            pos_shk = ng.nodes.new("GeometryNodeInputPosition")
            dist_shk = ng.nodes.new("ShaderNodeVectorMath")
            dist_shk.operation = "DISTANCE"
            ng.links.new(pos_shk.outputs["Position"], dist_shk.inputs[0])
            ng.links.new(obj_shk.outputs["Location"], dist_shk.inputs[1])

            map_shk = ng.nodes.new("ShaderNodeMapRange")
            map_shk.interpolation_type = "SMOOTHSTEP"
            map_shk.inputs["From Min"].default_value = 0.0
            map_shk.inputs["From Max"].default_value = 0.16
            map_shk.inputs["To Min"].default_value = 1.0
            map_shk.inputs["To Max"].default_value = 0.0
            ng.links.new(dist_shk.outputs["Value"], map_shk.inputs["Value"])

            sep_scale = ng.nodes.new("ShaderNodeSeparateXYZ")
            ng.links.new(obj_shk.outputs["Scale"], sep_scale.inputs["Vector"])

            mult_shk = ng.nodes.new("ShaderNodeMath")
            mult_shk.operation = "MULTIPLY"
            ng.links.new(map_shk.outputs["Result"], mult_shk.inputs[0])
            ng.links.new(sep_scale.outputs["X"], mult_shk.inputs[1])

            mult_shk_deg = ng.nodes.new("ShaderNodeMath")
            mult_shk_deg.operation = "MULTIPLY"
            mult_shk_deg.inputs[1].default_value = math.radians(35.0)
            ng.links.new(mult_shk.outputs["Value"], mult_shk_deg.inputs[0])

            comb_shk = ng.nodes.new("ShaderNodeCombineXYZ")
            ng.links.new(mult_shk_deg.outputs["Value"], comb_shk.inputs["X"])
            ng.links.new(mult_shk_deg.outputs["Value"], comb_shk.inputs["Y"])

            add_rot_total = ng.nodes.new("ShaderNodeVectorMath")
            add_rot_total.operation = "ADD"
            ng.links.new(add_rot.outputs["Vector"], add_rot_total.inputs[0])
            ng.links.new(comb_shk.outputs["Vector"], add_rot_total.inputs[1])
            ng.links.new(add_rot_total.outputs["Vector"], rot_tg.inputs["Rotation"])
        else:
            ng.links.new(add_rot.outputs["Vector"], rot_tg.inputs["Rotation"])
    else:
        ng.links.new(rand_tgyaw.outputs["Value"], rot_tg.inputs["Rotation"])

    ng.links.new(rot_tg.outputs["Instances"], join_all.inputs["Geometry"])

    # 3. Flowers branch (Density ~850/m2)
    dist_fl = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist_fl.distribute_method = "RANDOM"
    dist_fl.inputs["Density"].default_value = 850.0
    ng.links.new(inp.outputs["Geometry"], dist_fl.inputs["Mesh"])

    fl_col = create_mask_sampler()
    cmp_fl = ng.nodes.new("FunctionNodeCompare")
    cmp_fl.data_type = "FLOAT"
    cmp_fl.operation = "LESS_THAN"
    cmp_fl.inputs["B"].default_value = 0.50
    ng.links.new(fl_col.outputs["Blue"], cmp_fl.inputs["A"])

    del_fl = ng.nodes.new("GeometryNodeDeleteGeometry")
    ng.links.new(dist_fl.outputs["Points"], del_fl.inputs["Geometry"])
    ng.links.new(cmp_fl.outputs["Result"], del_fl.inputs["Selection"])

    inst_fl = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(del_fl.outputs["Geometry"], inst_fl.inputs["Points"])

    obj_fl = ng.nodes.new("GeometryNodeObjectInfo")
    obj_fl.inputs["Object"].default_value = flower_proto
    obj_fl.transform_space = "RELATIVE"
    ng.links.new(obj_fl.outputs["Geometry"], inst_fl.inputs["Instance"])

    growth_fac_fl = create_reveal_scaler()
    if growth_fac_fl:
        comb_fl = ng.nodes.new("ShaderNodeCombineXYZ")
        ng.links.new(growth_fac_fl, comb_fl.inputs["X"])
        ng.links.new(growth_fac_fl, comb_fl.inputs["Y"])
        ng.links.new(growth_fac_fl, comb_fl.inputs["Z"])
        ng.links.new(comb_fl.outputs["Vector"], inst_fl.inputs["Scale"])

    rot_fl = ng.nodes.new("GeometryNodeRotateInstances")
    ng.links.new(inst_fl.outputs["Instances"], rot_fl.inputs["Instances"])
    rand_flyaw = ng.nodes.new("FunctionNodeRandomValue")
    rand_flyaw.data_type = "FLOAT_VECTOR"
    rand_flyaw.inputs["Min"].default_value = (0.0, 0.0, 0.0)
    rand_flyaw.inputs["Max"].default_value = (0.0, 0.0, 6.283)
    ng.links.new(rand_flyaw.outputs["Value"], rot_fl.inputs["Rotation"])
    ng.links.new(rot_fl.outputs["Instances"], join_all.inputs["Geometry"])

    ng.links.new(join_all.outputs["Geometry"], outp.inputs["Geometry"])


def build_apron_and_hills(tree_proto: bpy.types.Object, map_w: float = 6.40, map_h: float = 11.52):
    """Build textured ground apron, perimeter trees, and distant rolling hills."""
    extras_coll = bpy.data.collections.new("TerrainAndHills")
    bpy.context.scene.collection.children.link(extras_coll)

    ax0 = -14.0
    ax1 = map_w + 14.0
    ay0 = 12.0
    ay1 = -map_h - 12.0
    z_apron = -0.015

    verts_apron = [
        (ax0, ay0, z_apron), (map_w * 0.5, ay0, z_apron), (ax1, ay0, z_apron),
        (ax1, -map_h * 0.5, z_apron), (ax1, ay1, z_apron),
        (map_w * 0.5, ay1, z_apron), (ax0, ay1, z_apron), (ax0, -map_h * 0.5, z_apron),
        (0.0, 0.0, z_apron), (map_w, 0.0, z_apron),
        (map_w, -map_h, z_apron), (0.0, -map_h, z_apron),
    ]
    faces_apron_clean = [
        (0, 1, 9, 8), (1, 2, 9), (2, 10, 9),
        (2, 3, 10), (3, 4, 10), (4, 5, 10), (5, 11, 10),
        (5, 6, 11), (6, 7, 11), (7, 8, 11), (7, 0, 8),
    ]

    me_apron = bpy.data.meshes.new("ApronMesh")
    me_apron.from_pydata(verts_apron, [], faces_apron_clean)
    me_apron.update()

    mat_apron = bpy.data.materials.new("ApronMat")
    mat_apron.use_nodes = True
    a_nt = mat_apron.node_tree
    bsdf_a = a_nt.nodes["Principled BSDF"]

    a_tex = a_nt.nodes.new("ShaderNodeTexCoord")
    a_noise = a_nt.nodes.new("ShaderNodeTexNoise")
    a_noise.inputs["Scale"].default_value = 1.2
    a_noise.inputs["Detail"].default_value = 2.0
    a_nt.links.new(a_tex.outputs["Object"], a_noise.inputs["Vector"])

    a_mix = a_nt.nodes.new("ShaderNodeMixRGB")
    a_mix.inputs["Color1"].default_value = (0.22, 0.54, 0.14, 1.0)
    a_mix.inputs["Color2"].default_value = (0.32, 0.62, 0.18, 1.0)
    a_nt.links.new(a_noise.outputs["Fac"], a_mix.inputs["Fac"])
    a_nt.links.new(a_mix.outputs["Color"], bsdf_a.inputs["Base Color"])
    bsdf_a.inputs["Roughness"].default_value = 0.85
    me_apron.materials.append(mat_apron)

    apron_obj = bpy.data.objects.new("Apron", me_apron)
    extras_coll.objects.link(apron_obj)

    cx_map = map_w * 0.5
    cy_map = -map_h * 0.5
    hill_size = 400.0
    N = 80
    xs = np.linspace(cx_map - hill_size * 0.5, cx_map + hill_size * 0.5, N + 1)
    ys = np.linspace(cy_map - hill_size * 0.5, cy_map + hill_size * 0.5, N + 1)

    verts_hills = []
    tree_line_points = []
    np.random.seed(101)

    for r in range(N + 1):
        y = ys[r]
        for c in range(N + 1):
            x = xs[c]
            d = math.hypot(x - cx_map, y - cy_map)
            if d < 14.0:
                z = z_apron
            else:
                blend = min(1.0, max(0.0, (d - 14.0) / 30.0))
                noise_val = 1.8 + 4.5 * math.sin(x * 0.012) * math.cos(y * 0.012) + 2.0 * math.cos(x * 0.035 + y * 0.025)
                z = z_apron + blend * max(0.0, noise_val)
                if d > 16.0 and d < 130.0 and z > 0.8:
                    if np.random.rand() < 0.045:
                        tree_line_points.append((x, y, z))
            verts_hills.append((x, y, z))

    faces_hills = []
    for r in range(N):
        for c in range(N):
            i0 = r * (N + 1) + c
            i1 = i0 + 1
            i2 = (r + 1) * (N + 1) + c + 1
            i3 = i2 - 1
            faces_hills.append((i0, i1, i2, i3))

    me_hills = bpy.data.meshes.new("HillsMesh")
    me_hills.from_pydata(verts_hills, [], faces_hills)
    me_hills.update()
    me_hills.materials.append(mat_apron)
    hills_obj = bpy.data.objects.new("Hills", me_hills)
    extras_coll.objects.link(hills_obj)

    all_extra_trees = tree_line_points
    if all_extra_trees:
        me_ht = bpy.data.meshes.new("HillsTreePoints")
        me_ht.from_pydata(all_extra_trees, [], [])
        me_ht.update()
        ht_obj = bpy.data.objects.new("HillsTrees", me_ht)
        extras_coll.objects.link(ht_obj)

        mod_ht = ht_obj.modifiers.new("HillsTreeScatter", "NODES")
        ng_ht = bpy.data.node_groups.new("HillsTreeScatterTree", "GeometryNodeTree")
        mod_ht.node_group = ng_ht

        ng_ht.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng_ht.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

        inp_ht = ng_ht.nodes.new("NodeGroupInput")
        outp_ht = ng_ht.nodes.new("NodeGroupOutput")

        inst_ht = ng_ht.nodes.new("GeometryNodeInstanceOnPoints")
        ng_ht.links.new(inp_ht.outputs["Geometry"], inst_ht.inputs["Points"])

        info_ht = ng_ht.nodes.new("GeometryNodeObjectInfo")
        info_ht.inputs["Object"].default_value = tree_proto
        info_ht.transform_space = "RELATIVE"
        ng_ht.links.new(info_ht.outputs["Geometry"], inst_ht.inputs["Instance"])

        scale_rand = ng_ht.nodes.new("FunctionNodeRandomValue")
        scale_rand.data_type = "FLOAT_VECTOR"
        scale_rand.inputs["Min"].default_value = (4.0, 4.0, 4.0)
        scale_rand.inputs["Max"].default_value = (8.0, 8.0, 8.0)
        ng_ht.links.new(scale_rand.outputs["Value"], inst_ht.inputs["Scale"])

        ng_ht.links.new(inst_ht.outputs["Instances"], outp_ht.inputs["Geometry"])


def setup_hook_compositor(sun_pos_track: np.ndarray | None = None):
    """Setup compositor with Mist Haze mix, Glare BLOOM, Glare STREAKS, and Sun Beams."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")

    # 1. Mist Pass Aerial Haze Mix (reduced multiplier so blue sky remains clear at top of frame)
    mult_mist = tree.nodes.new("CompositorNodeMath")
    mult_mist.operation = "MULTIPLY"
    mult_mist.inputs[1].default_value = 0.35
    tree.links.new(rl.outputs["Mist"], mult_mist.inputs[0])

    mix_mist = tree.nodes.new("CompositorNodeMixRGB")
    mix_mist.inputs[2].default_value = (0.75, 0.82, 0.90, 1.0)
    tree.links.new(mult_mist.outputs["Value"], mix_mist.inputs["Fac"])
    tree.links.new(rl.outputs["Image"], mix_mist.inputs[1])

    # 2. Glare BLOOM
    glare_bloom = tree.nodes.new("CompositorNodeGlare")
    glare_bloom.glare_type = "BLOOM"
    glare_bloom.inputs["Threshold"].default_value = 1.0
    glare_bloom.inputs["Size"].default_value = 7
    tree.links.new(mix_mist.outputs["Image"], glare_bloom.inputs["Image"])

    mix_bloom = tree.nodes.new("CompositorNodeMixRGB")
    mix_bloom.blend_type = "ADD"
    mix_bloom.inputs["Fac"].default_value = 0.16
    tree.links.new(mix_mist.outputs["Image"], mix_bloom.inputs[1])
    tree.links.new(glare_bloom.outputs["Glare"], mix_bloom.inputs[2])

    # 3. Glare STREAKS
    glare_streaks = tree.nodes.new("CompositorNodeGlare")
    glare_streaks.glare_type = "STREAKS"
    glare_streaks.inputs["Threshold"].default_value = 2.2
    glare_streaks.inputs["Streaks"].default_value = 4
    glare_streaks.inputs["Fade"].default_value = 0.90
    glare_streaks.inputs["Streaks Angle"].default_value = math.radians(45.0)
    tree.links.new(mix_bloom.outputs["Image"], glare_streaks.inputs["Image"])

    mix_streaks = tree.nodes.new("CompositorNodeMixRGB")
    mix_streaks.blend_type = "ADD"
    mix_streaks.inputs["Fac"].default_value = 0.10
    tree.links.new(mix_bloom.outputs["Image"], mix_streaks.inputs[1])
    tree.links.new(glare_streaks.outputs["Glare"], mix_streaks.inputs[2])

    # 4. Sun Beams node (driven from bright glints and sun disc)
    val_to_rgb = tree.nodes.new("CompositorNodeValToRGB")
    val_to_rgb.color_ramp.elements[0].position = 0.85
    val_to_rgb.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    val_to_rgb.color_ramp.elements[1].position = 1.00
    val_to_rgb.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    tree.links.new(mix_streaks.outputs["Image"], val_to_rgb.inputs["Fac"])

    sun_beams = tree.nodes.new("CompositorNodeSunBeams")
    if "Length" in sun_beams.inputs:
        sun_beams.inputs["Length"].default_value = 0.10
    elif hasattr(sun_beams, "ray_length"):
        sun_beams.ray_length = 0.10
    if "Source" in sun_beams.inputs:
        sun_beams.inputs["Source"].default_value = (0.28, 0.85)
    tree.links.new(val_to_rgb.outputs["Image"], sun_beams.inputs["Image"])

    mix_beams = tree.nodes.new("CompositorNodeMixRGB")
    mix_beams.blend_type = "ADD"
    mix_beams.inputs["Fac"].default_value = 0.06
    tree.links.new(mix_streaks.outputs["Image"], mix_beams.inputs[1])
    tree.links.new(sun_beams.outputs["Image"], mix_beams.inputs[2])

    tree.links.new(mix_beams.outputs["Image"], composite.inputs["Image"])


def build_exclamation_mark(exclamation_track: np.ndarray) -> bpy.types.Object:
    """Build 3D '!' popping above player's head with white face and black outline."""
    excl_coll = bpy.data.collections.get("Exclamation")
    if excl_coll is None:
        excl_coll = bpy.data.collections.new("Exclamation")
        bpy.context.scene.collection.children.link(excl_coll)

    font_curve = bpy.data.curves.new(type="FONT", name="ExclamationCurve")
    font_curve.body = "!"
    font_curve.font = bpy.data.fonts.load("C:/Windows/Fonts/seguibl.ttf")  # Segoe UI Black: a thin "!" was hard to see
    font_curve.extrude = 0.02
    font_curve.bevel_depth = 0.005
    font_curve.align_x = "CENTER"
    font_curve.align_y = "CENTER"

    mat_white = bpy.data.materials.new("ExclamationWhiteMat")
    mat_white.use_nodes = True
    bsdf_w = mat_white.node_tree.nodes["Principled BSDF"]
    bsdf_w.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    if "Emission Color" in bsdf_w.inputs:
        bsdf_w.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf_w.inputs["Emission Strength"].default_value = 0.5
    font_curve.materials.append(mat_white)

    excl_obj = bpy.data.objects.new("ExclamationMark", font_curve)
    excl_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(-15.0))
    excl_coll.objects.link(excl_obj)

    font_curve_outline = bpy.data.curves.new(type="FONT", name="ExclamationOutlineCurve")
    font_curve_outline.body = "!"
    font_curve_outline.font = font_curve.font
    font_curve_outline.extrude = 0.022
    font_curve_outline.offset = 0.035
    font_curve_outline.align_x = "CENTER"
    font_curve_outline.align_y = "CENTER"

    mat_black = bpy.data.materials.new("ExclamationBlackMat")
    mat_black.use_nodes = True
    bsdf_b = mat_black.node_tree.nodes["Principled BSDF"]
    bsdf_b.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    font_curve_outline.materials.append(mat_black)

    outline_obj = bpy.data.objects.new("ExclamationOutline", font_curve_outline)
    outline_obj.parent = excl_obj
    excl_coll.objects.link(outline_obj)

    F = len(exclamation_track)
    for f in range(F):
        x, y, z, s = exclamation_track[f]
        excl_obj.location = (float(x), float(y), float(z))
        s_val = float(s) * 0.16
        excl_obj.scale = (s_val, s_val, s_val)
        hide = (s_val <= 1e-4)
        excl_obj.hide_render = hide
        excl_obj.hide_viewport = hide
        outline_obj.hide_render = hide
        outline_obj.hide_viewport = hide

        excl_obj.keyframe_insert(data_path="location", frame=f)
        excl_obj.keyframe_insert(data_path="scale", frame=f)
        excl_obj.keyframe_insert(data_path="hide_render", frame=f)
        excl_obj.keyframe_insert(data_path="hide_viewport", frame=f)
        outline_obj.keyframe_insert(data_path="hide_render", frame=f)
        outline_obj.keyframe_insert(data_path="hide_viewport", frame=f)

    return excl_obj


def build_rattata_voxel(
    rattata_track: np.ndarray,
    sprite_rgb: np.ndarray,
    sprite_mask: np.ndarray,
) -> bpy.types.Object:
    """Build standup voxel Rattata model from real battle sprite color[122]."""
    coll = bpy.data.collections.get("Encounter")
    if coll is None:
        coll = bpy.data.collections.new("Encounter")
        bpy.context.scene.collection.children.link(coll)

    mat = bpy.data.materials.new("RattataMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "col"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Emission Color" in bsdf.inputs:  # flatter light, so the pixel art reads instead of noisy shading
        nt.links.new(vc.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.35

    root_obj = bpy.data.objects.new("WildRattata", None)
    root_obj.empty_display_type = "PLAIN_AXES"
    root_obj.empty_display_size = 0.05
    coll.objects.link(root_obj)

    H, W = sprite_mask.shape
    voxel_size = 0.0052  # 33 px tall -> 0.17 m, about the player's height so it reads
    verts = []
    faces = []
    cols = []
    v_idx = 0

    active_voxels = np.where(sprite_mask)
    for r_px, c_px in zip(*active_voxels):
        rgb = sprite_rgb[r_px, c_px]
        if int(rgb.astype(int).sum()) < 160:  # darkest shade -> near-black outline, like the player's sprite
            rgb = np.array([22, 12, 32], dtype=np.uint8)
        lin_col = srgb_to_linear(rgb)
        rgba = [float(lin_col[0]), float(lin_col[1]), float(lin_col[2]), 1.0]

        x0 = (c_px - W * 0.5) * voxel_size
        x1 = x0 + voxel_size
        z0 = (H - 1 - r_px) * voxel_size
        z1 = z0 + voxel_size

        for d in range(4):  # 4 voxels deep like the player
            y0 = (d - 2.0) * voxel_size
            y1 = y0 + voxel_size
            verts.extend([
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
            ])
            quad = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            for q in quad:
                faces.append((v_idx + q[0], v_idx + q[1], v_idx + q[2], v_idx + q[3]))
            v_idx += 8
            cols.extend([rgba] * 24)

    me = bpy.data.meshes.new("RattataVoxelMesh")
    me.from_pydata(verts, [], faces)
    me.update(calc_edges=True)

    col_attr = me.color_attributes.new("col", "FLOAT_COLOR", "CORNER")
    col_attr.data.foreach_set("color", np.array(cols, dtype=np.float32).reshape(-1))
    me.materials.append(mat)

    mesh_obj = bpy.data.objects.new("RattataMesh", me)
    # vertices are built upright (pixel rows along +Z), so only the yaw towards the camera; a 90-degree X rotation laid it flat
    mesh_obj.rotation_euler = (0.0, 0.0, math.radians(-15.0))
    mesh_obj.parent = root_obj
    coll.objects.link(mesh_obj)

    F = len(rattata_track)
    for f in range(F):
        x, y, z, s = rattata_track[f]
        root_obj.location = (float(x), float(y), float(z))
        s_val = float(s)
        root_obj.scale = (s_val, s_val, s_val)
        hide = (f < 665 or s_val <= 1e-4)
        root_obj.hide_render = hide
        root_obj.hide_viewport = hide
        mesh_obj.hide_render = hide
        mesh_obj.hide_viewport = hide

        root_obj.keyframe_insert(data_path="location", frame=f)
        root_obj.keyframe_insert(data_path="scale", frame=f)
        root_obj.keyframe_insert(data_path="hide_render", frame=f)
        root_obj.keyframe_insert(data_path="hide_viewport", frame=f)
        mesh_obj.keyframe_insert(data_path="hide_render", frame=f)
        mesh_obj.keyframe_insert(data_path="hide_viewport", frame=f)

    return root_obj


def build_3d_title_and_ball(pond_center: tuple[float, float, float] = (1.76, -4.16, 0.0), total_frames: int = 723):
    """Build 3D extruded title 'PokeDreamer' and stylized Pokéball icon rising from pond per DESIGN §2.7 and Round 4 review.
    
    Round 4 Owner Review:
    - Font: C:/Windows/Fonts/segoeuib.ttf (Segoe UI Bold)
    - Poké Ball split shader on letters (red top, 7% black middle line, white bottom), emission ~0.50
    - Black outline around every letter (child text object with curve.offset = 0.035, black mat, -0.003 m on extrusion axis)
    - Slower rise: rises 24-60 (1.2 s), fully risen at 60, held to 128, sinks 129-144, submerged f < 24 and f >= 145
    - In-plane scaling (X, Y) covers >= 55% at f=60 and ~80% at f=128, side margins >= 5% across 60-128
    - Extrusion depth (Z) scale remains CONSTANT at 0.18 (depth 0.018 m never grows!)
    """
    title_coll = bpy.data.collections.new("Title3D")
    bpy.context.scene.collection.children.link(title_coll)

    yaw_deg = 35.0
    u_rx = math.cos(math.radians(yaw_deg))
    u_ry = math.sin(math.radians(yaw_deg))

    # 1. 3D Text 'PokeDreamer'
    font_curve = bpy.data.curves.new(name="PokeDreamerFont", type="FONT")
    font_curve.body = "PokeDreamer"
    font_curve.font = bpy.data.fonts.load("C:/Windows/Fonts/segoeuib.ttf")  # Segoe UI Bold
    font_curve.extrude = 0.10
    font_curve.bevel_depth = 0.012
    font_curve.align_x = "CENTER"
    font_curve.align_y = "BOTTOM"
    font_curve.space_character = 1.08  # owner: a bit more space between letters

    text_obj = bpy.data.objects.new("TitleText", font_curve)
    text_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(yaw_deg))
    text_obj.location = (pond_center[0], pond_center[1], -0.20)
    title_coll.objects.link(text_obj)

    # Poké Ball split material for text
    mat_text = bpy.data.materials.new("TitleTextMat")
    mat_text.use_nodes = True
    t_nt = mat_text.node_tree
    t_bsdf = t_nt.nodes["Principled BSDF"]

    tex_coord = t_nt.nodes.new("ShaderNodeTexCoord")
    sep_pos = t_nt.nodes.new("ShaderNodeSeparateXYZ")
    t_nt.links.new(tex_coord.outputs["Object"], sep_pos.inputs["Vector"])

    div_h = t_nt.nodes.new("ShaderNodeMath")
    div_h.operation = "DIVIDE"
    div_h.inputs[1].default_value = 0.75  # Segoe UI Bold cap height in object space
    t_nt.links.new(sep_pos.outputs["Y"], div_h.inputs[0])

    t_ramp = t_nt.nodes.new("ShaderNodeValToRGB")
    t_ramp.color_ramp.interpolation = "CONSTANT"
    # Bottom white: sRGB #F4F4F4 -> linear (0.909, 0.909, 0.909, 1.0)
    t_ramp.color_ramp.elements[0].position = 0.0
    t_ramp.color_ramp.elements[0].color = (0.909, 0.909, 0.909, 1.0)
    # Middle black: 7% of cap height, centred on the middle of the lowercase letters (owner): x-height 500 / cap 701
    # units in segoeuib.ttf -> 0.357 of the cap height (0.322 to 0.392)
    el_mid = t_ramp.color_ramp.elements.new(0.322)
    el_mid.color = (0.01, 0.01, 0.01, 1.0)
    # Top red: sRGB #E3350D -> linear (0.768, 0.036, 0.003, 1.0)
    el_top = t_ramp.color_ramp.elements.new(0.392)
    el_top.color = (0.768, 0.036, 0.003, 1.0)

    t_nt.links.new(div_h.outputs["Value"], t_ramp.inputs["Fac"])
    t_nt.links.new(t_ramp.outputs["Color"], t_bsdf.inputs["Base Color"])
    t_bsdf.inputs["Roughness"].default_value = 0.25

    if "Emission Color" in t_bsdf.inputs:
        t_nt.links.new(t_ramp.outputs["Color"], t_bsdf.inputs["Emission Color"])
        t_bsdf.inputs["Emission Strength"].default_value = 0.50
    elif "Emission" in t_bsdf.inputs:
        t_nt.links.new(t_ramp.outputs["Color"], t_bsdf.inputs["Emission"])

    text_obj.data.materials.append(mat_text)

    # 2. Black outline child text object
    font_curve_outline = bpy.data.curves.new(name="PokeDreamerOutlineFont", type="FONT")
    font_curve_outline.body = "PokeDreamer"
    font_curve_outline.font = font_curve.font
    font_curve_outline.extrude = font_curve.extrude
    font_curve_outline.bevel_depth = font_curve.bevel_depth
    font_curve_outline.offset = 0.035
    font_curve_outline.align_x = "CENTER"
    font_curve_outline.align_y = "BOTTOM"
    font_curve_outline.space_character = font_curve.space_character

    outline_obj = bpy.data.objects.new("TitleTextOutline", font_curve_outline)
    outline_obj.parent = text_obj
    outline_obj.location = (0.0, 0.0, -0.003 / 0.18)  # -0.003 m on extrusion axis
    title_coll.objects.link(outline_obj)

    mat_outline = bpy.data.materials.new("TitleOutlineMat")
    mat_outline.use_nodes = True
    o_bsdf = mat_outline.node_tree.nodes["Principled BSDF"]
    o_bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1.0)
    o_bsdf.inputs["Roughness"].default_value = 0.80
    outline_obj.data.materials.append(mat_outline)

    # 3. Stylized Pokéball Sphere (UV sphere r 5.5 cm)
    rad_ball = 0.055
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=rad_ball, location=(pond_center[0], pond_center[1], -0.20)
    )
    ball_obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()  # owner: the faceted 24x16 sphere looked like square segments
    ball_obj.name = "PokeBallSphere"
    title_coll.objects.link(ball_obj)
    if ball_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(ball_obj)

    mat_ball = bpy.data.materials.new("PokeBallMat")
    mat_ball.use_nodes = True
    b_nt = mat_ball.node_tree
    b_bsdf = b_nt.nodes["Principled BSDF"]

    b_tex = b_nt.nodes.new("ShaderNodeTexCoord")
    b_sep = b_nt.nodes.new("ShaderNodeSeparateXYZ")
    b_nt.links.new(b_tex.outputs["Object"], b_sep.inputs["Vector"])

    b_ramp = b_nt.nodes.new("ShaderNodeValToRGB")
    b_ramp.color_ramp.interpolation = "CONSTANT"
    b_ramp.color_ramp.elements[0].position = 0.0
    b_ramp.color_ramp.elements[0].color = (0.909, 0.909, 0.909, 1.0)
    b_el_mid = b_ramp.color_ramp.elements.new(0.46)  # black band 8% of the diameter, like a real ball
    b_el_mid.color = (0.01, 0.01, 0.01, 1.0)
    b_el_top = b_ramp.color_ramp.elements.new(0.54)
    b_el_top.color = (0.768, 0.036, 0.003, 1.0)

    b_math_add = b_nt.nodes.new("ShaderNodeMath")
    b_math_add.operation = "ADD"
    b_math_add.inputs[1].default_value = rad_ball
    b_nt.links.new(b_sep.outputs["Z"], b_math_add.inputs[0])

    b_math_div = b_nt.nodes.new("ShaderNodeMath")
    b_math_div.operation = "DIVIDE"
    b_math_div.inputs[1].default_value = 2.0 * rad_ball
    b_nt.links.new(b_math_add.outputs["Value"], b_math_div.inputs[0])
    b_nt.links.new(b_math_div.outputs["Value"], b_ramp.inputs["Fac"])
    b_nt.links.new(b_ramp.outputs["Color"], b_bsdf.inputs["Base Color"])
    b_bsdf.inputs["Roughness"].default_value = 0.25

    if "Emission Color" in b_bsdf.inputs:
        b_nt.links.new(b_ramp.outputs["Color"], b_bsdf.inputs["Emission Color"])
        b_bsdf.inputs["Emission Strength"].default_value = 0.50

    ball_obj.data.materials.append(mat_ball)

    # Front button (white disc in a black ring) facing the camera like the text (yaw 35 deg); no spin, so it stays in view
    front = (math.sin(math.radians(35.0)), -math.cos(math.radians(35.0)), 0.0)
    for name, r_frac, d_frac, out_frac, col in (("PokeBallRing", 0.40, 0.10, 0.955, (0.01, 0.01, 0.01, 1.0)),
                                                ("PokeBallButton", 0.27, 0.10, 0.995, (0.95, 0.95, 0.95, 1.0))):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=r_frac * rad_ball, depth=d_frac * rad_ball,
                                            location=(0.0, 0.0, 0.0))
        part = bpy.context.active_object
        part.name = name
        bpy.ops.object.shade_smooth()
        part.rotation_euler = (math.radians(90.0), 0.0, math.radians(35.0))  # cylinder axis along the front direction
        part.location = tuple(out_frac * rad_ball * c for c in front)
        m = bpy.data.materials.new(name + "Mat")
        m.use_nodes = True
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = col
        m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.3
        part.data.materials.append(m)
        part.parent = ball_obj
        title_coll.objects.link(part)
        if part.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(part)

    # Keyframing across all frames:
    # 0..23: submerged (z = -0.20), scale = 0, hide_render = True
    # 24..60: rises out of pond water (1.2 s), scales 0 -> 0.4125 in-plane, z -> target_z
    # 60..128: full size, steady crane push-in (s_xy 0.4125 -> 0.1875, s_z = 0.18 constant!)
    # Owner review 2: 25% smaller than 0.55 -> 0.25, letter spacing 1.08.
    # Sizes measured on the render (r4_f128.png: text 795 px = 41% of the frame at s_xy 0.152, so the text is
    # ~3.5 font units wide); title + ball now cover ~57% at frame 60 and ~79% at frame 128 (lab/viz3d/tests/test_hook.py).
    # 126..140: sinks into pond, scales down to 0 (owner review 3: starts during the fade, done while the pond is in view)
    # 141+: hidden (z = -0.20, scale = 0, hide_render = True)
    target_z_text = 0.12
    target_z_ball = 0.14

    for f in range(total_frames):
        if f < 24:
            s_xy = 0.0
            z_t = -0.20
            z_b = -0.20
            hide = True
        elif f <= 60:
            t = (f - 24) / 36.0
            ease = 3.0 * t * t - 2.0 * t * t * t
            s_xy = ease * 0.4125
            z_t = -0.15 + ease * (target_z_text - (-0.15))
            z_b = -0.15 + ease * (target_z_ball - (-0.15))
            hide = False
        elif f <= 125:
            t = (f - 60) / 68.0
            ease = 3.0 * t * t - 2.0 * t * t * t
            s_xy = 0.4125 + (0.1875 - 0.4125) * ease
            z_t = target_z_text
            z_b = target_z_ball
            hide = False
        elif f <= 140:
            t = (140 - f) / 15.0
            s = max(0.0, 3.0 * t * t - 2.0 * t * t * t)
            s_xy = s * 0.1875
            z_t = s * target_z_text - (1.0 - s) * 0.15
            z_b = s * target_z_ball - (1.0 - s) * 0.15
            hide = False
        else:
            s_xy = 0.0
            z_t = -0.20
            z_b = -0.20
            hide = True

        # Position text and ball so assembly is perfectly centered on pond_center
        tx = pond_center[0] + 0.380 * s_xy * u_rx
        ty = pond_center[1] + 0.380 * s_xy * u_ry
        text_obj.location = (tx, ty, z_t)
        text_obj.scale = (s_xy, s_xy, 0.18)
        text_obj.hide_render = hide
        text_obj.hide_viewport = hide
        text_obj.keyframe_insert(data_path="scale", frame=f)
        text_obj.keyframe_insert(data_path="location", frame=f)
        text_obj.keyframe_insert(data_path="hide_render", frame=f)
        text_obj.keyframe_insert(data_path="hide_viewport", frame=f)

        outline_obj.hide_render = hide
        outline_obj.hide_viewport = hide
        outline_obj.keyframe_insert(data_path="hide_render", frame=f)
        outline_obj.keyframe_insert(data_path="hide_viewport", frame=f)

        # ball centre: text half-width 1.89 (1.75 x spacing 1.08) + gap 0.15 + ball radius 0.31 (font units)
        bx = tx - 2.5 * s_xy * u_rx
        by = ty - 2.5 * s_xy * u_ry
        bz = z_b + 0.38 * s_xy
        s_ball = (s_xy / 0.18) if s_xy > 0.0 else 0.0
        ball_obj.location = (bx, by, bz)
        ball_obj.scale = (s_ball, s_ball, s_ball)
        ball_obj.hide_render = hide
        ball_obj.hide_viewport = hide
        ball_obj.keyframe_insert(data_path="scale", frame=f)
        ball_obj.keyframe_insert(data_path="location", frame=f)
        ball_obj.keyframe_insert(data_path="hide_render", frame=f)
        ball_obj.keyframe_insert(data_path="hide_viewport", frame=f)

    return text_obj, ball_obj



def build_hook_camera(
    cam_pos_track: np.ndarray,
    cam_aim_track: np.ndarray,
    cam_focus_track: np.ndarray,
    cam_fstop_track: np.ndarray,
    topdown: bool = False,
    macro: bool = False,
    map_w: float = 6.40,
    map_h: float = 11.52,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Build camera, aim empty, and focus target empty with 40 mm lens and physical DOF."""
    cam_data = bpy.data.cameras.new("MainCamera")

    if topdown:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = max(map_w, map_h) * 1.05
        cam_data.dof.use_dof = False
        cam_obj = bpy.data.objects.new("MainCamera", cam_data)
        cam_obj.location = (map_w * 0.5, -map_h * 0.5, 20.0)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        return cam_obj, None

    if macro:
        cam_data.type = "PERSP"
        cam_data.lens = 40.0
        cam_data.clip_end = 600.0
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = 2.8

        aim_obj = bpy.data.objects.new("Aim", None)
        aim_obj.location = (3.520, -9.440, 0.020)
        bpy.context.scene.collection.objects.link(aim_obj)

        cam_data.dof.focus_object = aim_obj
        cam_obj = bpy.data.objects.new("MainCamera", cam_data)
        cam_obj.location = (3.622, -9.617, 0.163)

        track_to = cam_obj.constraints.new("TRACK_TO")
        track_to.target = aim_obj
        track_to.track_axis = "TRACK_NEGATIVE_Z"
        track_to.up_axis = "UP_Y"

        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        return cam_obj, aim_obj

    aim_obj = bpy.data.objects.new("Aim", None)
    aim_obj.empty_display_type = "PLAIN_AXES"
    aim_obj.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(aim_obj)

    focus_obj = bpy.data.objects.new("FocusTarget", None)
    focus_obj.empty_display_type = "SPHERE"
    focus_obj.empty_display_size = 0.04
    bpy.context.scene.collection.objects.link(focus_obj)

    cam_data.type = "PERSP"
    cam_data.lens = 40.0
    cam_data.clip_end = 600.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = focus_obj
    cam_data.dof.aperture_fstop = 2.8

    cam_obj = bpy.data.objects.new("MainCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    track_to = cam_obj.constraints.new("TRACK_TO")
    track_to.target = aim_obj
    track_to.track_axis = "TRACK_NEGATIVE_Z"
    track_to.up_axis = "UP_Y"

    F = len(cam_pos_track)
    for f in range(F):
        cam_obj.location = tuple(cam_pos_track[f])
        cam_obj.keyframe_insert(data_path="location", frame=f)

        aim_obj.location = tuple(cam_aim_track[f])
        aim_obj.keyframe_insert(data_path="location", frame=f)

        focus_obj.location = tuple(cam_focus_track[f])
        focus_obj.keyframe_insert(data_path="location", frame=f)

        cam_data.dof.aperture_fstop = float(cam_fstop_track[f])
        cam_data.dof.keyframe_insert(data_path="aperture_fstop", frame=f)

    return cam_obj, aim_obj


def main():
    if "--" in sys.argv:
        args_idx = sys.argv.index("--") + 1
        cli_args = sys.argv[args_idx:]
    else:
        cli_args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Blender Hook v2 diorama builder and renderer")
    parser.add_argument("world_npz", help="Path to world.npz")
    parser.add_argument("hook_tracks_npz", help="Path to hook_tracks.npz")
    parser.add_argument("out_dir", help="Output directory for rendered frames")
    parser.add_argument("--engine", default="cycles", choices=["cycles", "eevee"], help="Render engine")
    parser.add_argument("--frame", type=int, default=None, help="Render single frame")
    parser.add_argument("--frames", type=int, nargs=2, default=None, help="Render frame range [start, end]")
    parser.add_argument("--width", type=int, default=1920, help="Output image width")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel")
    parser.add_argument("--preview", action="store_true", help="Preview mode (960x540, 16 spp)")
    parser.add_argument("--topdown", action="store_true", help="Render orthographic top-down map view")
    parser.add_argument("--macro", action="store_true", help="Render macro grass close-up still")
    parser.add_argument("--check-round2", action="store_true", help="Render all Round 2 check frames with timing")
    parser.add_argument("--check-r3", action="store_true", help="Render the four Round 3 fixed check frames")
    parser.add_argument("--bench-repeat", action="store_true", help="Render frame twice in same process for timing benchmark")
    parser.add_argument("--out-name", default=None, help="Explicit output filename for single-frame renders")

    args = parser.parse_args(cli_args)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BLENDER HOOK V2 SCENE BUILD & RENDER")
    print(f"Engine: {args.engine.upper()} | Samples: {args.spp} | Width: {args.width}")
    print("=" * 65)

    clear_default_startup_scene()
    setup_color_management()

    width = 960 if args.preview else args.width
    height = width * 9 // 16
    spp = 16 if args.preview else args.spp
    configure_engine(args.engine, spp=spp)
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    if args.engine == "cycles":
        c_scene = bpy.context.scene.cycles
        c_scene.max_bounces = 8
        c_scene.transmission_bounces = 4
        c_scene.caustics_reflective = False
        c_scene.caustics_refractive = False
        c_scene.blur_glossy = 0.50

    w_data = np.load(args.world_npz)
    cls_map = w_data["cls"]
    shade_map = w_data["shade"]
    tree_blocks = w_data["tree_blocks"]
    fence_tiles = w_data["fence_tiles"]
    sign_tiles = w_data["sign_tiles"]
    pond_bed_depth = float(w_data.get("pond_bed_depth", -0.030))

    t_data = np.load(args.hook_tracks_npz)
    cam_pos = t_data["cam_pos"]
    cam_aim = t_data["cam_aim"]
    cam_focus = t_data["cam_focus"]
    cam_fstop = t_data["cam_fstop"]
    player_track = t_data["player_track"]
    edge_a_track = t_data["edge_a"]
    edge_b_track = t_data["edge_b"]
    sun_pos_track = t_data.get("sun_pos", None)

    H_px, W_px = cls_map.shape
    map_w = W_px * S
    map_h = H_px * S

    setup_nishita_world_and_sun()
    setup_hook_compositor(sun_pos_track)

    world_dir = Path(args.world_npz).resolve().parent
    ground_color_path = (world_dir / "ground_color.png").resolve()
    if ground_color_path.exists():
        ground_img = bpy.data.images.load(str(ground_color_path))
    else:
        from bpy_build import create_ground_texture
        ground_img = create_ground_texture(cls_map, shade_map)

    ground_obj, ground_mat, _ = build_ground_mesh(cls_map, shade_map, ground_img)

    # Setup LookState multi-texture mix (raw -> crisp -> color + cyan laser hairlines)
    driver_obj = setup_lookstate_ground_material(ground_mat, world_dir, edge_a_track, edge_b_track)

    for v in ground_obj.data.vertices:
        if abs(v.co.z - (-0.012)) < 1e-4:
            v.co.z = pond_bed_depth
    ground_obj.data.update()

    build_hook_water_surface(cls_map)
    build_solid_buildings(cls_map, ground_mat)
    build_fences(cls_map)
    fence_mat = bpy.data.materials.get("FenceMat")
    if fence_mat and driver_obj:
        setup_lookstate_solid_color(fence_mat, (0.50, 0.38, 0.24, 1.0), driver_obj, roughness=0.70)

    sign_boards = w_data.get("sign_boards", np.empty((0, 4), dtype=np.float32))
    build_sign_posts(sign_tiles, sign_boards=sign_boards, world_dir=world_dir, ground_mat=ground_mat, driver_obj=driver_obj)

    tree_proto = build_tree_prototypes(driver_obj=driver_obj)
    tree_pc_obj = build_tree_instances(tree_blocks, tree_proto)

    player_obj = None
    if "player_sprites_cls" in t_data:
        p_sprites_cls = t_data["player_sprites_cls"]
        p_sprites_sh = t_data["player_sprites_shade"]
        distinct_steps = t_data["distinct_steps"]
        player_hop = t_data.get("player_hop", None)
        player_obj = build_voxel_player_hook(
            player_track, p_sprites_cls, p_sprites_sh, distinct_steps, player_hop=player_hop
        )

    # Encounter: Rattata & Exclamation mark & Grass Shake Driver
    if "rattata_track" in t_data and "rattata_sprite_rgb" in t_data:
        build_rattata_voxel(t_data["rattata_track"], t_data["rattata_sprite_rgb"], t_data["rattata_sprite_mask"])
    if "exclamation_track" in t_data:
        build_exclamation_mark(t_data["exclamation_track"])

    shake_obj = None
    if "shake_track" in t_data:
        shake_obj = bpy.data.objects.new("GrassShakeDriver", None)
        shake_obj.empty_display_size = 0.01
        shake_obj.location = (4.08, -6.88, 0.0)
        bpy.context.scene.collection.objects.link(shake_obj)
        for f in range(len(t_data["shake_track"])):
            shake_obj.scale.x = float(t_data["shake_track"][f])
            shake_obj.keyframe_insert(data_path="scale", frame=f)

    blade_proto, tg_proto, flower_proto = build_blade_prototypes()
    mask_png_path = world_dir / "scatter_mask.png"
    add_grass_scatter_modifier(
        ground_obj, mask_png_path, blade_proto, tg_proto, flower_proto,
        driver_obj=driver_obj, player_obj=player_obj, shake_obj=shake_obj, map_w=map_w, map_h=map_h
    )

    build_apron_and_hills(tree_proto, map_w=map_w, map_h=map_h)
    build_3d_title_and_ball(pond_center=(1.76, -4.16, 0.0), total_frames=len(cam_pos))

    cam_obj, aim_obj = build_hook_camera(
        cam_pos, cam_aim, cam_focus, cam_fstop,
        topdown=args.topdown, macro=args.macro,
        map_w=map_w, map_h=map_h
    )

    print("\nScene built successfully:")
    print(f"  Ground: {map_w:.2f} m x {map_h:.2f} m")
    print(f"  Trees: {len(tree_blocks)} town instances")
    print(f"  Fences: {len(fence_tiles)} bollards")
    print(f"  Sign posts: {len(sign_tiles)} signs")

    scene = bpy.context.scene

    if args.topdown:
        out_file = out_dir / (args.out_name if args.out_name else "r1_topdown.png")
        scene.render.filepath = str(out_file)
        print(f"Rendering topdown orthographic to {out_file}...")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        dur = time.perf_counter() - t0
        print(f"Rendered {out_file.name} in {dur:.2f} s")
        return

    if args.macro:
        scene.frame_set(160)
        out_file = out_dir / (args.out_name if args.out_name else "r1_grass_macro.png")
        scene.render.filepath = str(out_file)
        print(f"Rendering grass macro to {out_file}...")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        dur = time.perf_counter() - t0
        print(f"Rendered {out_file.name} in {dur:.2f} s")
        return

    if args.check_round2:
        check_frames = [
            (45, "r2_f045_title.png"),
            (120, "r2_f120_title_low.png"),
            (185, "r2_f185_raw.png"),
            (215, "r2_f215_sweep.png"),
            (250, "r2_f250_colour.png"),
            (330, "r2_f330_follow.png"),
            (520, "r2_f520_route1.png"),
        ]
        print("\n" + "=" * 65)
        print("RENDERING ROUND 2 CHECK FRAMES (CYCLES 64 SPP)")
        print("=" * 65)
        for f_idx, out_fname in check_frames:
            check_gpu_temp_and_throttle()
            scene.frame_set(f_idx)
            out_file = out_dir / out_fname
            scene.render.filepath = str(out_file)

            t0 = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            dur = time.perf_counter() - t0
            print(f"Rendered {out_fname} (frame {f_idx:03d}) in {dur:.2f} s")

            # For frame 330 (the follow shot), benchmark second warm pass
            if f_idx == 330:
                t0 = time.perf_counter()
                bpy.ops.render.render(write_still=True)
                dur_warm = time.perf_counter() - t0
                print(f"[Benchmark Warm] Frame 330 repeated in {dur_warm:.2f} s")
        return

    if args.check_r3:
        check_frames = [
            (0, "r3_check_f000_wide.png"),
            (45, "r3_check_f045_title.png"),
            (215, "r3_check_f215_sweep.png"),
            (520, "r3_check_f520_route1.png"),
        ]
        print("\n" + "=" * 65)
        print("RENDERING ROUND 3 CHECK FRAMES (CYCLES 64 SPP)")
        print("=" * 65)
        for f_idx, out_fname in check_frames:
            check_gpu_temp_and_throttle()
            scene.frame_set(f_idx)
            out_file = out_dir / out_fname
            scene.render.filepath = str(out_file)

            t0 = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            dur = time.perf_counter() - t0
            print(f"Rendered {out_fname} (frame {f_idx:03d}) in {dur:.2f} s")
        return

    frames_to_render = []
    if args.frame is not None:
        frames_to_render = [args.frame]
        if args.bench_repeat:
            frames_to_render.append(args.frame)
    elif args.frames is not None:
        frames_to_render = list(range(args.frames[0], args.frames[1] + 1))
    else:
        frames_to_render = [0]

    measured_times = []
    for count, f in enumerate(frames_to_render):
        if count % 10 == 0:
            check_gpu_temp_and_throttle()

        scene.frame_set(f)
        if args.out_name and (count == 0 or not args.bench_repeat):
            out_file = out_dir / args.out_name
        else:
            out_file = out_dir / f"frame_{f:04d}.png"
        scene.render.filepath = str(out_file)

        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        dur = time.perf_counter() - t0
        measured_times.append(dur)
        print(f"Rendered frame {f:04d} in {dur:.2f} s -> {out_file}")

    if args.bench_repeat and len(measured_times) >= 2:
        print(f"\n[Benchmark Result] Frame {args.frame}:")
        print(f"  First render (cold/kernel): {measured_times[0]:.2f} s")
        print(f"  Second render (warm benchmark): {measured_times[1]:.2f} s")


if __name__ == "__main__":
    main()
