"""Blender 4.5 procedural 3D diorama builder and renderer."""
import argparse
import math
import os
import subprocess
import sys
import time

import bpy
import numpy as np

# Scale: 1 Game Boy pixel = 1 cm = 0.01 Blender units (meters)
S = 0.01

# Palette linear RGB ramps (sRGB -> linear) matching lab/color/palette.json exactly
PALETTE_SRGB = {
    "grass": [(220, 245, 180), (140, 210, 100), (70, 160, 60), (30, 80, 40)],
    "tall_grass": [(210, 240, 160), (120, 195, 80), (55, 140, 50), (25, 70, 35)],
    "tree": [(200, 235, 185), (100, 180, 110), (45, 120, 70), (20, 60, 40)],
    "path": [(245, 238, 215), (215, 195, 155), (160, 135, 95), (90, 70, 45)],
    "water": [(210, 245, 255), (110, 190, 240), (50, 130, 205), (20, 65, 130)],
    "ledge": [(240, 230, 200), (200, 180, 140), (140, 115, 80), (80, 60, 35)],
    "fence": [(235, 225, 205), (190, 165, 130), (130, 100, 70), (75, 50, 30)],
    "flowers": [(255, 230, 240), (240, 130, 160), (190, 60, 90), (100, 25, 50)],
    "building_wall": [(245, 245, 240), (200, 200, 195), (140, 140, 140), (70, 70, 75)],
    "roof_red": [(255, 225, 215), (235, 110, 95), (180, 55, 45), (100, 25, 20)],
    "roof_blue": [(220, 235, 255), (115, 160, 225), (55, 95, 175), (25, 45, 100)],
    "door": [(245, 230, 200), (195, 155, 100), (135, 95, 50), (70, 45, 20)],
    "sign": [(245, 240, 220), (205, 180, 135), (145, 115, 75), (80, 55, 30)],
    "floor_indoor": [(248, 242, 228), (218, 198, 168), (165, 138, 108), (98, 75, 52)],
    "counter": [(245, 240, 235), (205, 185, 170), (145, 120, 105), (80, 60, 50)],
    "carpet": [(250, 225, 225), (215, 105, 115), (160, 50, 65), (85, 20, 30)],
    "text_box": [(252, 252, 248), (195, 195, 200), (110, 110, 120), (35, 35, 45)],
    "sprite_player": [(250, 245, 240), (225, 85, 75), (150, 45, 40), (30, 30, 40)],
    "sprite_npc": [(245, 245, 240), (100, 175, 190), (45, 110, 130), (25, 45, 60)],
    "battle_bg": [(240, 248, 235), (180, 210, 175), (120, 150, 120), (50, 75, 60)],
    "battle_hp_bar": [(245, 250, 245), (120, 200, 120), (70, 150, 70), (30, 50, 35)],
    "other": [(245, 245, 245), (175, 180, 185), (105, 110, 120), (35, 38, 45)],
    "rock": [(235, 230, 220), (180, 170, 155), (120, 110, 95), (60, 55, 50)],
}

CLASS_NAMES = [
    "grass", "tall_grass", "tree", "path", "water", "ledge", "fence",
    "flowers", "building_wall", "roof_red", "roof_blue", "door", "sign",
    "floor_indoor", "counter", "carpet", "text_box", "sprite_player",
    "sprite_npc", "battle_bg", "battle_hp_bar", "other", "rock",
]


def srgb_to_linear(rgb):
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def setup_world():
    """Configure golden hour atmospheric world lighting with cool sky fill."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    bg = tree.nodes.new("ShaderNodeBackground")
    # Cool sky fill per feedback 3
    bg.inputs["Color"].default_value = (0.42, 0.60, 0.90, 1.0)
    bg.inputs["Strength"].default_value = 0.45

    out = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


def create_sun(sun_elevation_deg=14.0, sun_azimuth_deg=38.0, strength=5.5, angle_deg=3.5, topdown=False):
    """Create low warm golden-hour sun lamp with soft shadows (elevation 14 deg, azimuth 38 deg, ~4200 K)."""
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 4.0 if topdown else strength
    sun_data.angle = math.radians(angle_deg)
    # Warm golden amber (~4200 K)
    sun_data.color = (1.0, 0.78, 0.52)
    sun_obj = bpy.data.objects.new("Sun", sun_data)

    if topdown:
        sun_obj.rotation_euler = (math.radians(15.0), 0.0, math.radians(25.0))
    else:
        # Low golden hour sun (14 deg elevation, long soft shadows)
        rot_x = math.radians(90.0 - sun_elevation_deg)
        rot_z = math.radians(sun_azimuth_deg)
        sun_obj.rotation_euler = (rot_x, 0, rot_z)

    bpy.context.scene.collection.objects.link(sun_obj)
    return sun_obj


def create_gym_lights(map_w=2.24, map_h=1.60):
    """Create two warm area lights through high gym windows for Pewter Gym."""
    coll = bpy.context.scene.collection
    lights = []
    for i, (lx, ly, lz) in enumerate([
        (map_w * 0.25, 0.40, 1.40),
        (map_w * 0.75, 0.40, 1.40),
    ]):
        l_data = bpy.data.lights.new(f"GymLight_{i}", "AREA")
        l_data.energy = 16.0
        l_data.size = 0.5
        l_data.color = (1.0, 0.84, 0.62)
        l_obj = bpy.data.objects.new(f"GymLight_{i}", l_data)
        l_obj.location = (lx, ly, lz)
        l_obj.rotation_euler = (math.radians(-50.0), 0.0, 0.0)
        coll.objects.link(l_obj)
        lights.append(l_obj)
    return lights


def create_ground_texture(cls_map, shade_map, is_gym=False):
    """Generate crisp pixel texture from palette ramps, with deep navy-teal pond bed."""
    H, W = cls_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = cls_map == c_idx
        if not np.any(mask):
            continue
        ramp = PALETTE_SRGB.get(c_name, PALETTE_SRGB["other"])
        for sh in range(4):
            sh_mask = mask & (shade_map == sh)
            if np.any(sh_mask):
                if c_name == "water":
                    # Deep saturated navy-teal under water surface
                    lin = np.array([0.008, 0.06, 0.12], dtype=np.float32)
                elif is_gym and c_name == "path":
                    # Dark slate polished gym floor
                    lin = np.array([0.08, 0.09, 0.11], dtype=np.float32) * (0.8 + 0.2 * (3 - sh) / 3.0)
                else:
                    lin = srgb_to_linear(ramp[sh])
                rgba[sh_mask, :3] = lin
                rgba[sh_mask, 3] = 1.0

    # Flip vertically for Blender image UV orientation
    rgba_flipped = np.ascontiguousarray(rgba[::-1])
    img = bpy.data.images.new("GroundTexture", width=W, height=H, float_buffer=True)
    img.pixels.foreach_set(rgba_flipped.reshape(-1))
    return img


def build_ground_mesh(cls_map, shade_map, ground_img, is_gym=False):
    """Build tile terrain base with sunken pond bed, raised ledges, and vertical step skirts."""
    H_px, W_px = cls_map.shape
    H_tiles, W_tiles = H_px // 8, W_px // 8

    tile_heights = np.zeros((H_tiles, W_tiles), dtype=np.float32)
    for r in range(H_tiles):
        for c in range(W_tiles):
            c_patch = cls_map[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            dom_cls = int(np.bincount(c_patch.flatten()).argmax())
            dom_name = CLASS_NAMES[dom_cls] if dom_cls < len(CLASS_NAMES) else "other"

            if dom_name == "water":
                h = -0.012  # 1.2 cm sunken pond basin
            elif dom_name == "ledge":
                h = 0.040  # 4 cm raised step (0.25 tile)
            else:
                h = 0.000  # Flat path, grass, floors
            tile_heights[r, c] = h

    verts = []
    uvs = []
    faces = []
    v_idx = 0

    # 1. Tile top quads
    for r in range(H_tiles):
        for c in range(W_tiles):
            h = tile_heights[r, c]
            x0 = c * 8 * S
            x1 = (c + 1) * 8 * S
            y0 = -r * 8 * S
            y1 = -(r + 1) * 8 * S

            u0 = (c * 8) / float(W_px)
            u1 = ((c + 1) * 8) / float(W_px)
            v0 = 1.0 - (r * 8) / float(H_px)
            v1 = 1.0 - ((r + 1) * 8) / float(H_px)

            verts.extend([
                (x0, y0, h),
                (x1, y0, h),
                (x1, y1, h),
                (x0, y1, h),
            ])
            uvs.extend([
                (u0, v0),
                (u1, v0),
                (u1, v1),
                (u0, v1),
            ])
            faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
            v_idx += 4

    # 2. Vertical skirts between adjacent tiles with different heights
    for r in range(H_tiles):
        for c in range(W_tiles):
            h_cur = tile_heights[r, c]

            # South boundary with (r+1, c)
            if r + 1 < H_tiles:
                h_south = tile_heights[r + 1, c]
                if abs(h_cur - h_south) > 1e-4:
                    x0 = c * 8 * S
                    x1 = (c + 1) * 8 * S
                    y_edge = -(r + 1) * 8 * S
                    u0 = (c * 8) / float(W_px)
                    u1 = ((c + 1) * 8) / float(W_px)
                    v_edge = 1.0 - ((r + 1) * 8) / float(H_px)

                    verts.extend([
                        (x0, y_edge, h_cur),
                        (x1, y_edge, h_cur),
                        (x1, y_edge, h_south),
                        (x0, y_edge, h_south),
                    ])
                    uvs.extend([(u0, v_edge), (u1, v_edge), (u1, v_edge), (u0, v_edge)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

            # East boundary with (r, c+1)
            if c + 1 < W_tiles:
                h_east = tile_heights[r, c + 1]
                if abs(h_cur - h_east) > 1e-4:
                    x_edge = (c + 1) * 8 * S
                    y0 = -r * 8 * S
                    y1 = -(r + 1) * 8 * S
                    u_edge = ((c + 1) * 8) / float(W_px)
                    v0 = 1.0 - (r * 8) / float(H_px)
                    v1 = 1.0 - ((r + 1) * 8) / float(H_px)

                    verts.extend([
                        (x_edge, y0, h_cur),
                        (x_edge, y1, h_cur),
                        (x_edge, y1, h_east),
                        (x_edge, y0, h_east),
                    ])
                    uvs.extend([(u_edge, v0), (u_edge, v1), (u_edge, v1), (u_edge, v0)])
                    faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                    v_idx += 4

    me = bpy.data.meshes.new("GroundMesh")
    me.from_pydata(verts, [], faces)
    me.update(calc_edges=True)

    uv_layer = me.uv_layers.new(name="UVMap")
    uv_data = np.array(uvs, dtype=np.float32).reshape(-1)
    uv_layer.data.foreach_set("uv", uv_data)

    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if is_gym:
        # Polished dark gym floor with ray-traced reflections
        bsdf.inputs["Roughness"].default_value = 0.12
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.85
    else:
        bsdf.inputs["Roughness"].default_value = 0.85

    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = ground_img
    tex_node.interpolation = "Closest"
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    me.materials.append(mat)
    obj = bpy.data.objects.new("Ground", me)
    bpy.context.scene.collection.objects.link(obj)

    return obj, mat, tile_heights


def build_water_surface(cls_map):
    """Build glassy, refractive teal water surface with animated ripple normals and sharp sun glints."""
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
    h = -0.005  # Water surface: 5 mm below grass, 7 mm above pond bed

    verts = [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
    faces = [(0, 1, 2, 3)]

    me = bpy.data.meshes.new("WaterSurfaceMesh")
    me.from_pydata(verts, [], faces)
    me.update()

    mat = bpy.data.materials.new("WaterMat")
    mat.use_nodes = True
    mat.use_raytrace_refraction = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    # Deeper saturated teal-blue base color; coat weight 0 and standard IOR level to prevent silvery blowout
    bsdf.inputs["Base Color"].default_value = (0.01, 0.18, 0.32, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.035
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 0.90
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.45
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0

    # High-frequency animated 4D ripples
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 48.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.50

    fcurve = noise.inputs["W"].driver_add("default_value")
    fcurve.driver.expression = "frame * 0.05"

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    me.materials.append(mat)
    obj = bpy.data.objects.new("WaterSurface", me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def build_procedural_trees(cls_map):
    """Build 3D trees (trunk + 1.2-1.6 tiles tall rounded blob canopy) with wind sway."""
    H_px, W_px = cls_map.shape
    tree_idx = CLASS_NAMES.index("tree")

    tree_coll = bpy.data.collections.new("Trees")
    bpy.context.scene.collection.children.link(tree_coll)

    trunk_mat = bpy.data.materials.new("TrunkMat")
    trunk_mat.use_nodes = True
    trunk_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.24, 0.15, 0.08, 1.0)
    trunk_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.85

    canopy_mat = bpy.data.materials.new("CanopyMat")
    canopy_mat.use_nodes = True
    c_bsdf = canopy_mat.node_tree.nodes["Principled BSDF"]
    c_bsdf.inputs["Base Color"].default_value = (0.12, 0.42, 0.16, 1.0)
    c_bsdf.inputs["Roughness"].default_value = 0.65
    if "Subsurface Weight" in c_bsdf.inputs:
        c_bsdf.inputs["Subsurface Weight"].default_value = 0.12

    count = 0
    for r in range(0, H_px - 15, 16):
        for c in range(0, W_px - 15, 16):
            patch = cls_map[r : r + 16, c : c + 16]
            if np.mean(patch == tree_idx) > 0.55:
                cx = (c + 8) * S
                cy = -(r + 8) * S

                # 1. Trunk: r 1.4 cm, height 8 cm, center z = 4 cm
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.014, depth=0.08, location=(cx, cy, 0.04)
                )
                trunk_obj = bpy.context.active_object
                trunk_obj.name = f"Trunk_{count}"
                trunk_obj.data.materials.append(trunk_mat)
                if trunk_obj.name not in tree_coll.objects:
                    tree_coll.objects.link(trunk_obj)
                if trunk_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(trunk_obj)

                # 2. Main Canopy: r 8.5 cm at z 13 cm (total height ~22 cm = 1.4 tiles)
                bpy.ops.mesh.primitive_ico_sphere_add(
                    subdivisions=2, radius=0.085, location=(cx, cy, 0.13)
                )
                canopy_obj = bpy.context.active_object
                canopy_obj.name = f"Canopy_{count}"
                canopy_obj.scale = (1.0, 1.0, 0.85)
                canopy_obj.data.materials.append(canopy_mat)

                phase = (cx * 7.91 + cy * 11.37) % 6.28
                drv_x = canopy_obj.driver_add("rotation_euler", 0)
                drv_x.driver.expression = f"0.035 * sin(frame * 0.08 + {phase:.3f})"
                drv_y = canopy_obj.driver_add("rotation_euler", 1)
                drv_y.driver.expression = f"0.025 * cos(frame * 0.06 + {phase:.3f})"

                if canopy_obj.name not in tree_coll.objects:
                    tree_coll.objects.link(canopy_obj)
                if canopy_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(canopy_obj)
                count += 1

    return tree_coll


def build_tall_grass(cls_map):
    """Build 3D tall grass blades (~0.3 tile tall = 4.8 cm) with wind flutter."""
    H_px, W_px = cls_map.shape
    grass_coll = bpy.data.collections.new("TallGrass")
    bpy.context.scene.collection.children.link(grass_coll)

    tg_idx = CLASS_NAMES.index("tall_grass") if "tall_grass" in CLASS_NAMES else -1
    if tg_idx < 0:
        return grass_coll

    mat = bpy.data.materials.new("TallGrassMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.18, 0.52, 0.14, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.60

    count = 0
    for r in range(0, H_px - 7, 8):
        for c in range(0, W_px - 7, 8):
            patch = cls_map[r : r + 8, c : c + 8]
            if np.mean(patch == tg_idx) > 0.50:
                cx = (c + 4) * S
                cy = -(r + 4) * S

                h = 0.048
                w = 0.035
                verts = [
                    (cx - w, cy, 0.0), (cx + w, cy, 0.0), (cx + w, cy, h), (cx - w, cy, h),
                    (cx, cy - w, 0.0), (cx, cy + w, 0.0), (cx, cy + w, h), (cx, cy - w, h),
                ]
                faces = [(0, 1, 2, 3), (4, 5, 6, 7)]

                me = bpy.data.meshes.new(f"TuftMesh_{count}")
                me.from_pydata(verts, [], faces)
                me.update()
                me.materials.append(mat)

                obj = bpy.data.objects.new(f"TallGrass_{count}", me)
                grass_coll.objects.link(obj)

                phase = (cx * 15.3 + cy * 19.1) % 6.28
                drv = obj.driver_add("rotation_euler", 0)
                drv.driver.expression = f"0.06 * sin(frame * 0.12 + {phase:.3f})"
                count += 1

    return grass_coll


def build_fences(cls_map):
    """Build clean round bollards (0.45 tile diameter = 7.2 cm, 0.6 tile tall = 9.6 cm, rounded top) at tile centers."""
    H_px, W_px = cls_map.shape
    fence_coll = bpy.data.collections.new("Fences")
    bpy.context.scene.collection.children.link(fence_coll)

    fence_idx = CLASS_NAMES.index("fence") if "fence" in CLASS_NAMES else -1
    if fence_idx < 0:
        return fence_coll

    mat = bpy.data.materials.new("FenceMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.50, 0.38, 0.24, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.70

    # Build prototype round bollard mesh: cylinder body + rounded dome cap
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=0.036, depth=0.080, location=(0.0, 0.0, 0.040)
    )
    cyl_obj = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16, ring_count=8, radius=0.036, location=(0.0, 0.0, 0.080)
    )
    sphere_obj = bpy.context.active_object
    sphere_obj.scale = (1.0, 1.0, 0.44)

    bpy.ops.object.select_all(action="DESELECT")
    cyl_obj.select_set(True)
    sphere_obj.select_set(True)
    bpy.context.view_layer.objects.active = cyl_obj
    bpy.ops.object.join()

    proto_obj = cyl_obj
    proto_obj.name = "BollardProto"
    proto_obj.data.materials.append(mat)
    if proto_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(proto_obj)

    count = 0
    # Place exactly one bollard at center of each 16x16 fence tile
    for r in range(0, H_px - 15, 16):
        for c in range(0, W_px - 15, 16):
            patch = cls_map[r : r + 16, c : c + 16]
            if np.mean(patch == fence_idx) > 0.35:
                cx = (c + 8) * S
                cy = -(r + 8) * S

                b_obj = bpy.data.objects.new(f"Bollard_{count}", proto_obj.data)
                b_obj.location = (cx, cy, 0.0)
                fence_coll.objects.link(b_obj)
                count += 1

    return fence_coll


def build_flowers(cls_map):
    """Build tiny 3D flowers with gentle sway."""
    H_px, W_px = cls_map.shape
    fl_coll = bpy.data.collections.new("Flowers")
    bpy.context.scene.collection.children.link(fl_coll)

    fl_idx = CLASS_NAMES.index("flowers") if "flowers" in CLASS_NAMES else -1
    if fl_idx < 0:
        return fl_coll

    mat = bpy.data.materials.new("FlowerMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.88, 0.22, 0.52, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.50

    count = 0
    for r in range(0, H_px - 7, 8):
        for c in range(0, W_px - 7, 8):
            patch = cls_map[r : r + 8, c : c + 8]
            if np.mean(patch == fl_idx) > 0.40:
                cx = (c + 4) * S
                cy = -(r + 4) * S

                bpy.ops.mesh.primitive_ico_sphere_add(
                    subdivisions=1, radius=0.015, location=(cx, cy, 0.025)
                )
                fl_obj = bpy.context.active_object
                fl_obj.name = f"Flower_{count}"
                fl_obj.data.materials.append(mat)

                phase = (cx * 21.3 + cy * 17.5) % 6.28
                drv = fl_obj.driver_add("rotation_euler", 0)
                drv.driver.expression = f"0.05 * sin(frame * 0.10 + {phase:.3f})"

                if fl_obj.name not in fl_coll.objects:
                    fl_coll.objects.link(fl_obj)
                if fl_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(fl_obj)
                count += 1

    return fl_coll


def build_buildings_and_roofs(cls_map, ground_mat):
    """Build 3D textured building front walls (doors, windows, signs) and clay roofs."""
    H_px, W_px = cls_map.shape
    H_tiles, W_tiles = H_px // 8, W_px // 8

    wall_names = {"building_wall", "door", "sign"}
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

    if not np.any(is_wall) and not np.any(is_roof):
        return None

    verts = []
    uvs = []
    faces = []
    v_idx = 0

    z_wall = 0.28  # 1.75 tiles tall
    z_roof = 0.42  # 2.62 tiles tall

    # 1. Textured building walls
    for r in range(H_tiles):
        for c in range(W_tiles):
            if not is_wall[r, c]:
                continue
            x0 = c * 8 * S
            x1 = (c + 1) * 8 * S
            y0 = -r * 8 * S
            y1 = -(r + 1) * 8 * S

            u0 = (c * 8) / float(W_px)
            u1 = ((c + 1) * 8) / float(W_px)
            v0 = 1.0 - (r * 8) / float(H_px)
            v1 = 1.0 - ((r + 1) * 8) / float(H_px)

            # Front vertical face (facing South at y1, visible to camera looking North)
            # Displaying the exact tile texture (door, windows, sign)!
            verts.extend([
                (x0, y1, 0.0),
                (x1, y1, 0.0),
                (x1, y1, z_wall),
                (x0, y1, z_wall),
            ])
            uvs.extend([
                (u0, v1),
                (u1, v1),
                (u1, v0),
                (u0, v0),
            ])
            faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
            v_idx += 4

            # West side face (if no wall to west)
            if c == 0 or not is_wall[r, c - 1]:
                verts.extend([
                    (x0, y0, 0.0),
                    (x0, y1, 0.0),
                    (x0, y1, z_wall),
                    (x0, y0, z_wall),
                ])
                uvs.extend([(u0, v0), (u0, v1), (u0, v1), (u0, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            # East side face (if no wall to east)
            if c + 1 == W_tiles or not is_wall[r, c + 1]:
                verts.extend([
                    (x1, y1, 0.0),
                    (x1, y0, 0.0),
                    (x1, y0, z_wall),
                    (x1, y1, z_wall),
                ])
                uvs.extend([(u1, v1), (u1, v0), (u1, v0), (u1, v1)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            # North back face (if no wall to north)
            if r == 0 or not is_wall[r - 1, c]:
                verts.extend([
                    (x1, y0, 0.0),
                    (x0, y0, 0.0),
                    (x0, y0, z_wall),
                    (x1, y0, z_wall),
                ])
                uvs.extend([(u1, v0), (u0, v0), (u0, v0), (u1, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

    # 2. Textured roofs
    for r in range(H_tiles):
        for c in range(W_tiles):
            if not is_roof[r, c]:
                continue
            x0 = c * 8 * S
            x1 = (c + 1) * 8 * S
            y0 = -r * 8 * S
            y1 = -(r + 1) * 8 * S

            u0 = (c * 8) / float(W_px)
            u1 = ((c + 1) * 8) / float(W_px)
            v0 = 1.0 - (r * 8) / float(H_px)
            v1 = 1.0 - ((r + 1) * 8) / float(H_px)

            # Top horizontal face at z_roof (displaying the terracotta roof texture!)
            verts.extend([
                (x0, y0, z_roof),
                (x1, y0, z_roof),
                (x1, y1, z_roof),
                (x0, y1, z_roof),
            ])
            uvs.extend([
                (u0, v0),
                (u1, v0),
                (u1, v1),
                (u0, v1),
            ])
            faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
            v_idx += 4

            # South overhang skirt (from z_wall to z_roof)
            if r + 1 == H_tiles or not is_roof[r + 1, c]:
                verts.extend([
                    (x0, y1, z_wall),
                    (x1, y1, z_wall),
                    (x1, y1, z_roof),
                    (x0, y1, z_roof),
                ])
                uvs.extend([(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            # North overhang skirt
            if r == 0 or not is_roof[r - 1, c]:
                verts.extend([
                    (x1, y0, z_wall),
                    (x0, y0, z_wall),
                    (x0, y0, z_roof),
                    (x1, y0, z_roof),
                ])
                uvs.extend([(u1, v0), (u0, v0), (u0, v0), (u1, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            # West overhang skirt
            if c == 0 or not is_roof[r, c - 1]:
                verts.extend([
                    (x0, y0, z_wall),
                    (x0, y1, z_wall),
                    (x0, y1, z_roof),
                    (x0, y0, z_roof),
                ])
                uvs.extend([(u0, v0), (u0, v1), (u0, v1), (u0, v0)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

            # East overhang skirt
            if c + 1 == W_tiles or not is_roof[r, c + 1]:
                verts.extend([
                    (x1, y1, z_wall),
                    (x1, y0, z_wall),
                    (x1, y0, z_roof),
                    (x1, y1, z_roof),
                ])
                uvs.extend([(u1, v1), (u1, v0), (u1, v0), (u1, v1)])
                faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
                v_idx += 4

    b_me = bpy.data.meshes.new("BuildingsMesh")
    b_me.from_pydata(verts, [], faces)
    b_me.update(calc_edges=True)

    uv_layer = b_me.uv_layers.new(name="UVMap")
    uv_data = np.array(uvs, dtype=np.float32).reshape(-1)
    uv_layer.data.foreach_set("uv", uv_data)

    b_me.materials.append(ground_mat)

    b_obj = bpy.data.objects.new("Buildings", b_me)
    bpy.context.scene.collection.objects.link(b_obj)
    return b_obj


def build_voxel_player(player_track, player_sprites_cls, player_sprites_shade, distinct_steps):
    """Build standup voxel character with exact stage-1 red player palette."""
    st_idx = 0
    cls_crop = player_sprites_cls[st_idx]
    shade_crop = player_sprites_shade[st_idx]

    player_cls_idx = CLASS_NAMES.index("sprite_player")
    active_voxels = np.where(cls_crop == player_cls_idx)

    verts = []
    faces = []
    cols = []
    v_idx = 0

    ramp = PALETTE_SRGB["sprite_player"]

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

    me = bpy.data.meshes.new("PlayerVoxelMesh")
    me.from_pydata(verts, [], faces)
    me.update(calc_edges=True)

    col_attr = me.color_attributes.new("col", "FLOAT_COLOR", "CORNER")
    col_attr.data.foreach_set("color", np.array(cols, dtype=np.float32).reshape(-1))

    mat = bpy.data.materials.new("PlayerMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "col"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65
    me.materials.append(mat)

    player_obj = bpy.data.objects.new("Player", me)
    bpy.context.scene.collection.objects.link(player_obj)

    F = len(player_track)
    for f in range(F):
        px, py, facing_step = player_track[f]
        bob = 0.003 * math.sin(f * 0.8) if (f >= 105 and f < 220) else 0.0
        player_obj.location = (px * S, -py * S, bob + 0.005)
        player_obj.keyframe_insert(data_path="location", frame=f)

    return player_obj


def build_brock_voxel(brock_cls, brock_shade, pos_gb=(72.0, 24.0)):
    """Build standup voxel character for Gym Leader Brock facing South."""
    active_voxels = np.where(brock_cls != -1)
    if len(active_voxels[0]) == 0:
        return None

    verts = []
    faces = []
    cols = []
    v_idx = 0
    ramp = PALETTE_SRGB["sprite_npc"]

    for r_px, c_px in zip(*active_voxels):
        sh = brock_shade[r_px, c_px]
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

    me = bpy.data.meshes.new("BrockVoxelMesh")
    me.from_pydata(verts, [], faces)
    me.update(calc_edges=True)

    col_attr = me.color_attributes.new("col", "FLOAT_COLOR", "CORNER")
    col_attr.data.foreach_set("color", np.array(cols, dtype=np.float32).reshape(-1))

    mat = bpy.data.materials.new("BrockMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "col"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65
    me.materials.append(mat)

    brock_obj = bpy.data.objects.new("Brock", me)
    bpy.context.scene.collection.objects.link(brock_obj)
    brock_obj.location = (pos_gb[0] * S, -pos_gb[1] * S, 0.005)
    return brock_obj


def build_gym_boulders(cls_map):
    """Build 3D boulders for Pewter Gym."""
    H_px, W_px = cls_map.shape
    rock_idx = CLASS_NAMES.index("rock") if "rock" in CLASS_NAMES else -1
    if rock_idx < 0:
        return None
    boulder_coll = bpy.data.collections.new("GymBoulders")
    bpy.context.scene.collection.children.link(boulder_coll)

    mat = bpy.data.materials.new("BoulderMat")
    mat.use_nodes = True
    b_bsdf = mat.node_tree.nodes["Principled BSDF"]
    b_bsdf.inputs["Base Color"].default_value = (0.28, 0.27, 0.26, 1.0)
    b_bsdf.inputs["Roughness"].default_value = 0.70

    count = 0
    for r in range(0, H_px - 15, 16):
        for c in range(0, W_px - 15, 16):
            patch = cls_map[r : r + 16, c : c + 16]
            if np.mean(patch == rock_idx) > 0.40:
                cx = (c + 8) * S
                cy = -(r + 8) * S
                bpy.ops.mesh.primitive_ico_sphere_add(
                    subdivisions=2, radius=0.075, location=(cx, cy, 0.065)
                )
                b_obj = bpy.context.active_object
                b_obj.name = f"Boulder_{count}"
                b_obj.scale = (1.0, 1.0, 0.85)
                b_obj.data.materials.append(mat)
                if b_obj.name not in boulder_coll.objects:
                    boulder_coll.objects.link(b_obj)
                if b_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(b_obj)
                count += 1
    return boulder_coll


def build_dream_twin(dream_cls, dream_shade, reanchors, steps_per_frame, map_w=3.20, map_h=11.52):
    """Build the ghosted dream diorama next to the real Route 2 diorama with re-anchor flash pulses."""
    twin_coll = bpy.data.collections.new("DreamTwin")
    bpy.context.scene.collection.children.link(twin_coll)

    x_offset = map_w + 0.35  # 35 cm gap to the right

    # Ghost material: cyan-grey emission
    g_mat = bpy.data.materials.new("DreamGhostMat")
    g_mat.use_nodes = True
    nt = g_mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.18, 0.42, 0.48, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.50
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.25, 0.85, 0.95, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.35

    # 1. Base diorama table for dream
    dw = 1.60
    dh = 1.44
    verts = [
        (x_offset, 0.0, 0.0), (x_offset + dw, 0.0, 0.0),
        (x_offset + dw, -dh, 0.0), (x_offset, -dh, 0.0)
    ]
    me = bpy.data.meshes.new("DreamGroundMesh")
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    me.update()
    me.materials.append(g_mat)
    d_obj = bpy.data.objects.new("DreamGround", me)
    twin_coll.objects.link(d_obj)

    # 2. Keyframe reanchor flash pulse at steps 6874 and 6884
    F = len(steps_per_frame)
    for f in range(F):
        st = steps_per_frame[f]
        is_flash = (st in [6874, 6884]) and ((f % 8) < 3)
        str_val = 3.5 if is_flash else 0.35
        bsdf.inputs["Emission Strength"].default_value = str_val
        bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)

    # 3. Dream character cylinder walking in dream space
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=0.04, depth=0.16, location=(x_offset + dw * 0.5, -dh * 0.5, 0.08)
    )
    d_player = bpy.context.active_object
    d_player.name = "DreamPlayer"
    d_player.data.materials.append(g_mat)
    if d_player.name not in twin_coll.objects:
        twin_coll.objects.link(d_player)
    if d_player.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(d_player)

    return twin_coll


def build_camera_and_aim(camera_aim_track, camera_pos_track, player_obj, topdown=False, map_w=6.40, map_h=5.76):
    """Build camera and aim. If topdown=True, builds straight orthographic overhead camera."""
    cam_data = bpy.data.cameras.new("MainCamera")

    if topdown:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = max(map_w, map_h) * 1.02
        cam_data.dof.use_dof = False

        cam_obj = bpy.data.objects.new("MainCamera", cam_data)
        cam_obj.location = (map_w * 0.5, -map_h * 0.5, 15.0)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        return cam_obj, None

    aim_obj = bpy.data.objects.new("Aim", None)
    aim_obj.empty_display_type = "PLAIN_AXES"
    aim_obj.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(aim_obj)

    cam_data.type = "PERSP"
    cam_data.lens = 50.0
    cam_data.clip_end = 500.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = aim_obj
    cam_data.dof.aperture_fstop = 2.0

    cam_obj = bpy.data.objects.new("MainCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    track_to = cam_obj.constraints.new("TRACK_TO")
    track_to.target = aim_obj
    track_to.track_axis = "TRACK_NEGATIVE_Z"
    track_to.up_axis = "UP_Y"

    F = len(camera_aim_track)
    for f in range(F):
        ax, ay, az = camera_aim_track[f]
        aim_obj.location = (ax * S, -ay * S, az * S)
        aim_obj.keyframe_insert(data_path="location", frame=f)

        cx, cy, cz = camera_pos_track[f]
        cam_obj.location = (cx * S, -cy * S, cz * S)
        cam_obj.keyframe_insert(data_path="location", frame=f)

    return cam_obj, aim_obj


def build_cinematic_extras(map_w=6.40, map_h=5.76):
    """Build seamless outer background terrain on all four sides, rolling hills, distant trees, clouds."""
    extras_coll = bpy.data.collections.new("CinematicExtras")
    bpy.context.scene.collection.children.link(extras_coll)

    # Seamless background terrain matching the map's grass color
    h_mat = bpy.data.materials.new("HillsMat")
    h_mat.use_nodes = True
    h_bsdf = h_mat.node_tree.nodes["Principled BSDF"]
    # Exact linear grass color matching map shade 1: srgb (140, 210, 100)
    h_bsdf.inputs["Base Color"].default_value = (0.260, 0.644, 0.123, 1.0)
    h_bsdf.inputs["Roughness"].default_value = 0.85

    # Outer terrain grid surrounding the map on all four sides
    hill_verts = []
    hill_faces = []
    h_idx = 0

    N = 75
    xs = np.linspace(-150.0, 150.0, N)
    ys = np.linspace(-150.0, 150.0, N)

    for i in range(N - 1):
        for j in range(N - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]

            # If quad is inside or touching map bounds [0, map_w] x [-map_h, 0], skip it
            if not (x1 <= -0.02 or x0 >= map_w + 0.02 or y1 <= -map_h - 0.02 or y0 >= 0.02):
                continue

            def get_elev(x, y):
                dx = max(0.0, -x, x - map_w)
                dy = max(0.0, y, -map_h - y)
                d_map = math.sqrt(dx * dx + dy * dy)
                if d_map <= 15.0:
                    return -0.015  # Slightly below map base to seat cleanly on all four sides
                # Distant rolling hills rising smoothly in all directions
                t_rise = min(1.0, (d_map - 15.0) / 60.0)
                undulate = 0.8 * math.sin(x * 0.04) + 0.8 * math.cos(y * 0.04)
                h_elev = -0.015 + t_rise * (3.5 + undulate)
                return h_elev

            z00 = get_elev(x0, y0)
            z10 = get_elev(x1, y0)
            z11 = get_elev(x1, y1)
            z01 = get_elev(x0, y1)

            hill_verts.extend([
                (x0, y0, z00),
                (x1, y0, z10),
                (x1, y1, z11),
                (x0, y1, z01),
            ])
            hill_faces.append((h_idx, h_idx + 1, h_idx + 2, h_idx + 3))
            h_idx += 4

    h_me = bpy.data.meshes.new("RollingHillsMesh")
    h_me.from_pydata(hill_verts, [], hill_faces)
    h_me.update(calc_edges=True)
    h_me.materials.append(h_mat)

    hills_obj = bpy.data.objects.new("RollingHills", h_me)
    extras_coll.objects.link(hills_obj)

    # Distant tree silhouettes along the perimeter
    for i in range(48):
        ang = i * (2.0 * math.pi / 48)
        rad = 32.0 + (i % 5) * 8.0
        tx = (map_w * 0.5) + rad * math.cos(ang)
        ty = (-map_h * 0.5) + rad * math.sin(ang)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1, radius=1.6, location=(tx, ty, 2.5 + (i % 3) * 0.5)
        )
        d_tree = bpy.context.active_object
        d_tree.scale = (1.0, 1.0, 1.6)
        d_tree.data.materials.append(h_mat)
        if d_tree.name not in extras_coll.objects:
            extras_coll.objects.link(d_tree)
        if d_tree.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(d_tree)

    # Soft golden-hour cumulus clouds in distance
    cloud_mat = bpy.data.materials.new("CloudMat")
    cloud_mat.use_nodes = True
    c_bsdf = cloud_mat.node_tree.nodes["Principled BSDF"]
    c_bsdf.inputs["Base Color"].default_value = (1.0, 0.94, 0.85, 1.0)
    if "Emission Color" in c_bsdf.inputs:
        c_bsdf.inputs["Emission Color"].default_value = (1.0, 0.90, 0.76, 1.0)
        c_bsdf.inputs["Emission Strength"].default_value = 0.35

    for c_i, (cx, cy, cz) in enumerate([
        (-25.0, 65.0, 24.0),
        (10.0, 75.0, 28.0),
        (45.0, 60.0, 22.0),
        (-60.0, 50.0, 20.0),
    ]):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2, radius=12.0, location=(cx, cy, cz)
        )
        cloud = bpy.context.active_object
        cloud.name = f"Cloud_{c_i}"
        cloud.scale = (2.4, 1.0, 0.35)
        cloud.data.materials.append(cloud_mat)
        if cloud.name not in extras_coll.objects:
            extras_coll.objects.link(cloud)
        if cloud.name in bpy.context.scene.collection.objects:
            extras_coll.objects.unlink(cloud)


def setup_compositor():
    """Setup compositor with bloom/streaks glare for sun specular glints on water."""
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("CompositorNodeComposite")

    glare_bloom = tree.nodes.new("CompositorNodeGlare")
    glare_bloom.glare_type = "BLOOM"
    if "Threshold" in glare_bloom.inputs:
        glare_bloom.inputs["Threshold"].default_value = 1.1
    if "Size" in glare_bloom.inputs:
        glare_bloom.inputs["Size"].default_value = 8

    tree.links.new(rl.outputs["Image"], glare_bloom.inputs["Image"])
    tree.links.new(glare_bloom.outputs["Image"], out.inputs["Image"])


def configure_engine(engine_name="eevee", spp=64):
    """Configure render settings for OptiX Cycles or EEVEE Next."""
    scene = bpy.context.scene
    r = scene.render

    if engine_name.lower() in ("cycles", "cycles_gpu"):
        r.engine = "CYCLES"
        c = scene.cycles
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "OPTIX"
        p.refresh_devices()
        for d in p.devices:
            d.use = (d.type == "OPTIX")

        c.device = "GPU"
        c.samples = spp
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.05
        c.use_denoising = True
        c.denoiser = "OPTIX"
        c.denoising_use_gpu = True
        c.max_bounces = 4
        r.use_persistent_data = True

    elif engine_name.lower() in ("cycles_cpu", "cpu"):
        r.engine = "CYCLES"
        c = scene.cycles
        c.device = "CPU"
        c.samples = spp
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.05
        c.use_denoising = True
        c.max_bounces = 4
        r.threads_mode = "FIXED"
        r.threads = 16
        r.use_persistent_data = True

    else:
        # EEVEE Next
        r.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
            e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        ] else "BLENDER_EEVEE"

        eevee = scene.eevee
        eevee.taa_render_samples = spp
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = True
        if hasattr(eevee, "ray_tracing_method"):
            eevee.ray_tracing_method = "SCREEN"
        if hasattr(eevee, "use_shadows"):
            eevee.use_shadows = True
        if hasattr(eevee, "shadow_ray_count"):
            eevee.shadow_ray_count = 2
        if hasattr(eevee, "shadow_step_count"):
            eevee.shadow_step_count = 4

    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"

    scene.view_settings.exposure = 0.1


def check_gpu_temp_and_throttle():
    """Query nvidia-smi temperature and sleep 60s if >= 85 C."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader"],
            text=True,
        )
        parts = out.strip().split(",")
        temp = int(parts[0].strip())
        power = parts[1].strip() if len(parts) > 1 else "N/A"
        print(f"[GPU Status] Temp: {temp} C, Power: {power}")
        if temp >= 85:
            print(f"[GPU Guard] Temp >= 85 C ({temp} C). Sleeping 60 s...")
            time.sleep(60)
        return temp, power
    except Exception as e:
        print(f"[GPU Status] Could not query nvidia-smi: {e}")
        return 0, "N/A"


def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b -P bpy_build.py -- <scene.npz> <out_dir> [options]")
        return

    args_list = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_npz", help="Path to scene.npz")
    parser.add_argument("out_dir", help="Output directory for rendered frames")
    parser.add_argument("--engine", default="eevee", choices=["eevee", "cycles", "cycles_cpu", "cpu"], help="Render engine")
    parser.add_argument("--frame", type=int, default=None, help="Render single frame")
    parser.add_argument("--frames", type=int, nargs=2, default=None, help="Render range of frames A B")
    parser.add_argument("--width", type=int, default=1920, help="Resolution width")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel")
    parser.add_argument("--preview", action="store_true", help="Set 960x540, 16 spp")
    parser.add_argument("--topdown", action="store_true", help="Render straight top-down orthographic frame of full map")

    args = parser.parse_args(args_list)

    # 1. Reset factory settings
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Load scene data
    scene_path = os.path.abspath(args.scene_npz)
    print(f"Loading scene data from {scene_path}...")
    data = np.load(scene_path)
    cls_map = data["cls"]
    shade_map = data["shade"]
    player_track = data["player_track"]
    camera_aim = data["camera_track"]
    camera_pos = data["camera_pos"]
    player_sprites_cls = data["player_sprites_cls"]
    player_sprites_shade = data["player_sprites_shade"]
    distinct_steps = data["distinct_steps"]

    H_px, W_px = cls_map.shape
    map_w = W_px * S
    map_h = H_px * S

    # 3. Build scene geometry
    shot_name = str(data["shot_name"]) if "shot_name" in data else ""
    is_gym = (shot_name == "gym_closer") or ("map54" in scene_path)
    is_twin = (shot_name == "route2_twin") or ("dream_cls" in data)

    print(f"Building procedural diorama (shot: {shot_name}, gym: {is_gym}, twin: {is_twin})...")
    setup_world()

    if is_gym:
        create_gym_lights(map_w=map_w, map_h=map_h)
        ground_img = create_ground_texture(cls_map, shade_map, is_gym=True)
        ground_obj, ground_mat, tile_heights = build_ground_mesh(cls_map, shade_map, ground_img, is_gym=True)
        build_gym_boulders(cls_map)
        if "brock_cls" in data:
            build_brock_voxel(data["brock_cls"], data["brock_shade"])
        player_obj = build_voxel_player(player_track, player_sprites_cls, player_sprites_shade, distinct_steps)
        build_camera_and_aim(camera_aim, camera_pos, player_obj, topdown=args.topdown, map_w=map_w, map_h=map_h)
    elif is_twin:
        create_sun(topdown=args.topdown)
        ground_img = create_ground_texture(cls_map, shade_map)
        ground_obj, ground_mat, tile_heights = build_ground_mesh(cls_map, shade_map, ground_img)
        build_procedural_trees(cls_map)
        build_tall_grass(cls_map)
        build_fences(cls_map)
        build_flowers(cls_map)
        build_buildings_and_roofs(cls_map, ground_mat)
        player_obj = build_voxel_player(player_track, player_sprites_cls, player_sprites_shade, distinct_steps)
        build_dream_twin(data["dream_cls"], data["dream_shade"], data["reanchors"], data["steps"], map_w=map_w, map_h=map_h)
        build_camera_and_aim(camera_aim, camera_pos, player_obj, topdown=args.topdown, map_w=map_w + 2.0, map_h=map_h)
        build_cinematic_extras(map_w=map_w + 2.0, map_h=map_h)
    else:
        create_sun(topdown=args.topdown)
        ground_img = create_ground_texture(cls_map, shade_map)
        ground_obj, ground_mat, tile_heights = build_ground_mesh(cls_map, shade_map, ground_img)
        build_water_surface(cls_map)
        build_procedural_trees(cls_map)
        build_tall_grass(cls_map)
        build_fences(cls_map)
        build_flowers(cls_map)
        build_buildings_and_roofs(cls_map, ground_mat)
        player_obj = build_voxel_player(player_track, player_sprites_cls, player_sprites_shade, distinct_steps)
        build_camera_and_aim(camera_aim, camera_pos, player_obj, topdown=args.topdown, map_w=map_w, map_h=map_h)
        build_cinematic_extras(map_w=map_w, map_h=map_h)

    setup_compositor()

    # 4. Configure render engine & output
    if args.topdown:
        width = 1280
        height = 1152
        spp = 32
    else:
        width = 960 if args.preview else args.width
        height = 540 if args.preview else (args.width * 9 // 16)
        spp = 16 if args.preview else args.spp

    configure_engine(engine_name=args.engine, spp=spp)

    scene = bpy.context.scene
    r = scene.render
    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 5. Render target frame(s)
    if args.topdown:
        out_path = os.path.join(out_dir, "v3_topdown.png")
        r.filepath = out_path
        print(f"Rendering top-down orthographic view ({width}x{height}) -> {out_path}...")
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dur = time.time() - t0
        print(f"Top-down render completed in {dur:.2f} s -> {out_path}")
        return

    if args.frame is not None:
        frames_to_render = [args.frame]
    elif args.frames is not None:
        frames_to_render = list(range(args.frames[0], args.frames[1] + 1))
    else:
        frames_to_render = [0]

    print(f"Starting render: {len(frames_to_render)} frame(s) with {args.engine} ({width}x{height}, {spp} spp)...")

    frame_times = []
    for count, f in enumerate(frames_to_render):
        if count % 10 == 0:
            check_gpu_temp_and_throttle()

        scene.frame_set(f)
        out_path = os.path.join(out_dir, f"frame_{f:04d}.png")
        r.filepath = out_path

        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dur = time.time() - t0
        frame_times.append(dur)
        print(f"[Frame {f:04d}] Rendered in {dur:.2f} s -> {out_path}")

    avg_s = np.mean(frame_times) if frame_times else 0.0
    print(f"Render completed. Average: {avg_s:.2f} s/frame over {len(frames_to_render)} frames.")


if __name__ == "__main__":
    main()
