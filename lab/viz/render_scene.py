"""Blender (4.5, Cycles on the CPU) render of voxel boxes from build_boxes.py as a tilt-shift diorama.
Run: blender -b --factory-startup -P lab/viz/render_scene.py -- <boxes.npz> <out.png> <width> <samples> <fstop>
"""
import math
import os
import sys

import bpy
import numpy as np

npz, out, width, samples, fstop = sys.argv[sys.argv.index("--") + 1:]
width, samples, fstop = int(width), int(samples), float(fstop)
S = 0.01                                             # 1 Game Boy pixel = 1 cm, so the camera behaves like a miniature

d = np.load(npz)
b, rgb = d["boxes"] * S, d["rgb"]
n = len(b)
x0, y0, z0, x1, y1, z1 = b.T
corners = np.stack([np.stack(v, 1) for v in (
    (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))], 1)
quad = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
faces = (quad[None] + 8 * np.arange(n)[:, None, None]).reshape(-1)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
me = bpy.data.meshes.new("diorama")
me.vertices.add(8 * n)
me.vertices.foreach_set("co", corners.reshape(-1).astype(np.float32))
me.loops.add(len(faces))
me.loops.foreach_set("vertex_index", faces.astype(np.int32))
me.polygons.add(6 * n)
me.polygons.foreach_set("loop_start", (4 * np.arange(6 * n)).astype(np.int32))
me.update(calc_edges=True)
lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
rgba = np.concatenate([lin, np.ones((n, 1))], 1).astype(np.float32)
col = me.color_attributes.new("col", "FLOAT_COLOR", "CORNER")
col.data.foreach_set("color", np.repeat(rgba, 24, 0).reshape(-1))
obj = bpy.data.objects.new("diorama", me)
scene.collection.objects.link(obj)

mat = bpy.data.materials.new("vox")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
vc = nt.nodes.new("ShaderNodeVertexColor")
vc.layer_name = "col"
nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.72
me.materials.append(mat)

focus = bpy.data.objects.new("focus", None)
focus.location = (0.73, 0.70, 0.06)                  # the player
scene.collection.objects.link(focus)
aim = bpy.data.objects.new("aim", None)
aim.location = (0.80, 0.64, 0.0)
scene.collection.objects.link(aim)
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = 40
cam_d.dof.use_dof = True
cam_d.dof.focus_object = focus
cam_d.dof.aperture_fstop = fstop
cam = bpy.data.objects.new("cam", cam_d)
el, dist = math.radians(46), 3.15
cam.location = (0.80, 0.64 - dist * math.cos(el), dist * math.sin(el))
bpy.ops.mesh.primitive_plane_add(size=20, location=(0.8, 0.72, -0.1))   # floor that only catches shadows
bpy.context.active_object.is_shadow_catcher = True
tc = cam.constraints.new("TRACK_TO")
tc.target, tc.track_axis, tc.up_axis = aim, "TRACK_NEGATIVE_Z", "UP_Y"
scene.collection.objects.link(cam)
scene.camera = cam

sun_d = bpy.data.lights.new("sun", "SUN")
sun_d.energy, sun_d.angle, sun_d.color = 4.2, math.radians(4), (1.0, 0.9, 0.78)
sun = bpy.data.objects.new("sun", sun_d)
sun.rotation_euler = (math.radians(52), 0, math.radians(-38))   # from the south-west, ~38 deg above the horizon
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.42, 0.58, 0.85, 1)
bg.inputs["Strength"].default_value = 0.55
scene.world = world

r = scene.render
r.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = 6
r.threads_mode, r.threads = "FIXED", 12
r.resolution_x, r.resolution_y, r.resolution_percentage = width, width * 9 // 16, 100
r.film_transparent = True
r.image_settings.file_format, r.image_settings.color_mode = "PNG", "RGBA"
r.filepath = os.path.abspath(out)          # a relative path resolves against the drive root
bpy.ops.render.render(write_still=True)
print("rendered", out)
