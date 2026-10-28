"""Everything that needs bpy / BlenderProc. Only imported inside `blenderproc run`.

World units: blender_units = mm * 0.001 * world_scale (Bullet is unstable at millimetre scale).
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import blenderproc as bproc  # noqa: F401  (must be importable: we run inside blenderproc)
import bpy
import numpy as np

from .camera import RenderCamera

log = logging.getLogger(__name__)
SLOTS = ("body", "insert")


def _input(node: bpy.types.Node, names: Sequence[str]):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def _mix_rgba(nt: bpy.types.NodeTree):
    """Mix node in colour mode: (factor, A, B, result) sockets picked by type, since Blender 4 has
    several sockets called A/B/Result (float, vector, colour) on the same node."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    a = next(s for s in mix.inputs if s.name == "A" and s.type == "RGBA")
    b = next(s for s in mix.inputs if s.name == "B" and s.type == "RGBA")
    fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
    out = next(s for s in mix.outputs if s.name == "Result" and s.type == "RGBA")
    return mix, fac, a, b, out


def _set(node: bpy.types.Node, names: Sequence[str], value: Any) -> None:
    sock = _input(node, names)
    if sock is not None:
        sock.default_value = value


# ------------------------------------------------------------------------------------ renderer


def init_renderer(cfg: Mapping[str, Any]) -> None:
    bproc.renderer.set_render_devices(desired_gpu_device_type=[cfg.get("device", "OPTIX"), "CUDA"])
    bproc.renderer.set_max_amount_of_samples(int(cfg.get("samples", 64)))
    bproc.renderer.set_noise_threshold(float(cfg.get("noise_threshold", 0.02)))
    den = cfg.get("denoiser", "OPTIX")
    bproc.renderer.set_denoiser(None if den in (None, "none", "NONE") else den)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=int(cfg.get("diffuse_bounces", 3)),
        glossy_bounces=int(cfg.get("glossy_bounces", 4)),
        max_bounces=int(cfg.get("max_bounces", 6)),
        transmission_bounces=int(cfg.get("transmission_bounces", 2)),
        transparent_max_bounces=4,
    )
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = (
        "Standard"  # no filmic tone curve: linear -> sRGB like a camera
    )
    scene.view_settings.look = "None"
    scene.render.film_transparent = False
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"], default_values={"category_id": 0}
    )
    if cfg.get("depth", False):
        bproc.renderer.enable_depth_output(activate_antialiasing=False)


def set_camera(cam: RenderCamera, world_scale: float) -> None:
    w, h = cam.resolution
    bproc.camera.set_intrinsics_from_K_matrix(cam.K, w, h, clip_start=0.01, clip_end=100.0)
    bproc.camera.add_camera_pose(cam.cam2world_blender(world_scale), frame=0)


def check_camera(cam: RenderCamera, world_scale: float, tol_px: float = 0.1) -> float:
    """Project plate points with Blender's camera and with K/T in numpy; they must agree."""
    x0, y0, x1, y1 = cam.visible_plate_rect(-5.0)
    pts = np.array(
        [
            [x, y, z]
            for x in np.linspace(x0, x1, 5)
            for y in np.linspace(y0, y1, 5)
            for z in (0.0, 8.0)
        ]
    )
    px_np = cam.project(pts)
    px_bl = bproc.camera.project_points(pts * 0.001 * world_scale, frame=0)
    err = float(np.abs(px_np - px_bl).max())
    if err > tol_px:
        raise RuntimeError(f"Blender camera disagrees with K/T by {err:.3f} px (> {tol_px})")
    log.info("camera check: Blender vs numpy projection max error %.4f px", err)
    return err


# --------------------------------------------------------------------------------- materials


def make_material(name: str, p: Mapping[str, Any]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    _set(bsdf, ["Base Color"], (*p["base_color"], 1.0))
    _set(bsdf, ["Metallic"], p["metallic"])
    _set(bsdf, ["Roughness"], p["roughness"])
    _set(bsdf, ["Anisotropic"], p.get("anisotropic", 0.0))
    _set(bsdf, ["Anisotropic Rotation"], p.get("anisotropic_rotation", 0.0))
    _set(bsdf, ["Coat Weight", "Clearcoat"], p.get("coat", 0.0))
    _set(bsdf, ["Subsurface Weight", "Subsurface"], p.get("subsurface", 0.0))
    if p.get("iridescence", 0.0) > 0:
        _set(bsdf, ["Thin Film Thickness"], 300.0 * p["iridescence"])
    coord = nt.nodes.new("ShaderNodeTexCoord")
    seed_offset = (p.get("noise_seed", 0) % 1000) * 0.137
    rough_in = _input(bsdf, ["Roughness"])
    if p.get("scratches") or p.get("fingerprints"):
        noise = nt.nodes.new("ShaderNodeTexNoise")
        _set(noise, ["Scale"], 400.0 if p.get("scratches") else 60.0)
        _set(noise, ["Detail"], 8.0)
        _set(noise, ["W"], seed_offset)
        noise.noise_dimensions = "4D"
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeMapRange")
        _set(ramp, ["From Min"], 0.45)
        _set(ramp, ["From Max"], 0.75)
        _set(ramp, ["To Min"], p["roughness"])
        _set(ramp, ["To Max"], min(1.0, p["roughness"] + (0.35 if p.get("fingerprints") else 0.2)))
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], rough_in)
    if p.get("grime"):
        gnoise = nt.nodes.new("ShaderNodeTexNoise")
        _set(gnoise, ["Scale"], 25.0)
        _set(gnoise, ["W"], seed_offset + 3.1)
        gnoise.noise_dimensions = "4D"
        nt.links.new(coord.outputs["Object"], gnoise.inputs["Vector"])
        mix, m_fac, m_a, m_b, m_out = _mix_rgba(nt)
        mix.blend_type = "MULTIPLY"
        m_a.default_value = (*p["base_color"], 1.0)
        m_b.default_value = (0.55, 0.5, 0.42, 1.0)
        fac = nt.nodes.new("ShaderNodeMapRange")
        _set(fac, ["From Min"], 0.5)
        _set(fac, ["From Max"], 0.8)
        nt.links.new(gnoise.outputs["Fac"], fac.inputs["Value"])
        nt.links.new(fac.outputs["Result"], m_fac)
        nt.links.new(m_out, _input(bsdf, ["Base Color"]))
    return mat


# ------------------------------------------------------------------------------------- assets


class AssetLibrary:
    """Loads each part once (hidden prototype) and hands out duplicates."""

    def __init__(self, assets_dir: Path, world_scale: float) -> None:
        self.dir = Path(assets_dir)
        self.s = 0.001 * world_scale
        self.protos: dict[str, tuple[Any, list[Any], dict]] = {}

    def _load(self, part_id: str, meta: dict) -> tuple[Any, list[Any], dict]:
        folder = self.dir / part_id
        obj = bproc.loader.load_obj(str(folder / "render.ply"), forward_axis="Y", up_axis="Z")[0]
        obj.set_scale([self.s] * 3)
        obj.persist_transformation_into_mesh()
        obj.set_shading_mode("AUTO", 35)
        mesh = obj.get_mesh()
        slots = meta["mesh"].get("material_slots", {"body": [0, len(mesh.polygons)]})
        # the PLY loader adds a default material; drop it (and the datablock) so slots follow SLOTS
        loader_mats = [m.name for m in mesh.materials if m is not None]
        mesh.materials.clear()
        for name in loader_mats:
            m = bpy.data.materials.get(name)
            if m is not None and m.users == 0:
                bpy.data.materials.remove(m)
        for _ in SLOTS[: max(1, len(slots))]:
            mesh.materials.append(None)
        if "insert" in slots:
            a, b = slots["insert"]
            idx = np.zeros(len(mesh.polygons), dtype=np.int32)
            idx[a:b] = 1
            mesh.polygons.foreach_set("material_index", idx)
        pieces = bproc.loader.load_obj(str(folder / "collision.obj"), forward_axis="Y", up_axis="Z")
        for p in pieces:
            p.set_scale([self.s] * 3)
            p.persist_transformation_into_mesh()
            p.set_parent(obj)
            p.hide(True)
        obj.hide(True)
        obj.set_location([0, 0, -1000])  # prototypes stay far below the scene
        return obj, pieces, meta

    def instance(self, part_id: str, meta: dict) -> tuple[Any, list[Any]]:
        if part_id not in self.protos:
            self.protos[part_id] = self._load(part_id, meta)
        proto = self.protos[part_id][0]
        # not linked: BlenderProc's physics moves each object's origin to its centre of mass, which
        # would corrupt mesh data shared between instances
        dup = proto.duplicate(duplicate_children=True, linked=False)
        dup.hide(False)
        children = dup.get_children()
        for c in children:
            c.hide(True)
        return dup, children


# ------------------------------------------------------------------------------- static scene


class StaticScene:
    """Plate (with the real empty-plate texture), bed, collider, dark shroud and the light panel."""

    def __init__(
        self, cam: RenderCamera, cfg: Mapping[str, Any], calib_dir: Path, world_scale: float
    ) -> None:
        self.cfg, self.cam, self.ws = cfg, cam, world_scale
        self.s = 0.001 * world_scale
        pc = cfg.get("plate", {}) or {}
        x0, y0, x1, y1 = cam.visible_plate_rect(30.0)
        self.rect = (x0, y0, x1, y1)
        self.plate = self._plate_mesh(x0, y0, x1, y1, int(pc.get("grid", 24)))
        self.texture = None
        tex_path = Path(calib_dir) / "plate_empty.png"
        if cam.source == "calibration" and tex_path.is_file():
            self.texture = bpy.data.images.load(str(tex_path))
        plate_mat = self._plate_material()
        self.plate_mat_name = plate_mat.name
        self.plate.blender_obj.data.materials.append(plate_mat)
        self.plate.set_cp("category_id", 0)
        th = float(pc.get("thickness_mm", 1.75))
        self.collider = bproc.object.create_primitive("CUBE")
        self.collider.set_scale([(x1 - x0) / 2 * self.s, (y1 - y0) / 2 * self.s, 5 * self.s])
        self.collider.set_location([(x0 + x1) / 2 * self.s, (y0 + y1) / 2 * self.s, -5 * self.s])
        self.collider.enable_rigidbody(
            False,
            collision_shape="BOX",
            collision_margin=float(cfg.get("physics", {}).get("collision_margin_mm", 0.05))
            * self.s,
            friction=float(cfg.get("physics", {}).get("friction", 0.6)),
        )
        self.collider.hide(True)
        self.collider.set_cp("category_id", 0)
        bed = bproc.object.create_primitive("CUBE")
        bed.set_scale([(x1 - x0) / 2 * self.s * 1.3, (y1 - y0) / 2 * self.s * 1.3, 1.5 * self.s])
        bed.set_location(
            [(x0 + x1) / 2 * self.s, (y0 + y1) / 2 * self.s, (-th - 1.5 - 0.01) * self.s]
        )
        bed_mat = bproc.material.create("bed")
        bed_mat.set_principled_shader_value("Base Color", [0.25, 0.25, 0.26, 1])
        bed_mat.set_principled_shader_value("Metallic", 0.8)
        bed_mat.set_principled_shader_value("Roughness", 0.5)
        bed.replace_materials(bed_mat)
        bed.set_cp("category_id", 0)
        sc = cfg.get("shroud", {}) or {}
        hw = float(sc.get("half_width_mm", 220.0))
        hh = float(sc.get("height_mm", 420.0))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.shroud = bproc.object.create_primitive("CUBE")
        self.shroud.set_scale([hw * self.s, hw * self.s, hh / 2 * self.s])
        self.shroud.set_location([cx * self.s, cy * self.s, (hh / 2 - 20) * self.s])
        smat = bproc.material.create("shroud")
        smat.set_principled_shader_value("Base Color", [*sc.get("colour", [0.02, 0.02, 0.02]), 1])
        smat.set_principled_shader_value("Roughness", 0.9)
        self.shroud.replace_materials(smat)
        self.shroud.set_cp("category_id", 0)
        self.light = bproc.types.Light("AREA", name="panel")
        self.light.blender_obj.data.shape = "RECTANGLE"
        self.world_strength = 0.0
        bproc.renderer.set_world_background([1.0, 1.0, 1.0], 0.0)

    def _plate_mesh(self, x0: float, y0: float, x1: float, y1: float, n: int):
        """Grid plane at z = 0 whose UVs map plate points to the undistorted ROI (plate_empty.png)."""
        xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
        X, Y = np.meshgrid(xs, ys)
        verts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
        faces = []
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                faces.append((a, a + 1, a + n + 2, a + n + 1))
        mesh = bpy.data.meshes.new("plate")
        mesh.from_pydata((verts * self.s).tolist(), [], faces)
        uv = mesh.uv_layers.new(name="UVMap")
        px = self.cam.project(verts)
        w, h = self.cam.resolution
        uvs = np.c_[(px[:, 0] + 0.5) / w, 1.0 - (px[:, 1] + 0.5) / h]
        loop_uv = np.array([uvs[v] for poly in mesh.polygons for v in poly.vertices])
        uv.data.foreach_set("uv", loop_uv.ravel())
        mesh.update()
        obj = bpy.data.objects.new("plate", mesh)
        bpy.context.collection.objects.link(obj)
        return bproc.types.MeshObject(obj)

    def _plate_material(self) -> bpy.types.Material:
        mat = bpy.data.materials.new("plate")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        bsdf.name = "plate_bsdf"
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.name = "plate_rgb"
        if self.texture is not None:
            img = nt.nodes.new("ShaderNodeTexImage")
            img.image = self.texture
            img.interpolation = "Cubic"
            mix, _, m_a, m_b, m_out = _mix_rgba(nt)
            mix.name = "plate_mix"
            nt.links.new(img.outputs["Color"], m_a)
            nt.links.new(rgb.outputs["Color"], m_b)
            nt.links.new(m_out, _input(bsdf, ["Base Color"]))
        else:
            nt.links.new(rgb.outputs["Color"], _input(bsdf, ["Base Color"]))
        # print lines: fine waves along x as a bump
        coord = nt.nodes.new("ShaderNodeTexCoord")
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X"
        pitch = float((self.cfg.get("plate", {}) or {}).get("line_pitch_mm", 0.4)) * self.s
        _set(wave, ["Scale"], 1.0 / max(pitch, 1e-6) / (2 * math.pi))
        nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.name = "plate_bump"
        _set(bump, ["Distance"], 0.02 * self.s)
        nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], _input(bsdf, ["Normal"]))
        return mat

    def jitter(self, rng: random.Random) -> dict[str, Any]:
        """Randomise lights and plate for one render; returns the parameters used."""
        pc = self.cfg.get("plate", {}) or {}
        lc = self.cfg.get("lights", {}) or {}
        rough = rng.uniform(*pc.get("roughness", [0.6, 0.9]))
        bump = rng.uniform(*pc.get("bump_strength", [0.0, 0.15]))
        nodes = bpy.data.materials[self.plate_mat_name].node_tree.nodes
        _set(nodes["plate_bsdf"], ["Roughness"], rough)
        _set(nodes["plate_bump"], ["Strength"], bump)
        solid = self.texture is None or rng.random() < float(
            pc.get("solid_colour_probability", 0.1)
        )
        colours = pc.get("solid_colours", [[0.12, 0.35, 0.65]])
        base = list(rng.choice(colours)) if solid else [0.5, 0.5, 0.5]
        if solid:
            base = [min(1.0, max(0.0, c * rng.uniform(0.85, 1.15))) for c in base]
        nodes["plate_rgb"].outputs["Color"].default_value = (*base, 1.0)
        mix = nodes.get("plate_mix")
        if mix is not None:  # factor of the texture/solid colour mix
            fac = next(s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE")
            fac.default_value = 1.0 if solid else 0.0
        size = rng.uniform(*lc.get("panel_size_mm", [250, 400]))
        height = rng.uniform(*lc.get("height_mm", [320, 380]))
        off = lc.get("offset_mm", [0, 40])
        r, a = rng.uniform(*off), rng.uniform(0, 2 * math.pi)
        x0, y0, x1, y1 = self.rect
        cx, cy = (x0 + x1) / 2 + r * math.cos(a), (y0 + y1) / 2 + r * math.sin(a)
        energy = rng.uniform(*lc.get("energy_w", [40, 120]))
        temp = rng.uniform(*lc.get("colour_temperature_k", [5000, 6500]))
        col = _kelvin_rgb(temp)
        self.light.set_location([cx * self.s, cy * self.s, height * self.s])
        self.light.set_rotation_euler([0, 0, rng.uniform(0, math.pi)])
        self.light.blender_obj.data.size = size * self.s
        self.light.blender_obj.data.size_y = size * self.s * rng.uniform(0.6, 1.0)
        self.light.set_energy(energy)
        self.light.set_color(col)
        world = rng.uniform(*lc.get("fill_world_strength", [0.0, 0.05]))
        bproc.renderer.set_world_background([1.0, 1.0, 1.0], world)
        exposure = rng.uniform(*self.cfg.get("render", {}).get("exposure", [0.0, 0.0]))
        bpy.context.scene.view_settings.exposure = exposure
        return {
            "plate_roughness": rough,
            "plate_bump": bump,
            "plate_solid_colour": base if solid else None,
            "light_size_mm": size,
            "light_height_mm": height,
            "light_xy_mm": [cx, cy],
            "light_energy_w": energy,
            "light_temperature_k": temp,
            "world_strength": world,
            "exposure_ev": exposure,
        }


def _kelvin_rgb(t: float) -> list[float]:
    """Approximate RGB of a black body (Tanner Helland), normalised to max 1."""
    t /= 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    b = (
        255.0
        if t >= 66
        else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    )
    rgb = [max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b)]
    m = max(rgb)
    return [c / m for c in rgb]


# ------------------------------------------------------------------------------ per instance


def place_instance(
    obj, children, T_plate_mm: np.ndarray, s: float, mass_kg: float, physics: Mapping[str, Any]
) -> None:
    T = np.array(T_plate_mm, dtype=float)
    T[:3, 3] *= s
    obj.set_local2world_mat(T)
    margin = float(physics.get("collision_margin_mm", 0.05)) * s
    obj.enable_rigidbody(
        True,
        collision_shape="COMPOUND",
        collision_margin=margin,
        mass=mass_kg,
        friction=float(physics.get("friction", 0.6)),
        angular_damping=float(physics.get("angular_damping", 0.3)),
        linear_damping=float(physics.get("linear_damping", 0.1)),
    )
    for c in children:
        c.enable_rigidbody(
            True,
            collision_shape="CONVEX_HULL",
            collision_margin=margin,
            friction=float(physics.get("friction", 0.6)),
        )


def assign_materials(obj, materials: Mapping[str, bpy.types.Material]) -> None:
    # on the mesh (each instance has its own, non-linked), not on the object: BlenderProc's
    # segmentation pass swaps obj.data.materials, and an object-linked material would override it
    mesh_mats = obj.blender_obj.data.materials
    for i in range(len(mesh_mats)):
        name = SLOTS[i] if i < len(SLOTS) else "body"
        mesh_mats[i] = materials.get(name, materials["body"])


def simulate(physics: Mapping[str, Any]) -> None:
    world = bpy.context.scene.rigidbody_world
    if world is not None:
        world.enabled = True  # BlenderProc disables it after every simulation
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=float(physics.get("min_time", 0.4)),
        max_simulation_time=float(physics.get("max_time", 1.5)),
        check_object_interval=float(physics.get("check_interval", 0.2)),
        substeps_per_frame=int(physics.get("substeps", 20)),
        solver_iters=int(physics.get("solver_iters", 20)),
    )


def final_pose_mm(obj, s: float) -> np.ndarray:
    T = np.array(obj.get_local2world_mat(), dtype=float)
    T[:3, 3] /= s
    return T


def datablock_counts() -> dict[str, int]:
    """Objects, meshes and materials alive in Blender; constant from scene to scene if nothing leaks."""
    d = bpy.data
    return {"objects": len(d.objects), "meshes": len(d.meshes), "materials": len(d.materials)}


def remove(objs: Sequence[Any], material_names: Sequence[str]) -> None:
    # by name: BlenderProc may already have freed the materials, and touching a freed Python
    # reference raises (or crashes Blender)
    # BlenderProc's delete() leaves the mesh data behind; with non-linked duplicates that leaks
    # every scene's meshes until Blender runs out of memory, so free them explicitly
    every = [e.blender_obj for o in objs for e in (o, *o.get_children())]
    mesh_names = {b.data.name for b in every if b.type == "MESH" and b.data is not None}
    for b in every:
        bpy.data.objects.remove(b, do_unlink=True)
    for name in mesh_names:
        m = bpy.data.meshes.get(name)
        if m is not None and m.users == 0:
            bpy.data.meshes.remove(m)
    for name in material_names:
        m = bpy.data.materials.get(name)
        if m is not None:
            bpy.data.materials.remove(m)


def render(depth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = bproc.renderer.render()
    rgb = np.asarray(data["colors"][0])[..., :3].astype(np.uint8)
    seg = np.asarray(data["category_id_segmaps"][0]).astype(np.uint16)
    dep = np.asarray(data["depth"][0], dtype=np.float32) if depth and "depth" in data else None
    return rgb, seg, dep
