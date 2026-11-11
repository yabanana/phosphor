#!/usr/bin/env python3
"""F13 independent CPU reference for the texture-free reflection fixtures.

Written by the verifier, not by the renderer's author, from the material and
transport contracts only. It consumes exportOfflineReference's same-frame
snapshot (geometry, world matrices, camera, lights, emission, material
factors) and never reads a renderer image to choose rays, samples or
convergence. Mitsuba 3 (llvm_ad_rgb) supplies ray casting and emitter
sampling; the BRDF and the path integrator below are this file's own.

Declared reference model (every item is part of the report):
- Material: Lambert (1-F)(1-metallic)albedo/pi plus GGX with height-correlated
  Smith visibility and Schlick Fresnel, f0 = mix(0.04, albedo, metallic),
  roughness clamped to [0.04, 1], alpha = max(roughness^2, 0.002).
- Camera: one ray through each pixel centre, no reconstruction filter.
- Transport: full material at the camera vertex with both lobes continued;
  after the specular lobe the first hit is lit with the full material and
  continues with its diffuse lobe; every deeper vertex uses the F12 diffuse
  reduction albedo(1-metallic)/pi. Emitters are counted once (light and BSDF
  sampling combined with the power heuristic).
- Excluded by construction: light reaching a diffuse receiver through a
  specular bounce (caustics) and specular inter-reflection beyond one bounce.
- Secondary specular light connections use alpha >= the camera vertex alpha
  (path regularisation) unless --secondary-specular raw|off is selected.
- Back faces absorb. The environment is the snapshot's single sky radiance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import read_pfm, scene_digest, snapshot, write_pfm  # noqa: E402

INVALID = 0xFFFFFFFF
MIN_ROUGHNESS, MIN_ALPHA, DIELECTRIC_F0, MIN_NDOTV = 0.04, 0.002, 0.04, 1e-4
SEEDS = (1234, 5678)
CONVERGENCE = 0.02
MODEL = {
    "material": "Lambert(1-F)(1-metallic) + GGX height-correlated Smith + Schlick; roughness>=0.04, alpha>=0.002",
    "camera": "pixel-centre rays, no reconstruction filter",
    "transport": "full material at camera vertex and at the first hit after its specular lobe; diffuse reduction deeper",
    "excluded": ["caustics through a specular bounce", "specular inter-reflection beyond one bounce"],
    "back_faces": "absorbing",
    "environment": "snapshot sky radiance only",
}


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_texture_free(data):
    """The adapter has no texture filter: refuse anything it cannot reproduce exactly."""
    for instance in data["instances"]:
        material = data["materials"][instance["material"]]
        if any(index != INVALID for index in material["textures"]):
            raise ValueError("instanced material samples a texture; this reference is texture-free only")
        if material["alpha_cutoff"] > 0 or material["flags"] & 1:
            raise ValueError("alpha-masked or double-sided material is outside the declared model")
    if any(light["type"] != 0 for light in data["lights"]):
        raise ValueError("only directional analytic lights are modelled")
    for light in data["sampled_lights"]:
        if light["type"] != 6 or not 0 <= int(light.get("material", INVALID)) < len(data["materials"]):
            raise ValueError("only emissive mesh triangles are modelled as sampled lights")


class Context:
    """Scene, per-shape material tables and camera for one snapshot."""

    def __init__(self, root, data, threads, secondary_specular, max_bounces=16, rr_depth=5, regularization_scale=1.0):
        import drjit as dr
        import mitsuba as mi
        import numpy as np

        mi.set_variant("llvm_ad_rgb")
        dr.set_flag(dr.JitFlag.FastMath, False)
        dr.set_thread_count(threads)
        self.mi, self.dr, self.np = mi, dr, np
        self.data, self.secondary_specular = data, secondary_specular
        # Declared path regularisation: the light connection after the camera's
        # specular lobe uses alpha' = max(alpha_secondary, k * alpha_camera).
        self.regularization_scale = float(regularization_scale)
        if not (self.regularization_scale >= 1.0):
            raise ValueError("regularization scale must be >= 1")
        self.max_bounces, self.rr_depth = max_bounces, rr_depth
        require_texture_free(data)
        transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
        description = {"type": "scene"}
        names = {}
        for instance in data["instances"]:
            material = data["materials"][instance["material"]]
            shape = {"type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]),
                     "to_world": transform(np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F"))}
            if any(value > 0 for value in material["emissive"]):
                shape["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": list(material["emissive"])}}
            name = f"instance_{instance['slot']}"
            description[name] = shape
            names[name] = material
        for index, light in enumerate(data["lights"]):
            irradiance = np.asarray(light["color"], dtype=np.float64) * light["intensity"]
            direction = np.asarray(light["direction"], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            radius = float(data.get("sun_angular_radius", 0.0))
            if radius <= 0:
                description[f"sun_{index}"] = {"type": "directional", "direction": direction.tolist(),
                                               "irradiance": {"type": "rgb", "value": irradiance.tolist()}}
                continue
            helper = np.asarray([0.0, 0.0, 1.0] if abs(direction[2]) < 0.99 else [0.0, 1.0, 0.0])
            u = np.cross(helper, direction)
            u /= np.linalg.norm(u)
            v = np.cross(direction, u)
            distance = 1e6
            matrix = np.eye(4)
            matrix[:3, 0], matrix[:3, 1] = u * distance * math.tan(radius), v * distance * math.tan(radius)
            matrix[:3, 2], matrix[:3, 3] = direction, -direction * distance
            radiance = irradiance / (math.pi * math.sin(radius) ** 2)
            description[f"sun_{index}"] = {"type": "disk", "to_world": transform(matrix),
                                           "emitter": {"type": "area", "radiance": {"type": "rgb", "value": radiance.tolist()}}}
        if any(value > 0 for value in data["sky"]):
            description["sky"] = {"type": "constant", "radiance": {"type": "rgb", "value": list(data["sky"])}}
        self.scene = mi.load_dict(description)
        shapes = self.scene.shapes()
        identifiers = np.asarray(dr.reinterpret_array(mi.UInt32, self.scene.shapes_dr()))
        table = np.zeros((int(identifiers.max()) + 1, 5), dtype=np.float32)  # r,g,b,metallic,alpha
        for shape, identifier in zip(shapes, identifiers):
            material = names.get(shape.id())
            if material is None:
                continue  # the sun disk: black, absorbing, emits through its area emitter
            roughness = min(1.0, max(MIN_ROUGHNESS, float(material["roughness"])))
            table[identifier] = [*material["base"][:3], min(1.0, max(0.0, float(material["metallic"]))),
                                 max(roughness * roughness, MIN_ALPHA)]
        self.table = [mi.Float(table[:, column].copy()) for column in range(5)]
        camera = data["camera"]
        if any(camera.get("jitter_pixels", [0, 0])):
            raise ValueError("pixel-centre reference requires an unjittered camera")
        self.width, self.height = int(camera["width"]), int(camera["height"])
        forward = np.asarray(camera["direction"], dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(camera["up"], dtype=np.float64))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.origin = [float(v) for v in camera["position"]]
        self.basis = (forward, right, up)
        self.scale = self.height / (2 * math.tan(camera["fov_y_radians"] / 2))
        self.window = (0, 0, self.width, self.height)  # x0, y0, x1, y1 rendered pixels; zero elsewhere
        self.sampler = mi.load_dict({"type": "independent"})
        self.vector, self.colour = mi.Vector3d, getattr(mi, "Color3d", mi.Vector3d)

    # -- camera ---------------------------------------------------------
    def primary_ray(self, pixel):
        mi, dr = self.mi, self.dr
        forward, right, up = (mi.Vector3f(*axis.tolist()) for axis in self.basis)
        x = mi.Float(pixel % self.width) + 0.5
        y = mi.Float(pixel // self.width) + 0.5
        direction = forward + right * ((x - self.width / 2) / self.scale) + up * ((self.height / 2 - y) / self.scale)
        return mi.Ray3f(mi.Point3f(*self.origin), dr.normalize(direction))

    # -- material -------------------------------------------------------
    def material(self, si):
        mi, dr = self.mi, self.dr
        identifier = dr.reinterpret_array(mi.UInt32, si.shape)
        values = [dr.gather(mi.Float, column, identifier, si.is_valid()) for column in self.table]
        return {"albedo": self.colour(values[0], values[1], values[2]), "metallic": mi.Float64(values[3]),
                "alpha": mi.Float64(values[4])}

    def lobes(self, material, wo, wi, alpha):
        """Engine BRDF times cos(theta_i) for local unit vectors; returns diffuse, specular, VNDF pdf."""
        mi, dr = self.mi, self.dr
        nl = dr.clip(wi.z, 0.0, 1.0)
        nv = dr.maximum(wo.z, MIN_NDOTV)
        half = wo + wi
        length = dr.norm(half)
        half = half / dr.maximum(length, 1e-30)
        sine2, cosine2 = half.x * half.x + half.y * half.y, half.z * half.z
        a2 = alpha * alpha
        denominator = a2 * cosine2 + sine2  # nh^2 (a2-1) + 1 without cancellation
        distribution = a2 / (math.pi * denominator * denominator)
        gv = nl * dr.sqrt(nv * nv * (1 - a2) + a2)
        gl = nv * dr.sqrt(nl * nl * (1 - a2) + a2)
        visibility = 0.5 / dr.maximum(gv + gl, 1e-5)
        vh = dr.clip(dr.dot(wo, half), 0.0, 1.0)
        f0 = material["albedo"] * material["metallic"] + DIELECTRIC_F0 * (1 - material["metallic"])
        fresnel = f0 + (1 - f0) * (1 - vh) ** 5
        valid = (wi.z > 0) & (length > 1e-10)
        diffuse = dr.select(valid, (1 - fresnel) * (1 - material["metallic"]) * material["albedo"] * (nl / math.pi), 0)
        specular = dr.select(valid, fresnel * (distribution * visibility * nl), 0)
        g1 = 2 * nv / (nv + dr.sqrt(a2 + (1 - a2) * nv * nv))
        pdf = dr.select(valid, g1 * distribution / (4 * nv), 0)
        return diffuse, specular, pdf

    def sample_specular(self, wo, alpha, u):
        """Visible-normal GGX sampling (Heitz 2018); returns the local incident direction."""
        mi, dr = self.mi, self.dr
        stretched = dr.normalize(self.vector(alpha * wo.x, alpha * wo.y, wo.z))
        lensq = stretched.x * stretched.x + stretched.y * stretched.y
        inverse = dr.rsqrt(dr.maximum(lensq, 1e-30))
        t1 = dr.select(lensq > 1e-30, self.vector(-stretched.y * inverse, stretched.x * inverse, 0), self.vector(1, 0, 0))
        t2 = dr.cross(stretched, t1)
        radius, phi = dr.sqrt(mi.Float64(u.x)), 2 * math.pi * mi.Float64(u.y)
        p1, p2 = radius * dr.cos(phi), radius * dr.sin(phi)
        blend = 0.5 * (1 + stretched.z)
        p2 = (1 - blend) * dr.sqrt(dr.maximum(0, 1 - p1 * p1)) + blend * p2
        normal = t1 * p1 + t2 * p2 + stretched * dr.sqrt(dr.maximum(0, 1 - p1 * p1 - p2 * p2))
        half = dr.normalize(self.vector(alpha * normal.x, alpha * normal.y, dr.maximum(0, normal.z)))
        return half * (2 * dr.dot(wo, half)) - wo

    @staticmethod
    def mis(dr, first, second):
        first2, second2 = first * first, second * second
        return dr.select(first2 + second2 > 0, first2 / (first2 + second2), 0)

    # -- transport ------------------------------------------------------
    def tail(self, previous, direction, beta, pdf_previous, active, full_first, alpha_floor):
        """Follow one sampled direction from `previous`; diffuse continuation only."""
        mi, dr, scene, sampler = self.mi, self.dr, self.scene, self.sampler
        radiance = mi.Color3f(0)
        ray = previous.spawn_ray(direction)
        for depth in range(self.max_bounces):
            si = scene.ray_intersect(ray, active)
            emitter = si.emitter(scene, active)
            sample = mi.DirectionSample3f(scene, si, previous)
            pdf_light = scene.pdf_emitter_direction(previous, sample, active)
            weight = self.mis(dr, mi.Float64(pdf_previous), mi.Float64(pdf_light))
            radiance += dr.select(active, beta * emitter.eval(si, active) * mi.Float(weight), 0)
            outgoing = -ray.d
            active &= si.is_valid() & (dr.dot(si.n, outgoing) > 0)
            material = self.material(si)
            wo = self.vector(si.to_local(outgoing))
            reduced = material["albedo"] * (1 - material["metallic"])
            first = full_first and depth == 0
            # Light sampling at this vertex.
            light, emitted = scene.sample_emitter_direction(si, sampler.next_2d(), True, active)
            wi = self.vector(si.to_local(light.d))
            lit = active & (light.pdf > 0) & (wi.z > 0) & (dr.dot(si.n, light.d) > 0)
            cosine_pdf = dr.maximum(wi.z, 0) / math.pi
            light_weight = dr.select(light.delta, 1.0, self.mis(dr, mi.Float64(light.pdf), cosine_pdf))
            if first:
                alpha = material["alpha"]
                if self.secondary_specular == "regularized":
                    alpha = dr.maximum(alpha, alpha_floor)
                diffuse, specular, _ = self.lobes(material, wo, wi, alpha)
                value = diffuse * light_weight
                if self.secondary_specular != "off":
                    value = value + specular  # light sampling is the only strategy for this lobe
            else:
                value = reduced * (dr.maximum(wi.z, 0) / math.pi) * light_weight
            radiance += dr.select(lit, beta * emitted * mi.Color3f(value), 0)
            # Diffuse continuation.
            local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
            pdf_previous = local.z / math.pi
            if first:
                diffuse, _, _ = self.lobes(material, wo, self.vector(local), material["alpha"])
                throughput = diffuse / dr.maximum(mi.Float64(pdf_previous), 1e-30)
            else:
                throughput = reduced
            beta = beta * mi.Color3f(throughput)
            direction = si.to_world(local)
            active &= (local.z > 0) & (dr.dot(si.n, direction) > 0) & (dr.max(beta) > 0)
            if depth + 2 >= self.rr_depth:
                survival = dr.minimum(dr.max(beta), 0.95)
                active &= sampler.next_1d() < survival
                beta = dr.select(active, beta / dr.maximum(survival, 1e-30), 0)
            previous, ray = si, si.spawn_ray(direction)
            dr.eval(radiance, beta, active, pdf_previous, ray, previous)
            if not dr.any(active):
                break
        return radiance

    def render_pass(self, seed, spp):
        """Return the per-pixel SUM of `spp` samples as float64 (height, width, 3) and primary hits."""
        mi, dr, np, scene, sampler = self.mi, self.dr, self.np, self.scene, self.sampler
        x0, y0, x1, y1 = self.window
        columns = x1 - x0
        pixels = columns * (y1 - y0)
        sampler.seed(seed, pixels * spp)
        local = dr.repeat(dr.arange(mi.UInt32, pixels), spp)
        pixel = local  # accumulation index inside the window
        ray = self.primary_ray((local // columns + y0) * self.width + local % columns + x0)
        si = scene.ray_intersect(ray)
        radiance = si.emitter(scene).eval(si)  # visible emitter, or the environment on a miss
        outgoing = -ray.d
        active = si.is_valid() & (dr.dot(si.n, outgoing) > 0)
        material = self.material(si)
        alpha = material["alpha"]
        wo = self.vector(si.to_local(outgoing))
        # Light sampling at the camera vertex, both lobes, each with its own partner strategy.
        light, emitted = scene.sample_emitter_direction(si, sampler.next_2d(), True, active)
        wi = self.vector(si.to_local(light.d))
        lit = active & (light.pdf > 0) & (wi.z > 0) & (dr.dot(si.n, light.d) > 0)
        diffuse, specular, specular_pdf = self.lobes(material, wo, wi, alpha)
        cosine_pdf = dr.maximum(wi.z, 0) / math.pi
        light_pdf = mi.Float64(light.pdf)
        value = (diffuse * dr.select(light.delta, 1.0, self.mis(dr, light_pdf, cosine_pdf)) +
                 specular * dr.select(light.delta, 1.0, self.mis(dr, light_pdf, specular_pdf)))
        radiance += dr.select(lit, emitted * mi.Color3f(value), 0)
        dr.eval(radiance)
        # Diffuse lobe.
        local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        pdf = local.z / math.pi
        diffuse, _, _ = self.lobes(material, wo, self.vector(local), alpha)
        beta = mi.Color3f(diffuse / dr.maximum(mi.Float64(pdf), 1e-30))
        direction = si.to_world(local)
        go = active & (local.z > 0) & (dr.dot(si.n, direction) > 0) & (dr.max(beta) > 0)
        radiance += self.tail(si, direction, beta, pdf, go, False, alpha)
        # Specular lobe.
        local = self.sample_specular(wo, alpha, sampler.next_2d())
        _, specular, specular_pdf = self.lobes(material, wo, local, alpha)
        beta = mi.Color3f(specular / dr.maximum(specular_pdf, 1e-30))
        direction = si.to_world(mi.Vector3f(local))
        go = active & (local.z > 0) & (specular_pdf > 0) & (dr.dot(si.n, direction) > 0) & (dr.max(beta) > 0)
        radiance += self.tail(si, direction, beta, mi.Float(specular_pdf), go, True,
                              dr.minimum(alpha * self.regularization_scale, 1.0))
        image = np.zeros((pixels, 3), dtype=np.float64)
        for channel, values in enumerate((radiance.x, radiance.y, radiance.z)):
            target = dr.zeros(mi.Float64, pixels)
            dr.scatter_add(target, mi.Float64(values), pixel)
            image[:, channel] = np.asarray(target)
        if not np.isfinite(image).all():
            raise ValueError("nonfinite reference radiance")
        full = np.zeros((self.height, self.width, 3), dtype=np.float64)
        full[y0:y1, x0:x1] = image.reshape(y1 - y0, columns, 3)
        return full

    def offset_hit(self, x, y, dx, dy):
        mi, dr, np = self.mi, self.dr, self.np
        forward, right, up = (np.asarray(axis) for axis in self.basis)
        direction = forward + right * ((x + 0.5 + dx - self.width / 2) / self.scale) + up * ((self.height / 2 - y - 0.5 - dy) / self.scale)
        direction /= np.linalg.norm(direction)
        si = self.scene.ray_intersect(mi.Ray3f(mi.Point3f(*self.origin), mi.Vector3f(*direction.tolist())))
        hit = bool(np.asarray(si.is_valid())[0])
        return hit, np.asarray([np.asarray(si.p.x)[0], np.asarray(si.p.y)[0], np.asarray(si.p.z)[0]], dtype=np.float64)

    def primary_positions(self):
        mi, dr, np = self.mi, self.dr, self.np
        si = self.scene.ray_intersect(self.primary_ray(dr.arange(mi.UInt32, self.width * self.height)))
        valid = np.asarray(si.is_valid())
        position = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], axis=1)
        position[~valid] = 0
        return position.reshape(self.height, self.width, 3), valid.reshape(self.height, self.width)


def seed_agreement(np, first, second):
    """Stricter of the two seed-relative errors, with the signal support (as bench/f13_budget/reference.py)."""
    support = min(float(np.sqrt(np.mean(image ** 2))) for image in (first, second))
    return support, float(np.sqrt(np.mean((first - second) ** 2))) / max(support, 1e-6)


def region_slices(regions, width, height, window=None):
    x0, y0, x1, y1 = window or (0, 0, width, height)
    result = {"frame": (slice(y0, y1), slice(x0, x1))}
    for name, (x0, y0, x1, y1) in (regions or {}).items():
        result[name] = (slice(math.floor(y0 * height), math.ceil(y1 * height)),
                        slice(math.floor(x0 * width), math.ceil(x1 * width)))
    return result


def converge(context, checkpoints, spp_per_pass, regions, log=print):
    """Accumulate both seeds to each checkpoint; stop at the first one where every region agrees."""
    np = context.np
    sums = {seed: np.zeros((context.height, context.width, 3)) for seed in SEEDS}
    done, records = 0, []
    slices = region_slices(regions, context.width, context.height, context.window)
    for target in checkpoints:
        start = time.monotonic()
        while done < target:
            step = min(spp_per_pass, target - done)
            for seed in SEEDS:
                # Distinct stream per seed and pass; 1000003 keeps the two families apart.
                sums[seed] += context.render_pass((seed * 1000003 + done) & 0x7FFFFFFF, step)
            done += step
        images = [sums[seed] / done for seed in SEEDS]
        agreement = {name: dict(zip(("support", "seed_nrmse"), seed_agreement(np, images[0][area], images[1][area])))
                     for name, area in slices.items()}
        converged = all(v["support"] > 1e-6 and v["seed_nrmse"] <= CONVERGENCE for v in agreement.values())
        records.append({"spp_per_seed": done, "regions": agreement, "converged": converged,
                        "seconds": round(time.monotonic() - start, 2)})
        log(json.dumps(records[-1]))
        if converged:
            break
    return np.mean(images, axis=0), images, records


def validate_positions(np, context, path, tolerance=1e-3):
    """Compare this file's pixel-centre primary hits with the renderer's exported receiver positions."""
    engine = read_pfm(path).astype(np.float64)
    reference, valid = context.primary_positions()
    if engine.shape != reference.shape:
        raise ValueError("position capture extent differs from the snapshot camera")
    engine_valid = np.abs(engine).sum(axis=2) > 0
    both = valid & engine_valid
    error = np.abs(engine - reference).max(axis=2)
    # A pixel centre within rounding of a silhouette edge is covered by one rule and not
    # the other. Admit a mismatch only if a ray 0.01 pixel away reproduces the engine's
    # coverage at the engine's position; count those pixels and cap them.
    silhouette, unexplained = [], []
    # Coverage differs, or both hit but on different surfaces across a depth edge.
    suspect = (valid != engine_valid) | (both & (error > tolerance))
    for y, x in zip(*np.where(suspect)):
        explained = False
        for dx, dy in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
            hit, point = context.offset_hit(int(x), int(y), dx, dy)
            if hit == bool(engine_valid[y, x]) and (not hit or np.abs(point - engine[y, x]).max() <= tolerance):
                explained = True
                break
        (silhouette if explained else unexplained).append([int(x), int(y)])
    cap = max(4, valid.size // 10000)
    return {"pixels": int(valid.size), "reference_hits": int(valid.sum()), "engine_hits": int(engine_valid.sum()),
            "coverage_mismatch": len(silhouette) + len(unexplained), "silhouette_pixels": silhouette,
            "silhouette_cap": cap, "unexplained_mismatch": unexplained,
            "max_position_error_m": float(error[both & ~suspect].max()) if (both & ~suspect).any() else None,
            "over_tolerance": int((both & (error > tolerance)).sum()), "tolerance_m": tolerance,
            "passed": bool(not unexplained and len(silhouette) <= cap)}


def closed_form_checks(threads=8):
    """BRDF and sampling checks that need no scene. Returns a list of {name, value, expected, passed}."""
    import drjit as dr
    import mitsuba as mi
    import numpy as np

    class Stub:  # reuse the material math without loading a scene
        pass
    mi.set_variant("llvm_ad_rgb")
    dr.set_flag(dr.JitFlag.FastMath, False)
    dr.set_thread_count(threads)
    context = Context.__new__(Context)
    context.mi, context.dr, context.np = mi, dr, np
    context.vector, context.colour = mi.Vector3d, getattr(mi, "Color3d", mi.Vector3d)
    count = 1 << 22
    rng = mi.load_dict({"type": "independent"})
    rng.seed(7, count)
    results = []

    def material(albedo, metallic):
        return {"albedo": context.colour(albedo, albedo, albedo), "metallic": mi.Float64(metallic)}

    def view(cosine):
        return context.vector(math.sqrt(max(0.0, 1 - cosine * cosine)), 0.0, cosine) + dr.zeros(mi.Vector3d, count)

    def uniform():
        u = rng.next_2d()
        z = mi.Float64(u.x)
        r = dr.sqrt(dr.maximum(0, 1 - z * z))
        phi = 2 * math.pi * mi.Float64(u.y)
        return context.vector(r * dr.cos(phi), r * dr.sin(phi), z), 1 / (2 * math.pi)

    def record(name, value, expected, tolerance):
        results.append({"name": name, "value": float(value), "expected": float(expected), "tolerance": tolerance,
                        "passed": bool(abs(value - expected) <= tolerance * max(1e-12, abs(expected)))})

    # Rough white conductor at normal incidence: albedo integral is exactly 1 - ln 2.
    white = material(1.0, 1.0)
    wo = view(1.0)
    wi, pdf = uniform()
    _, specular, _ = context.lobes(white, wo, wi, mi.Float64(1.0))
    record("white conductor alpha=1 N=V, uniform quadrature", dr.mean(specular.x / pdf)[0], 1 - math.log(2), 2e-3)
    wi = context.sample_specular(wo, mi.Float64(1.0), rng.next_2d())
    _, specular, pdf = context.lobes(white, wo, wi, mi.Float64(1.0))
    record("white conductor alpha=1 N=V, visible-normal sampling",
           dr.mean(dr.select(pdf > 0, specular.x / pdf, 0))[0], 1 - math.log(2), 2e-3)
    # Sampling and evaluation must describe the same integrand at every roughness and view angle.
    metal = material(0.8, 1.0)
    for alpha in (0.05, 0.2, 0.64, 1.0):
        for cosine in (1.0, 0.7071, 0.2588):
            wo = view(cosine)
            wi, pdf = uniform()
            _, specular, density = context.lobes(metal, wo, wi, mi.Float64(alpha))
            quadrature = dr.mean(specular.x / pdf)[0]
            unit = dr.mean(density / pdf)[0]
            wi = context.sample_specular(wo, mi.Float64(alpha), rng.next_2d())
            _, specular, density = context.lobes(metal, wo, wi, mi.Float64(alpha))
            sampled = dr.mean(dr.select(density > 0, specular.x / density, 0))[0]
            above = dr.mean(dr.select(wi.z > 0, 1.0, 0.0))[0]
            # Uniform sampling of a narrow lobe is noisy: 3% covers alpha=0.05 at 4M samples.
            record(f"lobe energy alpha={alpha} cos={cosine}: sampled vs quadrature", sampled, quadrature, 3e-2)
            # The visible-normal density integrates to 1 over the sphere; part of the reflected
            # directions fall below the horizon, so over the upper hemisphere it must equal the
            # fraction of drawn samples that land there.
            record(f"pdf over upper hemisphere alpha={alpha} cos={cosine} = sampled fraction", unit, above, 3e-2)
    # Dielectric at normal incidence with head-on light: Lambert term is (1-0.04) albedo / pi.
    plastic = material(0.65, 0.0)
    head = context.vector(0.0, 0.0, 1.0) + dr.zeros(mi.Vector3d, count)
    diffuse, _, _ = context.lobes(plastic, head, head, mi.Float64(0.64))
    record("Lambert term at normal incidence", diffuse.x[0], 0.96 * 0.65 / math.pi, 1e-9)
    return results


def render_frame(context, args, out, positions=None, regions=None):
    np = context.np
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "tool": "tools/f13_reference.py", "tool_sha256": sha(__file__), "model": MODEL,
              "secondary_specular": context.secondary_specular, "regularization_scale": context.regularization_scale,
              "seeds": list(SEEDS),
              "convergence_relative_rmse": CONVERGENCE, "frame": context.data["frame"],
              "candidate_read": False, "phase_accepted": False}
    if positions:
        report["primary_positions"] = validate_positions(np, context, positions)
        print(json.dumps(report["primary_positions"]), flush=True)
        if not report["primary_positions"]["passed"]:
            (out / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
    image, seeds, records = converge(context, args.spp, args.spp_per_pass, regions,
                                     log=lambda line: print(line, flush=True))
    write_pfm(out / "reference.pfm", image)
    for seed, picture in zip(SEEDS, seeds):
        write_pfm(out / f"seed-{seed}.pfm", picture)
    full = context.window == (0, 0, context.width, context.height)
    report.update(window_xyxy=list(context.window),
                  window_note=None if full else "radiance is defined only inside this window; pixels outside are zero, not reference values",
                  records=records, converged=records[-1]["converged"], regions=regions or {},
                  maximum_radiance=float(image.max()))
    (out / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def command_validate(args):
    results = closed_form_checks(args.threads)
    for row in results:
        print(json.dumps(row))
    passed = all(row["passed"] for row in results)
    print(json.dumps({"closed_form_checks": len(results), "passed": passed}))
    return 0 if passed else 1


def command_render(args):
    root, data = snapshot(args.snapshot)
    context = Context(root, data, args.threads, args.secondary_specular,
                      regularization_scale=getattr(args, "regularization_scale", 1.0))
    regions = json.loads(pathlib.Path(args.regions).read_text()) if args.regions else None
    report = render_frame(context, args, pathlib.Path(args.out).resolve(), args.positions, regions)
    report["snapshot_sha256"] = scene_digest(root)
    (pathlib.Path(args.out).resolve() / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = report.get("converged", False) and report.get("primary_positions", {"passed": True})["passed"]
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="closed-form BRDF and sampling checks")
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.set_defaults(function=command_validate)
    p = sub.add_parser("render", help="converged reference for one exported frame")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--positions", help="renderer shadow-position PFM of the same frame, for the primary-hit check")
    p.add_argument("--regions", help="JSON {name: [x0,y0,x1,y1]} normalised regions that must also converge")
    p.add_argument("--spp", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--spp-per-pass", type=int, default=16)
    p.add_argument("--secondary-specular", choices=["regularized", "raw", "off"], default="regularized")
    p.add_argument("--regularization-scale", type=float, default=1.0,
                   help="k in alpha' = max(alpha_secondary, k * alpha_camera) (default 1)")
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    if hasattr(args, "spp") and (any(n < 1 for n in args.spp) or args.spp != sorted(set(args.spp))):
        parser.error("strictly increasing positive SPP checkpoints required")
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F13 reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
