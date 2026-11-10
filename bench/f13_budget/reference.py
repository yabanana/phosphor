#!/usr/bin/env python3
"""Render an independent, sparse diffuse GI control from an F12 snapshot."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools'))
from f12_oracle_validate import ray
from f12_reference import build_scene, read_pfm, scene_digest, snapshot, write_pfm
from f13_f14_check import sha


def seed_agreement(first, second):
    """Use the stricter of the two seed-relative errors. Require signal support."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('incompatible independent renders')
    support = min(float(np.sqrt(np.mean(image**2))) for image in (first, second))
    error = float(np.sqrt(np.mean((first-second)**2)))
    return support, error / max(support, 1e-6)


def validate_camera(sensor, camera, crop):
    x, y, width, height = crop
    maximum_direction = maximum_origin = 0.0
    for u, v in ((0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)):
        expected_origin, expected_direction = ray(camera, x+u*width, y+v*height)
        actual, _ = sensor.sample_ray(0, 0, [u, v], [0, 0])
        maximum_direction = max(maximum_direction, float(np.max(np.abs(np.asarray(actual.d)-expected_direction))))
        maximum_origin = max(maximum_origin, float(np.max(np.abs(np.asarray(actual.o)-expected_origin))))
    return {'direction_error': maximum_direction, 'origin_error': maximum_origin,
            'passed': maximum_direction < 1e-6 and maximum_origin < 1e-4}


def primary_receiver(mi, scene, original):
    import drjit as dr

    ray = mi.RayDifferential3f(original)
    # Keep raster-facing rules on camera rays only. The path integrator still
    # uses the complete scene for visibility and subsequent scattering.
    for _ in range(256):
        hit = scene.ray_intersect(ray)
        if not hit.is_valid():
            return ray, hit
        bsdf = hit.bsdf()
        flags = bsdf.flags()
        back_face = dr.dot(hit.n, ray.d) > 0 and not mi.has_flag(flags, mi.BSDFFlags.BackSide)
        transparent = mi.has_flag(flags, mi.BSDFFlags.Null) and dr.all(bsdf.eval_null_transmission(hit) == 1)
        if not back_face and not transparent:
            return ray, hit
        epsilon = 4 * float(np.spacing(np.float32(max(1.0, float(dr.max(dr.abs(hit.p)))))))
        distance = float(hit.t) + epsilon
        origin = ray(distance)
        if dr.dot(origin-ray.o, ray.d) <= 0:
            raise RuntimeError('camera ray failed to advance past a rejected surface')
        ray.o = origin
        ray.maxt = max(0.0, ray.maxt-distance)
    raise RuntimeError('camera ray exceeded the primary visibility traversal bound')


def register_camera_path(mi, camera, crop, receivers):
    import drjit as dr

    x, y, width, height = crop
    forward = np.asarray(camera['direction'], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera['up'])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    forward, right, up = (mi.Vector3f(*axis) for axis in (forward, right, up))
    film_width, film_height = camera['width'], camera['height']
    pixels_per_slope = film_height / (2*math.tan(camera['fov_y_radians']/2))

    def pixel_index(direction):
        depth = dr.dot(direction, forward)
        column = math.floor(film_width/2 + pixels_per_slope*dr.dot(direction, right)/depth)-x
        row = math.floor(film_height/2 - pixels_per_slope*dr.dot(direction, up)/depth)-y
        column = min(width-1, max(0, column))
        row = min(height-1, max(0, row))
        return row*width+column

    class CameraPath(mi.SamplingIntegrator):
        def __init__(self, properties):
            super().__init__(properties)
            self.path = mi.load_dict({'type': 'path', 'max_depth': properties['depth'],
                                     'rr_depth': properties['roulette_depth']})

        def sample(self, scene, sampler, ray, medium=None, active=True):
            # The GPU samples each receiver at its pixel centre. Retain the
            # independent transport samples while using that same camera ray.
            visible_ray = receivers[pixel_index(ray.d)]
            return self.path.sample(scene, sampler, visible_ray, medium, active)

        def to_string(self):
            return 'SponzaCameraPath'

    mi.register_integrator('sponza_camera_path', CameraPath)
    return pixel_index


def validate_receivers(mi, scene, positions, crop, receivers, pixel_index):
    x, y, width, height = crop
    misses = mismatches = 0
    maximum = 0.0
    for row in range(height):
        for column in range(width):
            uv = [(column+0.5)/width, (row+0.5)/height]
            ray, _ = scene.sensors()[0].sample_ray(0, 0, uv, [0, 0])
            if pixel_index(ray.d) != row*width+column:
                raise ValueError('camera sample maps to the wrong reference pixel')
            visible_ray, hit = primary_receiver(mi, scene, ray)
            receivers.append(visible_ray)
            if not hit.is_valid():
                misses += 1
                continue
            error = float(np.max(np.abs(np.asarray(hit.p)-positions[y+row, x+column])))
            maximum = max(maximum, error)
            mismatches += error > 1e-4
    return {'pixels': width*height, 'misses': misses, 'mismatches': mismatches,
            'max_position_error_m': maximum, 'position_cap_m': 1e-4,
            'passed': misses == 0 and mismatches == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--positions', type=Path, required=True, help='same-frame shadow-position PFM and JSON sidecar')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crop', type=int, nargs=4, required=True, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--max-spp', type=int, choices=(256, 1024, 4096, 16384, 65536), default=4096)
    args = parser.parse_args()
    limits_path = REPO / 'tools/testdata/f12/oracle-protocol-v1.json'
    limits = json.loads(limits_path.read_text())['reference']
    root, data = snapshot(args.snapshot)
    camera = data['camera']
    x, y, width, height = args.crop
    if min(x, y) < 0 or min(width, height) <= 0 or x+width > camera['width'] or y+height > camera['height']:
        parser.error('crop lies outside the exported camera film')
    if any(camera.get('jitter_pixels', [0, 0])):
        parser.error('sparse diffuse control requires an unjittered camera')
    positions = read_pfm(args.positions)
    sidecar = args.positions.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    if (metadata.get('schema') != 'phosphor.dynamic-lighting-capture.v1' or
        metadata.get('frame') != data['frame'] or metadata.get('signal') != 4 or
        positions.shape != (camera['height'], camera['width'], 3)):
        parser.error('position readback identity differs from the snapshot')
    args.out.mkdir(parents=True)
    plan = {'schema': 1, 'frame': data['frame'], 'crop_xywh': args.crop,
            'snapshot_sha256': scene_digest(root), 'adapter_sha256': sha(REPO/'tools/f12_reference.py'),
            'positions_sha256': sha(args.positions), 'position_sidecar_sha256': sha(sidecar),
            'runner_sha256': sha(Path(__file__)),
            'limits_sha256': sha(limits_path), 'reference': limits, 'max_spp': args.max_spp,
            'scope': 'sparse centre-sampled surface indirect diffuse control', 'candidate_read': False}
    (args.out/'protocol.json').write_text(json.dumps(plan, indent=2)+'\n')
    mi, description, differences = build_scene(root, data, 'diffuse', False)
    rays = []
    pixel_index = register_camera_path(mi, camera, args.crop, rays)
    description['sensor']['film'].update(crop_offset_x=x, crop_offset_y=y, crop_width=width, crop_height=height)
    description['integrator'] = {'type': 'sponza_camera_path', 'depth': limits['full_depth'],
                                 'roulette_depth': limits['rr_depth']}
    full = mi.load_dict(description)
    camera_check = validate_camera(full.sensors()[0], camera, args.crop)
    (args.out/'camera.json').write_text(json.dumps(camera_check, indent=2)+'\n')
    if not camera_check['passed']:
        raise ValueError('Mitsuba crop rays differ from exported camera')
    receivers = validate_receivers(mi, full, positions, args.crop, rays, pixel_index)
    (args.out/'receivers.json').write_text(json.dumps(receivers, indent=2)+'\n')
    print(json.dumps(receivers), flush=True)
    if not receivers['passed']:
        return 1
    description['integrator']['depth'] = limits['direct_depth']
    direct = mi.load_dict(description)

    records = []
    for spp in limits['spp_checkpoints']:
        if spp > args.max_spp:
            break
        images = []
        for seed in limits['seeds']:
            start = time.monotonic()
            total = np.array(mi.render(full, spp=spp, seed=seed), dtype=np.float32)
            first = np.array(mi.render(direct, spp=spp, seed=seed), dtype=np.float32)
            image = total-first
            write_pfm(args.out/f'indirect-{spp}-{seed}.pfm', image)
            images.append(image)
            print(f'Frame {data["frame"]}, {spp} spp, seed {seed}: {time.monotonic()-start:.1f}s', flush=True)
        support, nrmse = seed_agreement(*images)
        converged = support > 1e-6 and nrmse <= limits['convergence_relative_rmse']
        records.append({'spp': spp, 'minimum_seed_rms': support, 'seed_nrmse': nrmse, 'converged': converged})
        (args.out/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        if converged:
            write_pfm(args.out/'indirect.pfm', np.mean(images, axis=0, dtype=np.float64))
            break
    report = {**plan, 'renderer': 'Mitsuba', 'version': mi.__version__, 'variant': 'scalar_rgb',
              'camera_check': camera_check, 'receivers': receivers, 'model_differences': differences, 'records': records,
              'converged': records[-1]['converged'], 'phase_accepted': False}
    (args.out/'reference_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['converged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
