#!/usr/bin/env python3
"""F15.1 (ticket 22, C22-A1): independent CPU model of the multi-bounce mirror fixtures (no GPU, no engine code).

Everything is a perfectly smooth metal (F0 = baseColor 0.8, Schlick Fresnel) under the declared corpus rig's constant environment (radiance 0.5);
a ray that leaves the scene returns the environment.  The model follows the camera ray through the mirrors and returns, per pixel, the radiance
the engine's declared bounded estimator must reproduce and the number of mirror events behind it:

  class 0   the camera ray misses everything: the environment
  class k   k mirror events (the camera-visible mirror is event 1, every ray its estimator traces that meets another mirror adds one) before
            the chain ends in the environment: radiance = 0.5 * product of the Fresnel factors of the k events
  class 5   the chain is longer than the declared bound (1 + MAX_EXTRA_BOUNCES + 1 events): the engine drops the last hit, radiance 0 (a
            declared approximation, never gated)

The light rig's sun is not modelled: pixels whose any event leaves in a direction within SUN_MASK_DEGREES of the sun are masked (the sun glint
is the DI estimator's, not the bounce chain's).  Pixels are used only where the class is constant over a window of 2 * ERODE + 1 pixels
(silhouettes, mirror edges and the denoiser's reach are excluded).
"""
from __future__ import annotations

import math

MIRROR_F0 = 0.8
ENVIRONMENT = 0.5
MAX_EXTRA_BOUNCES = 3
SUN_TOWARD = (-0.45, 0.55, 0.70)
SUN_MASK_DEGREES = 8.0
ERODE = 3
EPS = 1e-5


def normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add_scaled(a, b, s):
    return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


DIELECTRIC_F0 = 0.04  # a black dielectric (baseColor 0, metallic 0): only its specular reflection remains


def schlick(cos_theta, f0=MIRROR_F0):
    return f0 + (1.0 - f0) * (1.0 - min(max(cos_theta, 0.0), 1.0)) ** 5


class Quad:
    """Parallelogram: corner + s * edge_u + t * edge_v, s and t in [0, 1]; the normal is cross(edge_u, edge_v).  Single sided: a ray from
    behind is a miss for the engine (culled), so it is here too."""

    def __init__(self, corner, edge_u, edge_v):
        self.corner, self.u, self.v = corner, edge_u, edge_v
        self.normal = normalize(cross(edge_u, edge_v))
        self.uu, self.vv = dot(edge_u, edge_u), dot(edge_v, edge_v)

    def hit(self, origin, direction):
        denom = dot(direction, self.normal)
        if denom >= -1e-9:
            return None
        t = dot(sub(self.corner, origin), self.normal) / denom
        if t <= EPS:
            return None
        rel = sub(add_scaled(origin, direction, t), self.corner)
        s, w = dot(rel, self.u) / self.uu, dot(rel, self.v) / self.vv
        if not (0.0 <= s <= 1.0 and 0.0 <= w <= 1.0):
            return None
        return t, self.normal


class Sphere:
    def __init__(self, centre, radius):
        self.centre, self.radius = centre, radius

    def hit(self, origin, direction):
        oc = sub(origin, self.centre)
        b = dot(oc, direction)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        for t in (-b - root, -b + root):
            if t > EPS:
                n = normalize(sub(add_scaled(origin, direction, t), self.centre))
                if dot(n, direction) < 0.0:  # front face only: the inside of a closed mesh is culled
                    return t, n
        return None


def nearest(scene, origin, direction):
    best = None
    for shape in scene:
        h = shape.hit(origin, direction)
        if h and (best is None or h[0] < best[0]):
            best = h
    return best


def radiance(scene, origin, direction, bounces=MAX_EXTRA_BOUNCES, f0=MIRROR_F0):
    """Returns (radiance, class, sun_masked) of one camera ray; `bounces` = extra bounces of the estimator (0: single bounce)."""
    sun = normalize(SUN_TOWARD)
    cos_mask = math.cos(math.radians(SUN_MASK_DEGREES))
    hit = nearest(scene, origin, direction)
    if hit is None:
        return ENVIRONMENT, 0, False
    throughput, events, masked = 1.0, 0, False
    o, d = origin, direction
    for event in range(bounces + 2):
        t, n = hit
        point = add_scaled(o, d, t)
        events += 1
        fresnel = schlick(-dot(n, d), f0)
        throughput *= fresnel
        r = normalize(add_scaled(d, n, -2.0 * dot(d, n)))
        masked = masked or dot(r, sun) > cos_mask
        if event == bounces + 1:
            return 0.0, 5, masked  # the chain ends at a mirror-like hit: dropped
        o, d = add_scaled(point, n, 1e-4), r
        hit = nearest(scene, o, d)
        if hit is None:
            return ENVIRONMENT * throughput, events, masked
    raise AssertionError("unreachable")


def frame(bounds_min, bounds_max, aspect, fov_y_degrees=60.0, margin=1.12):
    """Mirror of frameCorpusRig for yaw 0, pitch 0: (camera origin, tan(fov/2))."""
    tan_half = math.tan(0.5 * math.radians(fov_y_degrees))
    centre = [0.5 * (a + b) for a, b in zip(bounds_min, bounds_max)]
    half = [0.5 * (b - a) for a, b in zip(bounds_min, bounds_max)]
    distance = 1e-3
    for corner in range(8):
        r = half[0] if corner & 1 else -half[0]
        u = half[1] if corner & 2 else -half[1]
        b = half[2] if corner & 4 else -half[2]
        distance = max(distance, margin * max(abs(r) / (tan_half * aspect), abs(u) / tan_half) + b)
    return (centre[0], centre[1], centre[2] + distance), tan_half, distance


def render(scene, width, height, bounds_min, bounds_max, bounces=MAX_EXTRA_BOUNCES, f0=MIRROR_F0):
    """Per-pixel grids: radiance, class, masked."""
    origin, tan_half, _ = frame(bounds_min, bounds_max, width / height)
    aspect = width / height
    rad = [[0.0] * width for _ in range(height)]
    cls = [[0] * width for _ in range(height)]
    mask = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            ndc_x = (x + 0.5) / width * 2.0 - 1.0
            ndc_y = 1.0 - (y + 0.5) / height * 2.0
            rad[y][x], cls[y][x], mask[y][x] = radiance(scene, origin, normalize((ndc_x * tan_half * aspect, ndc_y * tan_half, -1.0)), bounces, f0)
    return rad, cls, mask


def stable_mask(cls, mask, erode=ERODE):
    """True where the class is constant and nothing is sun-masked over the (2 erode + 1)^2 window."""
    height, width = len(cls), len(cls[0])
    value = [[(cls[y][x] if not mask[y][x] else -1) for x in range(width)] for y in range(height)]

    def window(grid, op):
        horizontal = [[op(grid[y][max(0, x - erode):x + erode + 1]) for x in range(width)] for y in range(height)]
        return [[op(horizontal[j][x] for j in range(max(0, y - erode), min(height, y + erode + 1))) for x in range(width)] for y in range(height)]
    low, high = window(value, min), window(value, max)
    return [[low[y][x] == high[y][x] and low[y][x] >= 0 and erode <= x < width - erode and erode <= y < height - erode for x in range(width)] for y in range(height)]


def class_counts(cls, stable):
    counts = {}
    for y, row in enumerate(cls):
        for x, value in enumerate(row):
            if stable[y][x]:
                counts[value] = counts.get(value, 0) + 1
    return counts


# Fixture scenes (world metres): quads (corner, edge_u, edge_v), spheres (centre, radius).  tools/f15_reflection_bounce_check.py writes the same
# geometry as glTF meshes.
def _tilted_wall_and_ball(width=2.4, height=1.2, yaw_degrees=65.0, radius=0.6, gap=0.25):
    """A mirror wall turned by `yaw_degrees` about y (so its reflections go sideways, where the ball and the wall see each other) and a mirror
    ball next to its centre, `gap` metres off the wall surface."""
    p = math.radians(yaw_degrees)
    u = (width * math.cos(p), 0.0, width * math.sin(p))
    corner = (-0.5 * width * math.cos(p), -0.5 * height, -0.5 * width * math.sin(p))
    n = (-math.sin(p), 0.0, math.cos(p))
    centre = tuple(n[i] * (radius + gap) for i in range(3))
    return {"quads": [(corner, u, (0.0, height, 0.0))], "spheres": [(centre, radius)]}


SCENES = {
    # Two facing walls 1.0 apart, 8 long: the camera ray bounces up to 4 times between them before it leaves (classes 1..4).
    "corridor": {"quads": [((-0.5, -0.5, 0.0), (0.0, 0.0, -8.0), (0.0, 1.0, 0.0)),   # left wall x = -0.5, normal +x
                           ((0.5, -0.5, -8.0), (0.0, 0.0, 8.0), (0.0, 1.0, 0.0))],   # right wall x = +0.5, normal -x
                 "spheres": []},
    # A mirror ball in front of a tilted mirror wall: ball sees the wall (2), the wall sees the ball (2), the ball sees the wall seeing the ball (3).
    "ball": _tilted_wall_and_ball(),
}
# Same corridor made of a smooth black dielectric (F0 0.04): the owner's "every smooth surface" scope.  Entry: scene description + F0.
SCENES["dielectric-corridor"] = SCENES["corridor"]
SCENE_F0 = {"dielectric-corridor": DIELECTRIC_F0}


def build_scene(description):
    return [Quad(*q) for q in description["quads"]] + [Sphere(c, r) for c, r in description["spheres"]]


def scene_bounds(description):
    points = []
    for corner, u, v in description["quads"]:
        for a in (0, 1):
            for b in (0, 1):
                points.append(tuple(corner[i] + a * u[i] + b * v[i] for i in range(3)))
    for centre, radius in description["spheres"]:
        for sign in (-1, 1):
            for i in range(3):
                p = list(centre)
                p[i] += sign * radius
                points.append(tuple(p))
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or SCENES:
        desc = SCENES[name]
        lo, hi = scene_bounds(desc)
        rad, cls, mask = render(build_scene(desc), 160, 90, lo, hi)
        counts = class_counts(cls, stable_mask(cls, mask, 1))
        print(name, "classes (stable, 160x90):", dict(sorted(counts.items())))
        for row in cls[::3]:
            print("".join(".#" [0] if v == 0 else str(v) for v in row[::2]))
