"""F15 ticket 17/18 experiment: numpy emulation of the screen-space diffusion gather (shaders/subsurface.metal; it reproduced the GPU per-piece
luminance to 4 digits) with the profile variants V0 (CB15 searchlight fit, the engine), V1/V2 (dmfp fit), Kn (width scale) and LUT
(tools/f15_subsurface_profile.json), evaluated by the runner's own gate against a runner capture folder.
usage: F15_EMU_CAPTURES=<f15-check dir> python f15_subsurface_emulation.py 1 g 0 1 <scenario> <variant> [rows]   (needs mitsuba + numpy)
"""
import json, math, sys, pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'tools'))
import f15_check as fc
import f15_subsurface_reference as ref
import f15_reference as base
from f12_reference import read_pfm

SC = sys.argv[5] if len(sys.argv) > 5 else 'subsurface-volumetric-sun'
import os
T = pathlib.Path(os.environ['F15_EMU_CAPTURES']) / SC
VAR = sys.argv[6] if len(sys.argv) > 6 else 'V0'
ROWS = len(sys.argv) > 7
scale_mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
use_g = not (len(sys.argv) > 2 and sys.argv[2] == 'nog')
BOUND = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 none, 1 boundary albedo in S only, 2 in S and profile scale
SCALE_S = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
GOLDEN = 2.39996323

root, data = ref.snapshot(str(T / 'snapshot'))
plain = ref.off_data(data)
cast = base.cast_primary_rays(root, plain, 4)
valid, pos, nrm, inst, (W, H) = cast
engine = read_pfm(T / 'reference' / 'control-subsurface-off.pfm').astype(np.float64)
reference = read_pfm(T / 'reference' / 'reference.pfm')
masks = dict(np.load(T / 'reference' / 'masks.npz'))
rep = json.load(open(T / 'reference' / 'reference.json'))
spheres = rep['spheres']
blocks = ref.scene_blocks(data)
cam = data['camera']
cpos = np.asarray(cam['position'], dtype=np.float64)
fwd = np.asarray(cam['direction'], dtype=np.float64)
fwd /= np.linalg.norm(fwd)
tanY = math.tan(cam['fov_y_radians'] / 2)
tanX = tanY * W / H
L = -np.asarray(data['lights'][0]['direction'], dtype=np.float64)
L /= np.linalg.norm(L)
radiance = data['lights'][0]['intensity'] * 1.0

pos = pos.reshape(H, W, 3)
nrm = nrm.reshape(H, W, 3)
n = nrm / np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-9)
V = cpos[None, None, :] - pos
V /= np.maximum(np.linalg.norm(V, axis=2, keepdims=True), 1e-9)
Hh = V + L[None, None, :]
Hh /= np.maximum(np.linalg.norm(Hh, axis=2, keepdims=True), 1e-9)
VH = np.clip((V * Hh).sum(2), 0, 1)
F = 0.04 + 0.96 * (1 - VH) ** 5
NL = np.clip((n * L[None, None, :]).sum(2), 0, 1)
z = ((pos - cpos[None, None, :]) * fwd[None, None, :]).sum(2)
img = engine.copy()
out = engine.copy()


def burley_scale(A):
    A = min(max(A, 0.0), 1.0)
    return 1.85 - A + 7 * abs(A - 0.8) ** 3


def widen(A, g):
    if g == 0 or not use_g:
        return 1.0
    A = min(max(A, 0.0), 1.0)
    s = 4.09712 + 4.20863 * A - math.sqrt(9.59217 + 41.6808 * A + 17.7126 * A * A)
    rho = (1 - s * s) / max(1 - g * s * s, 1e-3)
    return 1.0 / max(1 - min(max(rho, 0), 1) * g, 0.05)


LUTD = json.load(open(str(REPO / 'tools' / 'f15_subsurface_profile.json')))
EDGES = np.exp(np.linspace(math.log(LUTD['x_min']), math.log(LUTD['x_max']), LUTD['bins'] + 1))
LOGE = np.log(EDGES)
X_MAX_L = LUTD['x_max']
def lut_cdf(C, g):
    gs, cs = LUTD['g_grid'], LUTD['c_grid']
    def at(gi, C):
        rows = [p for p in LUTD['profiles'] if p['g'] == gs[gi]]
        C = min(max(C, cs[0]), cs[-1])
        j = max(0, min(len(cs) - 2, int(np.searchsorted(cs, C)) - 1))
        t = (C - cs[j]) / (cs[j + 1] - cs[j])
        return (1 - t) * np.array(rows[j]['cdf']) + t * np.array(rows[j + 1]['cdf'])
    gg = min(max(g, gs[0]), gs[-1]); t = (gg - gs[0]) / (gs[1] - gs[0])
    return (1 - t) * at(0, C) + t * at(1, C)
def cdf(a, r):
    return 1 - 0.25 * np.exp(-a * r) - 0.75 * np.exp(-a * r / 3)


def shape(a, r):
    return a * (np.exp(-a * r) + np.exp(-a * r / 3))


for k, b in blocks.items():
    mat = data['instances'][k]['material']
    m = data['materials'][mat]
    base_c = np.array(m['base'][:3])
    metallic = m['metallic']
    sel = (inst.reshape(H, W) == k) & valid.reshape(H, W)
    E = (1 - F) * (1 - metallic) * radiance * NL / math.pi        # unit-albedo diffuse
    Dbase = base_c[None, None, :] * E[:, :, None]
    w = b['weight']
    A = b['color']
    S = np.zeros((H, W, 3))
    Aeff = np.array(A)
    S[sel] = (w * Aeff[None, :] * E[sel][:, None])
    a = np.zeros(3)
    curve = ref.curve_for(float(b['anisotropy']), 0.1225)
    for c in range(3):
        mfp = b['radius'] * b['radius_scale'][c]
        g = b['anisotropy']
        if VAR == 'LUT':
            a[c] = 1.0 / mfp
        elif VAR == 'V0':
            a[c] = burley_scale(A[c]) / (mfp * widen(A[c], g))
        elif VAR.startswith('K'):
            a[c] = float(VAR[1:]) ** -1 * burley_scale(A[c]) / (mfp * widen(A[c], g))
        else:
            if VAR == 'V1':
                alpha = ref.invert_observed_albedo(float(A[c]), curve)[0]
            else:
                alpha = float(ref.single_scatter_albedo(A[c], g))
            st = 1.0 / mfp
            sa = max(st * (1 - alpha), 1e-6 * st)
            stp = sa + st * alpha * (1 - g)
            ld = 1.0 / math.sqrt(3 * sa * stp)
            s3 = 3.5 + 100 * (A[c] - 0.33) ** 4
            a[c] = s3 / ld
        a[c] *= 1.0
    dmax = (1 / a).max()
    if VAR == 'LUT':
        CDFS, PDFS = [], []
        for c in range(3):
            mfp = b['radius'] * b['radius_scale'][c]
            cd = lut_cdf(A[c], b['anisotropy'])
            def mk(cd=cd, mfp=mfp):
                def cdf_f(r):
                    x = np.maximum(np.asarray(r, dtype=float) / mfp, 1e-12)
                    return np.interp(np.log(x), LOGE, cd, left=0.0, right=1.0)
                def pdf_f(r):
                    x = np.maximum(np.asarray(r, dtype=float) / mfp, 1e-12)
                    i = np.clip(np.searchsorted(EDGES, x) - 1, 0, len(EDGES) - 2)
                    inside = (x >= EDGES[0]) & (x < EDGES[-1])
                    dens = (cd[i + 1] - cd[i]) / (math.pi * (EDGES[i + 1] ** 2 - EDGES[i] ** 2)) / mfp ** 2
                    return np.where(inside, dens, 1e-9 / mfp ** 2) * 1e0
                return cdf_f, pdf_f
            f1, f2 = mk(); CDFS.append(f1); PDFS.append(f2)
    else:
        CDFS = [(lambda r, aj=a[j]: cdf(aj, r)) for j in range(3)]
        PDFS = [(lambda r, aj=a[j]: shape(aj, r) / np.maximum(r, 1e-12)) for j in range(3)]
    ys, xs = np.nonzero(sel)
    zz = z[ys, xs]
    mpp = 2 * zz * tanX / W
    N = 21 if (dmax < 2 * mpp.min()) else 55
    rhi = (X_MAX_L * max(b['radius'] * q for q in b['radius_scale'])) if VAR == 'LUT' else 28 / a.min()
    acc = np.zeros((len(ys), 3))
    wsum = np.zeros((len(ys), 3))
    rayw = pos - cpos[None, None, :]
    rayw /= np.linalg.norm(rayw, axis=2, keepdims=True)
    cosn = np.maximum(np.abs((n * rayw).sum(2)), 0.05)
    for i in range(N):
        u = (i + 0.5) / N
        lo, hi = 0.0, rhi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            c = sum(CDFS[j](mid) for j in range(3)) / 3
            if c < u:
                lo = mid
            else:
                hi = mid
        r = max(0.5 * (lo + hi), 1e-7)
        ang = i * GOLDEN
        off = (r / mpp)
        tx = np.floor(xs + 0.5 + off * math.cos(ang)).astype(int)
        ty = np.floor(ys + 0.5 + off * math.sin(ang)).astype(int)
        ok = (tx >= 0) & (ty >= 0) & (tx < W) & (ty < H)
        txc, tyc = np.clip(tx, 0, W - 1), np.clip(ty, 0, H - 1)
        ok &= sel[tyc, txc]
        zn = z[tyc, txc]
        dz = zn - zz
        d3 = np.sqrt(r * r + dz * dz)
        area = (zn / zz) ** 2 / cosn[tyc, txc]
        mixture = sum(PDFS[j](r) for j in range(3)) / 3
        wt = np.stack([PDFS[j](d3) for j in range(3)], axis=1) * (area / mixture)[:, None]
        wt = wt * ok[:, None]
        acc += wt * S[tyc, txc]
        wsum += wt
    import os
    if os.environ.get('NONORM'):
        spread = acc / N
    else:
        spread = np.where(wsum > 1e-20, acc / np.maximum(wsum, 1e-20), S[ys, xs])
    spec = engine[ys, xs] - Dbase[ys, xs]
    print('piece', k, 'N', N, 'mean S', S[ys, xs].mean(0), 'mean spread', spread.mean(0), 'mean wsum', wsum.mean(0), 'dmax', dmax, 'mpp', mpp.mean(), 'a', a)
    out[ys, xs] = spec + (1 - w) * Dbase[ys, xs] + spread
    if k == min(blocks):
        print('spec check min/mean', spec.min(), spec.mean())

print('scale_mult', scale_mult, 'use_g', use_g)
res = fc.evaluate_subsurface(np, out.astype(np.float32), reference, masks, spheres)
for p in res['pieces']:
    print('piece %2d w %.1f r %s gated %s mean %.3f p95 %.3f  cand_Y %.4f ref_Y %.4f %s' % (
        p['instance'], p['weight'], p.get('radius'), p.get('gated'), p.get('mean', -1), p.get('p95', -1),
        p.get('candidate_mean_luminance', 0), p.get('reference_mean_luminance', 0), 'MISS' if not p.get('bound_met', True) else ''))
print('worst mean', res['worst_mean'], 'pass', res['pass'])
for p in res['pieces']:
    pass
if 'radius' in SC:
    for s_ in spheres:
        if s_.get('series') is not None:
            print('BLEED', s_['series'][:14], s_['radius'], 'engine %.3f truth %.3f' % (fc.terminator_bleed(np, out.astype(np.float32), masks, s_['instance']), fc.terminator_bleed(np, reference, masks, s_['instance'])))
if ROWS:
    for k in (2, 3, 7):
        roi = fc.piece_roi(masks, k)
        ys_, xs_ = np.nonzero(roi)
        row = int(np.median(ys_))
        cols = np.nonzero(roi[row])[0]
        sample = cols[::max(1, len(cols) // 14)]
        for ch in (0, 1, 2):
            print('ROW inst %d ch %d' % (k, ch), [(round(float(reference[row, x, ch]), 3), round(float(out[row, x, ch]), 3)) for x in sample])
