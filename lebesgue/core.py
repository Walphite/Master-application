"""Rendering core for the vertical short.

The text system (CJK -> DejaVu glyph fallback, 【】 highlight parsing, alpha
blit) and the easing helpers are carried over from intro_video/core.py; the
2.5D part is new: an orthographic camera, extruded blocks drawn with PIL on a
2x supersampled canvas, soft shadows, a depth-of-field foreground layer, light
bands, chromatic aberration and paper grain.

All scene coordinates are in a 1080x1920 design space; LB_SCALE (env) scales
the output (0.5 -> 540x960 preview).
"""
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = float(os.environ.get('LB_SCALE', '1.0'))
DW, DH = 1080, 1920
W, H = int(round(DW * S)), int(round(DH * S))
FPS = 30
DUR = 72.0
NFRAMES = int(round(DUR * FPS))
SS = 2                                             # supersampling of the block canvas
N_FRAG = int(os.environ.get('LB_NFRAG', '600'))     # fragment budget (lower it if rendering is slow)


def hexc(h):
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32)


BG = hexc('#1B1B1F')
COBALT = hexc('#2F5BEA')
VERM = hexc('#E5533D')
MUST = hexc('#E8B931')
GREEN = hexc('#2E7D5B')
CREAM = hexc('#F2EDE4')
INK = hexc('#121215')


# ---------------------------------------------------------------- utilities
def clamp01(u):
    return min(max(u, 0.0), 1.0)


def smooth(a, b, x):
    if b <= a:
        return float(x >= b)
    u = clamp01((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def window(t, a, b, fi=0.4, fo=0.4):
    return smooth(a, a + fi, t) * (1.0 - smooth(b - fo, b, t))


def ease_io(u):
    u = clamp01(u)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_out(u, p=3):
    u = clamp01(u)
    return 1 - (1 - u) ** p


def ease_in(u, p=3):
    return clamp01(u) ** p


def ease_back(u, k=1.70158):
    u = clamp01(u)
    v = u - 1
    return 1 + (k + 1) * v ** 3 + k * v ** 2


def ramp(a, b, x):
    return clamp01((x - a) / (b - a)) if b > a else float(x >= b)


def spring(t, freq=9.0, damp=7.0):
    """Damped bounce starting at 1 for t=0 -> 0; 0 before t=0."""
    if t < 0:
        return 0.0
    return math.exp(-damp * t) * math.cos(freq * t)


def mix(a, b, u):
    return a * (1 - u) + b * u


def hash01(*args):
    h = 2166136261
    for a in args:
        h = ((h ^ (int(a) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / float(0xFFFFFF)


def euler(rx, ry, rz):
    """Vectorised rotation matrices R = Rz @ Ry @ Rx, shape (n,3,3)."""
    rx, ry, rz = (np.asarray(v, np.float64) for v in (rx, ry, rz))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    R = np.empty(rx.shape + (3, 3))
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


# ---------------------------------------------------------------- camera
class Cam:
    """Orthographic camera: yaw about world Y, pitch (looking down), roll, zoom.

    `target` (world) lands on design-space screen point (sx, sy)."""

    def __init__(self, yaw=-0.3, pitch=0.25, zoom=1.0, target=(0.0, 0.0, 0.0), sx=480.0, sy=640.0, roll=0.0):
        self.yaw, self.pitch, self.zoom, self.roll = yaw, pitch, zoom, roll
        self.target = np.asarray(target, np.float64)
        self.sx, self.sy = sx, sy

    def project(self, P):
        P = np.asarray(P, np.float64)
        X = P[..., 0] - self.target[0]
        Y = P[..., 1] - self.target[1]
        Z = P[..., 2] - self.target[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x1 = X * cy + Z * sy
        z1 = -X * sy + Z * cy
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y2 = Y * cp - z1 * sp
        z2 = Y * sp + z1 * cp
        u, v = x1, -y2
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        k = self.zoom * S
        xs = self.sx * S + k * (u * cr - v * sr)
        ys = self.sy * S + k * (u * sr + v * cr)
        return xs, ys, z2

    def depth_dir(self, N):
        """Camera-facing component of a world direction (positive = towards viewer)."""
        z1 = -N[..., 0] * math.sin(self.yaw) + N[..., 2] * math.cos(self.yaw)
        return N[..., 1] * math.sin(self.pitch) + z1 * math.cos(self.pitch)

    def screen(self, P):
        """World point -> design-space screen coords."""
        x, y, _ = self.project(P)
        return x / S, y / S


# ---------------------------------------------------------------- frame description
class Frame:
    def __init__(self, t):
        self.t = t
        self.cam = Cam()
        self.blocks = []        # dicts: c (n,3), h (n,3), col (n,3), R (n,3,3)|None, flat bool
        self.bgblocks = []      # far debris, drawn with bgcam
        self.trails = []        # (pts (k,3), width, col, alpha)
        self.back_lines = []    # (pts (k,3) world, width, col, alpha, dash)
        self.front_lines = []   # same, drawn over blocks
        self.front_shapes = []  # callables(draw_rgb, draw_mask, scale)
        self.cards = []         # dicts
        self.texts = []         # (ti, x, y, anchor, alpha)  design coords
        self.sprites = []       # (x, y, r, col, inten) design coords, additive
        self.images = []        # (rgba float arrays, x, y, alpha) pre-transformed overlays
        self.bands = []         # light bands: (u, strength)
        self.impact = 0.0
        self.hud = []
        self.fade = 1.0
        self.badges = []        # (badge image, x, y, alpha, scale) design coords
        self.fg = None          # foreground (depth-of-field) block batch

    def add_blocks(self, c, h, col, R=None, flat=False, layer='main'):
        c = np.asarray(c, np.float64).reshape(-1, 3)
        if len(c) == 0:
            return
        h = np.broadcast_to(np.asarray(h, np.float64), c.shape).copy()
        col = np.broadcast_to(np.asarray(col, np.float64), c.shape).copy()
        d = dict(c=c, h=h, col=col, R=None if R is None else np.asarray(R).reshape(-1, 3, 3), flat=flat)
        (self.bgblocks if layer == 'bg' else self.blocks).append(d)

    def text(self, ti, x, y, anchor='cm', alpha=1.0):
        if alpha > 0.003:
            self.texts.append((ti, x, y, anchor, min(alpha, 1.0)))


# ---------------------------------------------------------------- block rasteriser
CORNERS = np.array([[(i >> 2 & 1) * 2 - 1, (i >> 1 & 1) * 2 - 1, (i & 1) * 2 - 1] for i in range(8)], np.float64)
FACES = [
    (np.array([1.0, 0, 0]), [4, 5, 7, 6]),
    (np.array([-1.0, 0, 0]), [0, 2, 3, 1]),
    (np.array([0, 1.0, 0]), [2, 6, 7, 3]),
    (np.array([0, -1.0, 0]), [0, 1, 5, 4]),
    (np.array([0, 0, 1.0]), [1, 3, 7, 5]),
    (np.array([0, 0, -1.0]), [0, 4, 6, 2]),
]
FN = np.stack([f[0] for f in FACES])
LIGHT = np.array([-0.42, 0.75, 0.51])
LIGHT /= np.linalg.norm(LIGHT)


def _prep_blocks(cam, batches):
    if not batches:
        return None
    C = np.concatenate([b['c'] for b in batches])
    Hh = np.concatenate([b['h'] for b in batches])
    col = np.concatenate([b['col'] for b in batches])
    flat = np.concatenate([np.full(len(b['c']), b['flat']) for b in batches])
    Rs = []
    for b in batches:
        Rs.append(b['R'] if b['R'] is not None else np.broadcast_to(np.eye(3), (len(b['c']), 3, 3)))
    R = np.concatenate(Rs)
    keep = (Hh.min(axis=1) > 0.05)
    if not keep.any():
        return None
    C, Hh, col, flat, R = C[keep], Hh[keep], col[keep], flat[keep], R[keep]
    local = CORNERS[None] * Hh[:, None, :]
    V = C[:, None, :] + np.einsum('nij,nkj->nki', R, local)
    xs, ys, _ = cam.project(V)
    _, _, dz = cam.project(C)
    Nw = np.einsum('nij,fj->nfi', R, FN)
    vis = cam.depth_dir(Nw) > 1e-4
    tone = 0.66 + 0.40 * (Nw @ LIGHT)
    tone[flat] = 1.0
    order = np.argsort(dz)
    return xs, ys, vis, tone, col, order


def draw_blocks(cam, batches, drgb, dmask, dshadow=None, shadow_off=(0, 0), ss=SS, mult=1.0):
    pr = _prep_blocks(cam, batches)
    if pr is None:
        return
    xs, ys, vis, tone, col, order = pr
    xs = xs * ss
    ys = ys * ss
    Wc, Hc = W * ss, H * ss
    sq = 0.25 * (1.0 / ss) if dshadow is not None else 0
    for i in order:
        xi, yi = xs[i], ys[i]
        if xi.max() < -4 or xi.min() > Wc + 4 or yi.max() < -4 or yi.min() > Hc + 4:
            continue
        if (xi.max() - xi.min()) < 0.6 and (yi.max() - yi.min()) < 0.6:
            continue
        ci = col[i] * mult
        for f in range(6):
            if not vis[i, f]:
                continue
            idx = FACES[f][1]
            pts = [(xi[k], yi[k]) for k in idx]
            c = np.clip(ci * tone[i, f], 0, 1)
            fill = (int(c[0] * 255 + 0.5), int(c[1] * 255 + 0.5), int(c[2] * 255 + 0.5))
            drgb.polygon(pts, fill=fill)
            dmask.polygon(pts, fill=255)
            if dshadow is not None:
                dshadow.polygon([(x * sq + shadow_off[0], y * sq + shadow_off[1]) for x, y in pts], fill=255)


def poly_strip(cam, pts, width_px, taper=True):
    """Screen polygon of a tapered ribbon along a world polyline (in SS pixel units)."""
    xs, ys, _ = cam.project(pts)
    P = np.stack([xs, ys], 1) * SS
    if len(P) < 2:
        return None
    d = np.gradient(P, axis=0)
    n = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = -d[:, 1] / n, d[:, 0] / n
    w = np.linspace(0.0, 1.0, len(P)) ** 1.2 if taper else np.ones(len(P))
    w = w * width_px * SS * 0.5
    L = np.stack([P[:, 0] + nx * w, P[:, 1] + ny * w], 1)
    Rr = np.stack([P[:, 0] - nx * w, P[:, 1] - ny * w], 1)[::-1]
    return [tuple(p) for p in np.concatenate([L, Rr])]


# ---------------------------------------------------------------- static textures
def _value_noise(h, w, scale, seed):
    r = np.random.default_rng(seed)
    n = r.random((h // scale + 3, w // scale + 3)).astype(np.float32)
    img = Image.fromarray(n, 'F').resize((w + 3 * scale, h + 3 * scale), Image.BICUBIC)
    return np.asarray(img)[:h, :w]


class Textures:
    def __init__(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u, v = xx / W, yy / H
        # background: charcoal, a soft lift towards upper middle, mottling and paper grain
        mott = (_value_noise(H, W, max(8, int(260 * S)), 3) * 0.6 + _value_noise(H, W, max(4, int(70 * S)), 4) * 0.4)
        lift = np.exp(-(((u - 0.45) / 0.75) ** 2 + ((v - 0.36) / 0.55) ** 2))
        vig = 1 - 0.38 * np.clip(((u - 0.5) ** 2 / 0.25 + (v - 0.48) ** 2 / 0.30), 0, 1.6) ** 1.4
        base = BG[None, None, :] * (0.82 + 0.38 * lift[:, :, None]) * (0.94 + 0.12 * mott[:, :, None])
        rng = np.random.default_rng(11)
        grain = rng.normal(0, 1, (H, W)).astype(np.float32)
        fib = rng.normal(0, 1, (H, W)).astype(np.float32)
        fib = (fib + np.roll(fib, 1, 1) + np.roll(fib, -1, 1) + np.roll(fib, 2, 0)) * 0.5
        self.paper = (1 + 0.035 * grain + 0.06 * fib)[:, :, None].astype(np.float32)
        self.bg = base.astype(np.float32)
        self.vig = np.clip(vig, 0.55, 1)[:, :, None].astype(np.float32)
        # surface gradient for blocks: light from upper-left + slow mottling
        self.surface = ((1.10 - 0.22 * (0.6 * u + 0.4 * v)) * (0.93 + 0.14 * mott))[:, :, None].astype(np.float32)
        # light band coordinate (diagonal, ~28 deg), normalised 0..1 across the screen
        a = math.radians(62)
        D = xx * math.cos(a) * 0.55 / W + yy * math.sin(a) * 0.45 / H
        self.band_d = ((D - D.min()) / (D.max() - D.min())).astype(np.float32)


@lru_cache(None)
def gauss_kernel(r):
    R = max(1, int(math.ceil(r * 3)))
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    return np.exp(-(x * x + y * y) / (2.0 * r * r)).astype(np.float32)


def add_sprite(img, x, y, r, col, inten):
    """Additive gaussian light dot; x, y, r in design units."""
    r = max(0.6, r * S)
    k = gauss_kernel(round(r * 4) / 4.0)
    R = k.shape[0] // 2
    cx, cy = int(round(x * S)), int(round(y * S))
    x0, y0 = cx - R, cy - R
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + k.shape[1], W), min(y0 + k.shape[0], H)
    if xa >= xb or ya >= yb:
        return
    img[ya:yb, xa:xb] += k[ya - y0:yb - y0, xa - x0:xb - x0, None] * (np.asarray(col, np.float32) * inten)


# ---------------------------------------------------------------- text
NOTO = '/usr/share/fonts/opentype/noto/'
DV = '/usr/share/fonts/truetype/dejavu/'
CHAINS = {
    'sans': [(NOTO + 'NotoSansCJK-Regular.ttc', 2), (DV + 'DejaVuSans.ttf', 0)],
    'sansm': [(NOTO + 'NotoSansCJK-Medium.ttc', 2), (DV + 'DejaVuSans.ttf', 0)],
    'sansb': [(NOTO + 'NotoSansCJK-Bold.ttc', 2), (DV + 'DejaVuSans-Bold.ttf', 0)],
    'black': [(NOTO + 'NotoSansCJK-Black.ttc', 2), (DV + 'DejaVuSans-Bold.ttf', 0)],
    'serif': [(NOTO + 'NotoSerifCJK-Regular.ttc', 2), (DV + 'DejaVuSerif.ttf', 0), (DV + 'DejaVuSans.ttf', 0)],
    'serifb': [(NOTO + 'NotoSerifCJK-Bold.ttc', 2), (DV + 'DejaVuSerif-Bold.ttf', 0), (DV + 'DejaVuSans-Bold.ttf', 0)],
    # math: DejaVu first so that ∫ μ ∞ ℚ come from one consistent face, CJK falls back to Noto
    'math': [(DV + 'DejaVuSerif.ttf', 0), (DV + 'DejaVuSans.ttf', 0), (NOTO + 'NotoSerifCJK-Regular.ttc', 2)],
    'mathb': [(DV + 'DejaVuSerif-Bold.ttf', 0), (DV + 'DejaVuSans-Bold.ttf', 0), (NOTO + 'NotoSerifCJK-Bold.ttc', 2)],
}


@lru_cache(None)
def _font(path, idx, size):
    return ImageFont.truetype(path, size, index=idx)


@lru_cache(None)
def _has(path, idx, ch):
    if ch.isspace():
        return True
    f = _font(path, idx, 40)
    m = f.getmask(ch)
    nd = f.getmask('￿')
    if m.size == nd.size and bytes(m) == bytes(nd):
        return False
    return m.getbbox() is not None


def _pick(kind, ch):
    for (p, i) in CHAINS[kind]:
        if _has(p, i, ch):
            return p, i
    return CHAINS[kind][0]


def parse(text, col, hcol):
    """【...】 -> highlight colour."""
    out, c = [], col
    for ch in text:
        if ch == '【':
            c = hcol
        elif ch == '】':
            c = col
        else:
            out.append((ch, tuple(float(v) for v in c)))
    return out


_TC = {}


def text_img(text, size, kind='sans', col=CREAM, hcol=MUST, spacing=0.0, shadow=0.0, px=False):
    """Rasterise one line. size in design px (or real px if px=True). Returns dict with
    premultiplied rgb, alpha a, width/height in real pixels and per-char x ranges."""
    key = (text, size, kind, tuple(np.round(col, 3)), tuple(np.round(hcol, 3)), spacing, shadow, px)
    if key in _TC:
        return _TC[key]
    sz = max(4, int(round(size if px else size * S)))
    sp = spacing if px else spacing * S
    chars = parse(text, col, hcol)
    main = _font(*CHAINS[kind][0], sz)
    asc, desc = main.getmetrics()
    pad = int(6 + shadow * sz * 0.5)
    items, x = [], 0.0
    for ch, c in chars:
        p, i = _pick(kind, ch)
        f = _font(p, i, sz)
        adv = f.getlength(ch)
        items.append((x, adv, ch, f, c))
        x += adv + sp
    wid = int(math.ceil(x - sp)) if items else 1
    hh = asc + desc
    imgw, imgh = wid + 2 * pad, hh + 2 * pad
    cols = {}
    for (xx, adv, ch, f, c) in items:
        if c not in cols:
            cols[c] = Image.new('L', (imgw, imgh), 0)
        ImageDraw.Draw(cols[c]).text((pad + xx, pad + asc), ch, font=f, fill=255, anchor='ls')
    a = np.zeros((imgh, imgw), np.float32)
    rgb = np.zeros((imgh, imgw, 3), np.float32)
    for c, im in cols.items():
        m = np.asarray(im, np.float32) / 255.0
        a += m
        rgb += m[:, :, None] * np.array(c, np.float32)
    a = np.clip(a, 0, 1)
    if shadow > 0:
        sa = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(max(1.0, sz * 0.12))), np.float32) / 255.0
        sa = np.clip(sa * 1.6, 0, 1) * shadow
        rgb = rgb + INK[None, None, :] * (sa * (1 - a))[:, :, None]
        a = a + sa * (1 - a)
    ti = dict(rgb=rgb, a=a, w=wid, h=hh, pad=pad, asc=asc,
              chars=[(xx + pad, adv, ch) for (xx, adv, ch, f, c) in items])
    _TC[key] = ti
    return ti


def anchor_xy(w, h, x, y, anchor):
    ax, ay = anchor[0], anchor[1]
    if ax == 'c':
        x -= w / 2
    elif ax == 'r':
        x -= w
    if ay == 'm':
        y -= h * 0.5
    elif ay == 'b':
        y -= h
    return x, y


def blit_arr(dst, rgb, a, x0, y0, alpha):
    """Premultiplied 'over' blit at integer real-pixel position."""
    if alpha <= 0.002:
        return
    h, w = a.shape
    x0, y0 = int(round(x0)), int(round(y0))
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if xa >= xb or ya >= yb:
        return
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0, None] * alpha
    reg = dst[ya:yb, xa:xb]
    reg *= (1 - sa)
    reg += rgb[ya - y0:yb - y0, xa - x0:xb - x0] * alpha


def blit_text(dst, ti, x, y, anchor='lt', alpha=1.0):
    """x, y in design coords; anchor relative to the glyph box."""
    xx, yy = anchor_xy(ti['w'], ti['h'], x * S, y * S, anchor)
    blit_arr(dst, ti['rgb'], ti['a'], xx - ti['pad'], yy - ti['pad'], alpha)


def transform_ti(ti, scale=1.0, sx=1.0, rot=0.0):
    """Scaled / rotated copy of a text image -> (rgb, a, cx_off, cy_off) where the offsets
    locate the glyph-box centre inside the new array."""
    a8 = Image.fromarray((ti['a'] * 255).astype(np.uint8))
    rgbm = np.zeros(ti['rgb'].shape, np.float32)
    nz = ti['a'] > 1e-4
    rgbm[nz] = ti['rgb'][nz] / ti['a'][nz][:, None]
    im = Image.fromarray(np.clip(rgbm * 255, 0, 255).astype(np.uint8), 'RGB')
    im.putalpha(a8)
    w0, h0 = im.size
    nw, nh = max(1, int(round(w0 * scale * abs(sx)))), max(1, int(round(h0 * scale)))
    im = im.resize((nw, nh), Image.BICUBIC)
    if abs(rot) > 1e-3:
        im = im.rotate(math.degrees(rot), resample=Image.BICUBIC, expand=True)
    arr = np.asarray(im, np.float32) / 255.0
    a = arr[:, :, 3]
    return arr[:, :, :3] * a[:, :, None], a, im.size[0] / 2.0, im.size[1] / 2.0


def blit_centered(dst, ti, x, y, alpha=1.0, scale=1.0, sx=1.0, rot=0.0):
    """Blit a text image centred at design (x, y) with scale/rotation."""
    if alpha <= 0.002 or scale <= 0.01:
        return
    if abs(scale - 1) < 1e-3 and abs(sx - 1) < 1e-3 and abs(rot) < 1e-3:
        blit_arr(dst, ti['rgb'], ti['a'], x * S - (ti['w'] / 2 + ti['pad']), y * S - (ti['h'] / 2 + ti['pad']), alpha)
        return
    rgb, a, cx, cy = transform_ti(ti, scale, sx, rot)
    blit_arr(dst, rgb, a, x * S - cx, y * S - cy, alpha)


# ---------------------------------------------------------------- math typesetting
_MC = {}


def formula_img(parts, size, col=CREAM, kind='math'):
    """parts: list of str or ('lim', base, sub, sup). Returns a text-image dict."""
    key = (tuple(parts) if not any(isinstance(p, tuple) for p in parts) else repr(parts), size, tuple(np.round(col, 3)), kind)
    if key in _MC:
        return _MC[key]
    sz = int(round(size * S))
    pieces, x = [], 0.0
    small = int(sz * 0.52)
    for p in parts:
        if isinstance(p, str):
            ti = text_img(p, sz, kind, col, col, px=True)
            pieces.append((ti, x, 0))
            x += ti['w']
        else:
            _, base, sub, sup = p
            bi = text_img(base, int(sz * 1.25), kind, col, col, px=True)
            si = text_img(sub, small, kind, col, col, px=True)
            ui = text_img(sup, small, kind, col, col, px=True)
            pieces.append((bi, x, -0.12 * sz))
            bx = x + bi['w'] * 0.80
            pieces.append((si, bx, 0.48 * sz))
            pieces.append((ui, bx + 0.10 * sz, -0.55 * sz))
            x = bx + max(si['w'], ui['w'] + 0.1 * sz) + 0.12 * sz
    pad = int(sz * 0.9)
    Wf, Hf = int(x) + 2 * pad, int(sz * 1.4) + 2 * pad
    rgb = np.zeros((Hf, Wf, 3), np.float32)
    a = np.zeros((Hf, Wf), np.float32)
    base_y = pad + int(sz * 0.2)
    for ti, px, dy in pieces:
        x0 = int(pad + px - ti['pad'])
        # align baselines: glyph baseline sits at pad + asc in ti
        y0 = int(base_y + sz * 0.9 + dy - (ti['pad'] + ti['asc']))
        h, w = ti['a'].shape
        ys, xs = slice(max(y0, 0), min(y0 + h, Hf)), slice(max(x0, 0), min(x0 + w, Wf))
        sa = ti['a'][ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        a[ys, xs] = np.maximum(a[ys, xs], sa)
        rgb[ys, xs] = np.maximum(rgb[ys, xs], ti['rgb'][ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0])
    nzr = np.where(a.max(axis=1) > 0.01)[0]
    nzc = np.where(a.max(axis=0) > 0.01)[0]
    r0, r1 = max(nzr[0] - 6, 0), min(nzr[-1] + 7, Hf)
    c0, c1 = max(nzc[0] - 6, 0), min(nzc[-1] + 7, Wf)
    out = dict(rgb=rgb[r0:r1, c0:c1], a=a[r0:r1, c0:c1], w=c1 - c0 - 12, h=r1 - r0 - 12, pad=6, asc=0, chars=[])
    _MC[key] = out
    return out


# ---------------------------------------------------------------- 2.5D badges (screen space)
_BC = {}


def badge_img(text, size, face, kind='sansb', tcol=CREAM, depth=7, radius=12, padx=22, pady=12, hcol=MUST):
    """A rounded colour block with thickness and a line of text, as premultiplied rgb/a (real px)."""
    key = (text, size, tuple(np.round(face, 3)), kind, tuple(np.round(tcol, 3)), depth, radius, padx, pady)
    if key in _BC:
        return _BC[key]
    ti = text_img(text, size, kind, tcol, hcol)
    k = S * 2
    bw, bh = ti['w'] + 2 * padx * S, ti['h'] + 2 * pady * S
    dep = depth * S
    sh = 16 * S
    Wb, Hb = int(bw + dep + 2 * sh), int(bh + dep + 2 * sh)
    big = Image.new('RGBA', (int(Wb * 2), int(Hb * 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    # soft shadow
    shadow = Image.new('L', big.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        [2 * (sh + dep * 0.6), 2 * (sh + dep * 1.4), 2 * (sh + bw + dep * 0.6), 2 * (sh + bh + dep * 1.4)],
        radius=int(2 * radius * S), fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10 * S * 2 / 2))
    sl = Image.new('RGBA', big.size, (8, 8, 10, 0))
    sl.putalpha(shadow)
    big = Image.alpha_composite(big, sl)
    d = ImageDraw.Draw(big)
    dark = tuple(int(v * 255) for v in np.clip(face * 0.55, 0, 1)) + (255,)
    for s in range(int(dep * 2), 0, -1):
        d.rounded_rectangle([2 * sh + s, 2 * sh + s, 2 * (sh + bw) + s, 2 * (sh + bh) + s],
                            radius=int(2 * radius * S), fill=dark)
    # face with a vertical grainy gradient
    fw, fh = int(2 * bw), int(2 * bh)
    yy = np.linspace(0, 1, fh)[:, None, None]
    grad = face[None, None, :] * (1.12 - 0.24 * yy)
    rng = np.random.default_rng(len(text) * 7 + int(size))
    grad = grad * (1 + 0.05 * rng.normal(0, 1, (fh, fw, 1)))
    gimg = Image.fromarray((np.clip(grad, 0, 1) * 255).astype(np.uint8), 'RGB')
    m = Image.new('L', (fw, fh), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, fw - 1, fh - 1], radius=int(2 * radius * S), fill=255)
    big.paste(gimg, (int(2 * sh), int(2 * sh)), m)
    # top highlight edge
    ImageDraw.Draw(big).rounded_rectangle([2 * sh, 2 * sh, 2 * (sh + bw), 2 * (sh + bh)], radius=int(2 * radius * S),
                                          outline=tuple(int(v * 255) for v in np.clip(face * 1.35 + 0.08, 0, 1)) + (255,),
                                          width=max(1, int(2 * S)))
    big = big.resize((Wb, Hb), Image.LANCZOS)
    arr = np.asarray(big, np.float32) / 255.0
    a = arr[:, :, 3].copy()
    rgb = arr[:, :, :3] * a[:, :, None]
    tx, ty = sh + padx * S - ti['pad'], sh + pady * S - ti['pad']
    blit_arr(rgb, ti['rgb'], ti['a'], tx, ty, 1.0)
    h, w = ti['a'].shape
    x0, y0 = int(round(tx)), int(round(ty))
    a[y0:y0 + h, x0:x0 + w] = np.maximum(a[y0:y0 + h, x0:x0 + w], ti['a'][:Hb - y0, :Wb - x0])
    out = dict(rgb=rgb, a=a, w=Wb - 2 * sh, h=Hb - 2 * sh, pad=sh, asc=0, chars=[],
               face_c=(sh + bw / 2, sh + bh / 2))
    _BC[key] = out
    return out
