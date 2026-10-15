"""Rendering core: additive float canvas, glow, starfield, text, HUD, subtitles."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
DUR = 92.0
NFRAMES = int(round(DUR * FPS))

GOLD = np.array([1.00, 0.70, 0.28], np.float32)
GOLD2 = np.array([1.00, 0.80, 0.45], np.float32)
WARM = np.array([1.00, 0.90, 0.74], np.float32)
WHITE = np.array([0.95, 0.94, 0.92], np.float32)
TEAL = np.array([0.18, 0.86, 0.78], np.float32)
BLUEW = np.array([0.70, 0.80, 1.00], np.float32)
NAVY = np.array([0.10, 0.18, 0.45], np.float32)


# ---------------------------------------------------------------- utilities
def smooth(a, b, x):
    if b <= a:
        return float(x >= b)
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def window(t, a, b, fi=0.5, fo=0.5):
    return smooth(a, a + fi, t) * (1.0 - smooth(b - fo, b, t))


def ease_io(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def hash01(*args):
    h = 2166136261
    for a in args:
        h = ((h ^ (int(a) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / float(0xFFFFFF)


def arclen(pts):
    d = np.diff(pts, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])


def sub_polyline(pts, cum, s0, s1):
    """Part of a polyline between arc lengths s0..s1 (with interpolated ends)."""
    s0 = max(s0, 0.0)
    s1 = min(s1, cum[-1])
    if s1 <= s0:
        return None, None
    inner = (cum > s0) & (cum < s1)
    ss = np.concatenate([[s0], cum[inner], [s1]])
    x = np.interp(ss, cum, pts[:, 0])
    y = np.interp(ss, cum, pts[:, 1])
    return np.stack([x, y], 1), ss


def point_at(pts, cum, s):
    s = min(max(s, 0.0), cum[-1])
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    e = 2.0
    x2 = np.interp(min(s + e, cum[-1]), cum, pts[:, 0]) - np.interp(max(s - e, 0), cum, pts[:, 0])
    y2 = np.interp(min(s + e, cum[-1]), cum, pts[:, 1]) - np.interp(max(s - e, 0), cum, pts[:, 1])
    n = math.hypot(x2, y2) + 1e-9
    return np.array([x, y]), np.array([x2 / n, y2 / n])


def catmull(ctrl, n_per=60):
    ctrl = np.asarray(ctrl, np.float64)
    P = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    out = []
    u = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(ctrl[-1:])
    return np.vstack(out)


# ---------------------------------------------------------------- painter
@lru_cache(None)
def _kernel(r):
    R = int(math.ceil(r * 3.2))
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    return np.exp(-(x * x + y * y) / (2.0 * r * r)).astype(np.float32)


class Painter:
    """Collects additive light (bilinear splats) and flushes via bincount."""

    def __init__(self):
        self.idx, self.wr, self.wg, self.wb = [], [], [], []
        self.sprites = []

    def points(self, x, y, w, col):
        x = np.asarray(x, np.float32).ravel()
        y = np.asarray(y, np.float32).ravel()
        w = np.asarray(w, np.float32)
        w = w.ravel() if w.size == x.size else np.broadcast_to(w, x.shape).ravel()
        m = (x >= 0) & (x < W - 1.001) & (y >= 0) & (y < H - 1.001) & (w > 1e-6)
        if not m.any():
            return
        col = np.asarray(col, np.float32)
        x, y, w = x[m], y[m], w[m]
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx, fy = x - x0, y - y0
        i = y0.astype(np.int64) * W + x0.astype(np.int64)
        idx = np.concatenate([i, i + 1, i + W, i + W + 1])
        ww = np.concatenate([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
        if col.ndim == 1:
            self.wr.append(ww * col[0]); self.wg.append(ww * col[1]); self.wb.append(ww * col[2])
        else:
            c = np.tile(col[m], (4, 1))
            self.wr.append(ww * c[:, 0]); self.wg.append(ww * c[:, 1]); self.wb.append(ww * c[:, 2])
        self.idx.append(idx)

    def line(self, pts, col, inten=1.0, width=1.0, vw=None, dash=None, spacing=0.7):
        pts = np.asarray(pts, np.float64)
        if pts is None or len(pts) < 2 or inten <= 0:
            return
        # cull: skip lines fully outside the screen
        if pts[:, 0].max() < -5 or pts[:, 0].min() > W + 5 or pts[:, 1].max() < -5 or pts[:, 1].min() > H + 5:
            return
        cum = arclen(pts)
        Ltot = cum[-1]
        if Ltot <= 0.5:
            return
        n = int(Ltot / spacing) + 2
        s = np.linspace(0, Ltot, n)
        x = np.interp(s, cum, pts[:, 0])
        y = np.interp(s, cum, pts[:, 1])
        w = np.full(n, inten * Ltot / (n - 1), np.float64)
        if vw is not None:
            w *= np.interp(s, cum, np.asarray(vw, np.float64))
        if dash is not None:
            w *= ((s + dash[2] if len(dash) > 2 else s) % dash[0]) < dash[1]
        if width > 1.01:
            tx = np.gradient(x); ty = np.gradient(y)
            nn = np.hypot(tx, ty) + 1e-9
            nx, ny = -ty / nn, tx / nn
            k = int(math.ceil(width))
            offs = np.linspace(-(width - 1) / 2, (width - 1) / 2, k)
            wt = np.exp(-(offs / (0.5 * width + 0.3)) ** 2 * 1.2)
            wt = wt / wt.max()
            for o, f in zip(offs, wt):
                self.points(x + nx * o, y + ny * o, w * f, col)
        else:
            self.points(x, y, w, col)

    def sprite(self, x, y, r, col, inten):
        if inten > 1e-4 and -60 < x < W + 60 and -60 < y < H + 60:
            self.sprites.append((x, y, r, np.asarray(col, np.float32), inten))

    def flush(self):
        L = np.zeros((H * W, 3), np.float32)
        if self.idx:
            idx = np.concatenate(self.idx)
            for k, lst in enumerate((self.wr, self.wg, self.wb)):
                L[:, k] = np.bincount(idx, weights=np.concatenate(lst), minlength=H * W)
        L = L.reshape(H, W, 3)
        for (x, y, r, col, inten) in self.sprites:
            k = _kernel(round(r * 4) / 4.0)
            R = k.shape[0] // 2
            cx, cy = int(round(x)), int(round(y))
            x0, y0 = cx - R, cy - R
            xa, ya = max(x0, 0), max(y0, 0)
            xb, yb = min(x0 + k.shape[1], W), min(y0 + k.shape[0], H)
            if xa >= xb or ya >= yb:
                continue
            L[ya:yb, xa:xb] += k[ya - y0:yb - y0, xa - x0:xb - x0, None] * (col * inten)
        return L


# ---------------------------------------------------------------- glow
def _box(a, r, axis):
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode='edge')
    c = np.cumsum(p, axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a, r, it=2):
    for _ in range(it):
        a = _box(a, r, 0)
        a = _box(a, r, 1)
    return a


def down(a, f):
    h, w, c = a.shape
    return a.reshape(h // f, f, w // f, f, c).mean(axis=(1, 3))


def up(a, size=(W, H)):
    out = np.empty((size[1], size[0], 3), np.float32)
    for k in range(3):
        out[:, :, k] = np.asarray(Image.fromarray(np.ascontiguousarray(a[:, :, k]), 'F').resize(size, Image.BILINEAR))
    return out


def glow(L):
    d2 = down(L, 2)
    g1 = up(blur(d2, 2, 2))
    d8 = down(d2, 4)
    g2 = up(blur(d8, 3, 3))
    return g1, g2


# ---------------------------------------------------------------- background
class Sky:
    def __init__(self):
        rng = np.random.default_rng(7)
        FW, FH = W + 640, H + 400
        self.FW, self.FH = FW, FH
        sw, sh = FW // 8, FH // 8

        def noise(scale, seed):
            r = np.random.default_rng(seed)
            n = r.random((sh // scale + 3, sw // scale + 3)).astype(np.float32)
            img = Image.fromarray(n, 'F').resize((sw + 3 * scale, sh + 3 * scale), Image.BICUBIC)
            return np.asarray(img)[:sh, :sw]

        def fbm(seed):
            tot = np.zeros((sh, sw), np.float32)
            amp = 1.0
            for o, sc in enumerate([24, 12, 6, 3]):
                tot += amp * noise(sc, seed + o)
                amp *= 0.5
            tot /= 1.875
            return tot

        n1 = np.clip((fbm(11) - 0.42) * 2.4, 0, 1) ** 2
        n2 = np.clip((fbm(23) - 0.45) * 2.4, 0, 1) ** 2
        n3 = np.clip((fbm(37) - 0.52) * 3.0, 0, 1) ** 2
        yy = np.linspace(0, 1, sh)[:, None]
        base = np.zeros((sh, sw, 3), np.float32)
        base[:] = np.array([0.012, 0.016, 0.050])
        base += (yy * np.array([0.006, 0.006, 0.020]))[:, :, None].transpose(0, 2, 1) if False else 0
        base += yy[:, :, None] * np.array([0.004, 0.006, 0.018], np.float32)
        neb = (base + n1[:, :, None] * np.array([0.085, 0.035, 0.150], np.float32)
               + n2[:, :, None] * np.array([0.020, 0.050, 0.140], np.float32)
               + n3[:, :, None] * np.array([0.000, 0.070, 0.075], np.float32))
        neb = blur(neb.astype(np.float32), 2, 2) * 1.35
        self.neb = up(neb, (FW, FH))

        # star layers: (count, parallax, bright range)
        self.layers = []
        for cnt, par, bmin, bmax, seed in [(2400, 0.18, 0.04, 0.30, 1), (750, 0.42, 0.12, 0.65, 2), (130, 0.85, 0.35, 1.6, 3)]:
            r = np.random.default_rng(seed)
            pos = r.random((cnt, 2)) * np.array([W + 200, H + 200])
            b = bmin + (bmax - bmin) * r.random(cnt) ** 3
            hue = r.random(cnt)
            col = np.where(hue[:, None] < 0.70, BLUEW, np.where(hue[:, None] < 0.92, WARM, TEAL)).astype(np.float32)
            col = 0.55 * col + 0.45
            ph = r.random(cnt) * 6.283
            om = 0.6 + 2.2 * r.random(cnt)
            self.layers.append((pos, par, b, col, ph, om))

    @staticmethod
    def cam(t):
        # continuous global camera drift (used for parallax)
        x = 16.0 * t + 420.0 * smooth(12, 26, t) + 120 * smooth(58, 74, t)
        y = 30.0 * math.sin(t * 0.11) + 10 * math.sin(t * 0.37)
        return x, y

    def background(self, t):
        cx, cy = self.cam(t)
        ox = int(320 + cx * 0.10 - 140) % (self.FW - W)
        oy = int(200 + cy * 0.10)
        oy = min(max(oy, 0), self.FH - H)
        return self.neb[oy:oy + H, ox:ox + W]

    def stars(self, P, t):
        cx, cy = self.cam(t)
        for li, (pos, par, b, col, ph, om) in enumerate(self.layers):
            x = (pos[:, 0] - cx * par) % (W + 200) - 100
            y = (pos[:, 1] - cy * par) % (H + 200) - 100
            tw = 0.62 + 0.38 * np.sin(om * t + ph)
            w = b * tw
            P.points(x, y, w, col)
            if li == 2:
                top = np.argsort(-b)[:22]
                for i in top:
                    P.sprite(x[i], y[i], 1.6, col[i], w[i] * 0.55)


# ---------------------------------------------------------------- text
NOTO = '/usr/share/fonts/opentype/noto/'
DV = '/usr/share/fonts/truetype/dejavu/'
CHAINS = {
    'serif': [(NOTO + 'NotoSerifCJK-Regular.ttc', 2), (DV + 'DejaVuSerif.ttf', 0), (DV + 'DejaVuSans.ttf', 0)],
    'serifb': [(NOTO + 'NotoSerifCJK-Bold.ttc', 2), (DV + 'DejaVuSerif-Bold.ttf', 0), (DV + 'DejaVuSans-Bold.ttf', 0)],
    'sans': [(NOTO + 'NotoSansCJK-Regular.ttc', 2), (DV + 'DejaVuSans.ttf', 0)],
    'sansb': [(NOTO + 'NotoSansCJK-Bold.ttc', 2), (DV + 'DejaVuSans-Bold.ttf', 0)],
    # math: italic Latin, CJK and symbols fall back
    'math': [(DV + 'DejaVuSerif-Italic.ttf', 0), (DV + 'DejaVuSerif.ttf', 0), (DV + 'DejaVuSans.ttf', 0),
             (NOTO + 'NotoSerifCJK-Regular.ttc', 2)],
    'mathr': [(DV + 'DejaVuSerif.ttf', 0), (DV + 'DejaVuSans.ttf', 0), (NOTO + 'NotoSerifCJK-Regular.ttc', 2)],
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


def text_img(text, size, kind='serif', col=WARM, hcol=GOLD2, spacing=0.0):
    key = (text, size, kind, tuple(np.round(col, 3)), tuple(np.round(hcol, 3)), spacing)
    if key in _TC:
        return _TC[key]
    chars = parse(text, col, hcol)
    main = _font(*CHAINS[kind][0], size)
    asc, desc = main.getmetrics()
    pad = 6
    items, x = [], 0.0
    for ch, c in chars:
        p, i = _pick(kind, ch)
        f = _font(p, i, size)
        items.append((x, ch, f, c))
        x += f.getlength(ch) + spacing
    wid = int(math.ceil(x - spacing)) if items else 1
    hh = asc + desc
    imgw, imgh = wid + 2 * pad, hh + 2 * pad
    cols = {}
    for (xx, ch, f, c) in items:
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
    ti = dict(rgb=rgb, a=a, w=wid, h=hh, pad=pad, asc=asc)
    _TC[key] = ti
    return ti


def anchor_xy(ti, x, y, anchor):
    ax, ay = anchor[0], anchor[1]
    if ax == 'c':
        x -= ti['w'] / 2
    elif ax == 'r':
        x -= ti['w']
    if ay == 'm':
        y -= ti['h'] * 0.5
    elif ay == 'b':
        y -= ti['h']
    return x, y


def blit(dst, ti, x, y, alpha, add=False):
    if alpha <= 0.002:
        return
    rgb, a, pad = ti['rgb'], ti['a'], ti['pad']
    x0 = int(round(x)) - pad
    y0 = int(round(y)) - pad
    h, w = a.shape
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    sr = rgb[ya - y0:yb - y0, xa - x0:xb - x0]
    reg = dst[ya:yb, xa:xb]
    if add:
        reg += sr * alpha
    else:
        sa = a[ya - y0:yb - y0, xa - x0:xb - x0]
        reg *= (1 - sa[:, :, None] * alpha)
        reg += sr * alpha


class Overlay:
    def __init__(self):
        self.items = []

    def text(self, text, x, y, size, kind='serif', col=WARM, alpha=1.0, anchor='lt', glow=0.5,
             hcol=GOLD2, spacing=0.0):
        if alpha <= 0.002:
            return None
        ti = text_img(text, size, kind, col, hcol, spacing)
        x, y = anchor_xy(ti, x, y, anchor)
        self.items.append((ti, x, y, min(alpha, 1.0), glow))
        return (x, y, ti['w'], ti['h'])


# ---------------------------------------------------------------- HUD & subtitles
def draw_hud(img, P_unused, hud, alpha):
    if alpha <= 0.002 or hud is None:
        return
    title, en, sub = hud
    x0, y0 = 96, 62
    # vertical line
    yy = np.arange(y0 + 4, y0 + 128)
    img[yy, 74:76] = img[yy, 74:76] * (1 - 0.8 * alpha) + GOLD * 0.8 * alpha
    blit(img, text_img(title, 44, 'serif', WHITE), x0, y0 - 4, alpha)
    blit(img, text_img(en, 16, 'sans', GOLD, spacing=4.5), x0 + 2, y0 + 64, alpha * 0.95)
    blit(img, text_img(sub, 17, 'sans', np.array([0.70, 0.74, 0.82], np.float32)), x0 + 2, y0 + 94, alpha * 0.85)


def draw_subtitle(img, L, subs, t):
    for (a, b, txt) in subs:
        if a - 0.1 <= t <= b + 0.1:
            al = window(t, a, b, 0.45, 0.45)
            if al <= 0.002:
                continue
            ti = text_img(txt, 40, 'serif', np.array([0.94, 0.92, 0.88], np.float32), GOLD2, spacing=1.5)
            dy = (1 - ease_out((t - a) / 0.6)) * 8
            x, y = anchor_xy(ti, W / 2, 978 + dy, 'cm')
            if L is not None:
                blit(L, ti, x, y, al * 0.35, add=True)
            else:
                blit(img, ti, x, y, al)
