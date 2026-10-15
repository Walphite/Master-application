"""Timeline, camera and the seven acts of 《数学里最温柔的一个词》.

build(t) returns a core.Frame describing everything visible at time t: extruded
blocks in world space (one orthographic camera), trails, lines, cards, texts,
light sprites, light bands and impact strength. render.py turns it into pixels.

World: x to the right, y up, z towards the viewer. The Dirichlet graph lives on
x in [-400, 400] with value 0 at y = 0 (vermilion) and value 1 at y = ROW1
(mustard).
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import core
from core import (S, N_FRAG, FPS, COBALT, VERM, MUST, GREEN, CREAM, INK, Frame, Cam, smooth, window, ramp,
                  ease_io, ease_out, ease_in, ease_back, spring, mix, hash01, euler, clamp01, text_img,
                  formula_img, badge_img)

# ---------------------------------------------------------------- act boundaries
T2, T3, T4, T5, T6, T7, TEND = 4.0, 16.0, 25.6, 32.6, 51.4, 62.6, 72.0

LEVEL_COL = [GREEN, COBALT, VERM, MUST]
H_LEVEL = [0.25, 0.5, 0.75, 1.0]
MU_LEVEL = [0.4, 0.3, 0.2, 0.1]
LEVEL_LIFE = ['妈妈发来的 60 秒语音', '一年只见三次的朋友', '加班后楼下那碗面', '陌生人帮你按住的电梯门']
LEVEL_HM = ['0.25 × 0.4', '0.5 × 0.3', '0.75 × 0.2', '1 × 0.1']
DIM = np.array([0.62, 0.60, 0.58], np.float32)


# ---------------------------------------------------------------- camera
def track(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        k = keys[i]
        if t <= k[0]:
            t0, v0 = keys[i - 1][0], keys[i - 1][1]
            u = (t - t0) / (k[0] - t0) if k[0] > t0 else 1.0
            e = k[2] if len(k) > 2 else 'io'
            f = ease_io(u) if e == 'io' else ease_out(u) if e == 'out' else ease_in(u, 2) if e == 'in' else u
            return v0 + (k[1] - v0) * f
    return keys[-1][1]


ZOOM = [(0, 1.62), (1.9, 1.0, 'out'), (4.0, 1.04), (10.0, 1.12), (12.6, 1.12), (16.0, 1.0), (19.3, 1.06),
        (25.4, 1.13), (26.4, 1.02), (32.4, 1.07), (34.6, 0.94), (42.0, 0.98), (46.0, 0.97), (51.2, 0.90),
        (52.1, 1.0), (57.3, 1.07), (58.1, 0.97), (62.6, 1.0), (64.2, 1.0), (68.8, 1.05), (72.0, 0.97)]
YAW = [(0, -0.08), (2.6, -0.28), (16.0, -0.26), (25.6, -0.20), (32.6, -0.26), (36.0, -0.35), (51.4, -0.35),
       (52.1, -0.26), (57.3, -0.24), (58.1, -0.35), (62.6, -0.30), (64.0, -0.10), (72.0, -0.14)]
PITCH = [(0, 0.14), (2.6, 0.34), (16.0, 0.36), (25.6, 0.30), (32.6, 0.22), (36.0, 0.30), (42.0, 0.26),
         (51.4, 0.24), (52.1, 0.34), (57.3, 0.34), (58.1, 0.26), (62.6, 0.24), (64.0, 0.14), (72.0, 0.12)]
TX = [(0, 0.0), (51.4, 0.0), (52.1, -1500.0), (57.3, -1500.0), (58.1, 0.0)]
TY = [(0, 200.0), (32.6, 200.0), (34.6, 200.0), (51.4, 200.0), (52.1, 200.0), (57.3, 200.0), (58.1, 200.0),
      (62.6, 200.0), (64.0, 200.0), (68.8, 200.0), (70.0, 190.0)]
SX = [(0, 470.0), (32.6, 470.0), (34.6, 640.0), (51.4, 640.0), (52.1, 470.0), (57.3, 470.0), (58.1, 640.0),
      (62.6, 640.0), (63.8, 480.0)]
SY = [(0, 640.0), (32.6, 640.0), (34.6, 700.0), (51.4, 700.0), (52.1, 640.0), (57.3, 640.0), (58.1, 700.0),
      (62.6, 700.0), (63.8, 640.0), (68.8, 640.0), (70.0, 600.0)]
ROLL = [(0, 0.07), (1.9, 0.0, 'out'), (19.3, 0.0), (19.6, 0.02), (25.6, 0.0), (51.4, 0.0), (51.75, -0.035),
        (52.1, 0.0), (72.0, 0.0)]

IMPACTS = [(0.0, 1.0), (3.9, 0.3), (19.35, 0.45), (20.05, 1.0), (34.7, 0.7), (44.3, 0.9), (54.0, 0.45),
           (59.5, 0.25), (62.7, 0.4), (64.1, 0.8)]
PUNCHES = [(10.3, 0.10), (11.8, 0.12), (20.05, 0.08), (34.7, 0.06), (44.3, 0.08), (49.0, 0.10), (54.0, 0.06),
           (64.1, 0.08)]
BANDS = [(0.5, 1.0, 0.55), (7.8, 1.2, 0.38), (17.2, 1.2, 0.38), (26.3, 1.4, 0.32), (37.0, 1.3, 0.42),
         (44.5, 1.0, 0.5), (49.0, 1.0, 0.45), (55.0, 1.2, 0.32), (60.4, 1.3, 0.38), (64.5, 1.2, 0.62),
         (67.6, 1.4, 0.42)]


def impact(t):
    v = 0.0
    for t0, s in IMPACTS:
        if 0 <= t - t0 < 0.1:
            v = max(v, s * (1 - (t - t0) / 0.1 * 0.6))
    return v


def punch(t):
    v = 0.0
    for t0, a in PUNCHES:
        d = t - t0 + 0.12
        if 0 <= d < 0.12:
            v += a * ease_out(d / 0.12)
        elif d >= 0.12:
            v += a * spring(d - 0.12, 11.0, 6.0)
    return v


def orbit(t):
    # half turn around the tower; the tower is symmetric, so the yaw drops back by pi once it stops
    if 36.0 <= t < 42.6:
        return math.pi * ease_io(ramp(36.0, 42.0, t))
    return 0.0


def camera(t):
    z = track(ZOOM, t) * (1 + 0.012 * math.sin(0.6 * t + 0.4)) * (1 + punch(t))
    yaw = track(YAW, t) + 0.035 * math.sin(0.29 * t) + orbit(t)
    pitch = track(PITCH, t) + 0.012 * math.sin(0.37 * t + 1)
    tx = track(TX, t) + 9 * math.sin(0.23 * t)
    ty = track(TY, t) + 6 * math.sin(0.31 * t + 2)
    roll = track(ROLL, t) + 0.010 * math.sin(0.43 * t + 0.5)
    sx, sy = track(SX, t), track(SY, t)
    imp = impact(t)
    if imp > 0:
        fr = int(round(t * FPS))
        sx += (hash01(fr, 1) - 0.5) * 34 * imp
        sy += (hash01(fr, 2) - 0.5) * 34 * imp
        roll += (hash01(fr, 3) - 0.5) * 0.02 * imp
    return Cam(yaw, pitch, z, (tx, ty, 0.0), sx, sy, roll), imp


# ---------------------------------------------------------------- Dirichlet lattice
LAT_HW = 330.0
ROW1 = 400.0
LAT_ND = 5
LAT_DZ = 17.0
U0 = 0.55                       # log2 of the starting magnification (fractional part sets the density)
U_HOLD = 14.55                  # after the x100 and x10000 zooms (and a slow creep)
_NPOS = max(24, int(N_FRAG * 0.55 / LAT_ND))
LAT_SP = 2 * LAT_HW / _NPOS / 2 ** U0
CUBE = min(6.0, LAT_SP * 2 ** U0 * 0.44)
Y_FLOOR = -CUBE - 2.0
BAR_Y = ROW1 / 2


def _v2(k):
    k, z = abs(k), 0
    while k and k % 2 == 0:
        k //= 2
        z += 1
    return z


_LYC = {}


def lat_yellow(k, n):
    """Colour of the sample at index k on level n. It only depends on the reduced dyadic
    position, so zooming by 2 (k -> 2k, n -> n+1) keeps every point's colour."""
    key = (k, n)
    if key not in _LYC:
        if k == 0:
            a, b = 0, -999
        else:
            z = _v2(k)
            a, b = k // (1 << z), n - z
        _LYC[key] = hash01(a + 100000, b + 1000, 77) < 0.5
    return _LYC[key]


def lattice(u):
    n = math.floor(u)
    f = u - n
    step = LAT_SP * 2 ** f
    K = int(LAT_HW / step)
    ks = np.arange(-K, K + 1)
    x = ks * step
    grow = np.where(ks % 2 != 0, smooth(0.0, 0.4, f), 1.0)
    yel = np.array([lat_yellow(int(k), n) for k in ks])
    return ks, x, yel, grow, n


def lat_logmag(t):
    u = U0
    u += math.log2(100) * ease_io(ramp(10.3, 11.0, t))
    u += math.log2(100) * ease_io(ramp(11.8, 12.5, t))
    creep = (U_HOLD - (U0 + 2 * math.log2(100))) / 3.5
    u += creep * clamp01((t - 12.5) / 3.5) * 3.5
    return u


def cube_var(i, j=0):
    return 0.93 + 0.14 * hash01(i, j, 5)


def lattice_arrays(u, dx=0.0):
    ks, x, yel, grow, n = lattice(u)
    P, Hh, C, info = [], [], [], []
    for i, (k, xx, ye, g) in enumerate(zip(ks, x, yel, grow)):
        for d in range(LAT_ND):
            P.append((xx + dx, ROW1 if ye else 0.0, (d - 1) * LAT_DZ))
            Hh.append((CUBE * g, CUBE * g, CUBE * g))
            C.append((MUST if ye else VERM) * cube_var(int(k) * 7 + n, d))
            info.append((int(k), bool(ye), d))
    return np.array(P), np.array(Hh), np.array(C), info


def add_axes(F, dx=0.0, alpha=0.38, labels=True, a_lab=1.0):
    zf = LAT_DZ * 1.6 + 6
    F.back_lines.append((np.array([[-LAT_HW - 24 + dx, Y_FLOOR, zf], [LAT_HW + 30 + dx, Y_FLOOR, zf]]), 2.0,
                         CREAM, alpha, None))
    F.back_lines.append((np.array([[-LAT_HW - 24 + dx, Y_FLOOR, zf], [-LAT_HW - 24 + dx, ROW1 + 60, zf]]), 2.0,
                         CREAM, alpha, None))
    for yv in (0.0, ROW1):
        F.back_lines.append((np.array([[-LAT_HW - 30 + dx, yv, zf], [-LAT_HW - 18 + dx, yv, zf]]), 2.0,
                             CREAM, alpha, None))
    if labels and a_lab > 0.01:
        cam = F.cam
        for yv, s, col in ((0.0, '0', VERM), (ROW1, '1', MUST)):
            x, y = cam.screen((-LAT_HW - 48 + dx, yv, zf))
            F.text(text_img(s, 30, 'math', col), x, y, 'cm', a_lab)
        x, y = cam.screen((LAT_HW + 44 + dx, Y_FLOOR, zf))
        F.text(text_img('x', 28, 'math', CREAM), x, y, 'cm', 0.7 * a_lab)
        x, y = cam.screen((-LAT_HW - 24 + dx, ROW1 + 92, zf))
        F.text(text_img('D(x)', 28, 'math', CREAM), x, y, 'cm', 0.8 * a_lab)


# ---------------------------------------------------------------- Act 1: the bar shatters
def _s1_setup():
    rng = np.random.default_rng(101)
    P, Hh, C, info = lattice_arrays(U0)
    nL = len(P)
    n = max(N_FRAG, nL + 40)
    p0 = np.stack([rng.uniform(-LAT_HW, LAT_HW, n), BAR_Y + rng.uniform(-40, 40, n), rng.uniform(-26, 26, n)], 1)
    h0 = rng.uniform(4.5, 10.5, (n, 3))
    c = np.array([0.0, BAR_Y, 70.0])
    d = p0 - c + rng.normal(0, 25, (n, 3))
    d[:, 0] *= 1.3
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v = d * rng.uniform(260, 1150, (n, 1))
    v[:, 1] += rng.uniform(-60, 220, n)
    ang0 = rng.uniform(-0.25, 0.25, (n, 3))
    w = rng.uniform(-16, 16, (n, 3))
    sel = rng.choice(n, nL, replace=False)
    sel = sel[np.argsort(p0[sel, 0])]
    order = np.lexsort((P[:, 2], P[:, 0]))
    tgt = np.full(n, -1)
    tgt[sel] = order
    u = rng.random(n)
    land = np.zeros(n)
    m = tgt >= 0
    land[m] = 2.0 + 1.05 * (P[tgt[m], 0] + LAT_HW) / (2 * LAT_HW) + 0.2 * u[m]
    cv = 0.88 + 0.24 * rng.random(n)
    return dict(n=n, p0=p0, h0=h0, v=v, ang0=ang0, w=w, tgt=tgt, land=land, u=u, cv=cv, P=P, C=C)


S1 = _s1_setup()


def s1_shards(t):
    d = S1
    te = t + 0.05
    k = 2.3
    pos = d['p0'] + d['v'] * (1 - math.exp(-k * te)) / k
    ang = d['ang0'] + d['w'] * (1 - math.exp(-1.25 * te)) / 1.25
    half = d['h0'].copy()
    col = COBALT[None, :] * d['cv'][:, None]
    col = np.repeat(col, 1, 0)
    m = d['tgt'] >= 0
    q = np.clip((t - d['land']) / 0.55, 0, 1)
    qe = q * q * (3 - 2 * q)
    tg = d['P'][np.maximum(d['tgt'], 0)]
    pos[m] = pos[m] * (1 - qe[m, None]) + tg[m] * qe[m, None]
    ang[m] = ang[m] * (1 - qe[m, None])
    ang[m, 0] += math.pi * qe[m]
    half[m] = half[m] * (1 - qe[m, None]) + CUBE * qe[m, None]
    cf = np.clip((q - 0.35) / 0.3, 0, 1)
    tc = d['C'][np.maximum(d['tgt'], 0)]
    col[m] = col[m] * (1 - cf[m, None]) + tc[m] * cf[m, None]
    # squash on landing
    dt = t - (d['land'] + 0.55)
    b = np.where((dt > 0) & m, np.exp(-9 * np.maximum(dt, 0)) * np.cos(16 * np.maximum(dt, 0)), 0.0)
    half[:, 1] *= (1 - 0.45 * b)
    half[:, 0] *= (1 + 0.25 * b)
    half[:, 2] *= (1 + 0.25 * b)
    # spare shards drift away and shrink
    sm = ~m
    sc = 1 - np.clip((t - (0.9 + 1.4 * d['u'][sm])) / 0.9, 0, 1)
    half[sm] *= sc[:, None]
    R = euler(ang[:, 0], ang[:, 1], ang[:, 2])
    return pos, half, col, R, te


# ---------------------------------------------------------------- Act 2: cards
CARD_DEF = [  # text, anchor target world x, yellow?, card centre on screen, face colour, flip-in time
    ('早上七点四十的地铁', -210, True, (290, 355), COBALT, 6.9),
    ('被叫进办公室的五分钟', 100, False, (630, 945), VERM, 7.6),
    ('被夸了一句的下午', 210, True, (680, 290), GREEN, 8.8),
    ('打了一半又删掉的消息', -170, False, (300, 1020), COBALT, 9.4),
]


def _card_anchor(xw, yel):
    ks, x, ye, g, n = lattice(U0)
    best = None
    for k, xx, yy in zip(ks, x, ye):
        if bool(yy) == yel and (best is None or abs(xx - xw) < abs(best[1] - xw)):
            best = (int(k), xx)
    return best


CARD_ANCH = [_card_anchor(c[1], c[2]) for c in CARD_DEF]


def card_dict(text, x, y, face, back, flip, scale, alpha, size=28):
    return dict(text=text, x=x, y=y, face=face, back=back, flip=flip, scale=scale, alpha=alpha, size=size)


# ---------------------------------------------------------------- Act 3: Riemann columns
NCOL = 12
COL_X = -LAT_HW + (np.arange(NCOL) + 0.5) * (2 * LAT_HW / NCOL)
COL_HW = 2 * LAT_HW / NCOL * 0.40
COL_HD = 17.0
T_RISE0, COL_STAG, T_COLLAPSE = 16.25, 0.19, 19.35


def col_height(j, t):
    r = ease_back(ramp(T_RISE0 + j * COL_STAG, T_RISE0 + j * COL_STAG + 0.5, t), 1.4)
    om = 9.0 + 16.0 * ramp(17.6, 19.3, t)
    amp = 0.30 + 0.66 * ramp(17.0, 19.1, t)
    ph = om * t + j * 1.93 + 2.0 * math.sin(3.1 * t + j)
    return max(0.0, r) * ROW1 * (0.5 + 0.5 * amp * math.sin(ph)) + 2.0


def _simulate_debris():
    rng = np.random.default_rng(303)
    NP = 5
    pos, half, vel, w = [], [], [], []
    for j in range(NCOL):
        h = col_height(j, T_COLLAPSE)
        for p in range(NP):
            hy = h / NP / 2
            pos.append((COL_X[j], Y_FLOOR + h * (p + 0.5) / NP, rng.uniform(-3, 3)))
            half.append((COL_HW * rng.uniform(0.75, 1.0), max(hy, 4.0), COL_HD * rng.uniform(0.7, 1.0)))
            vel.append((rng.uniform(-170, 170), rng.uniform(-40, 260) * (p + 1) / NP, rng.uniform(-40, 40)))
            w.append(rng.uniform(-7, 7))
    pos, half, vel, w = (np.array(a, np.float64) for a in (pos, half, vel, w))
    rz = np.zeros(len(pos))
    dt = 1 / 240
    steps = int(7.0 / dt)
    rec_p, rec_r = [], []
    g = 2400.0
    for s in range(steps):
        vel[:, 1] -= g * dt
        vel[:, 0] *= (1 - 0.5 * dt)
        pos += vel * dt
        rz += w * dt
        ext = half[:, 1] * np.abs(np.cos(rz)) + half[:, 0] * np.abs(np.sin(rz))
        hit = pos[:, 1] - ext < Y_FLOOR
        pos[hit, 1] = Y_FLOOR + ext[hit]
        down = hit & (vel[:, 1] < 0)
        vel[down, 1] = -vel[down, 1] * 0.3
        vel[down, 0] *= 0.6
        w[down] *= 0.45
        slow = hit & (np.abs(vel[:, 1]) < 80)
        tgt = np.round(rz / (math.pi / 2)) * (math.pi / 2)
        rz[slow] += (tgt[slow] - rz[slow]) * min(1.0, 14 * dt)
        w[slow] *= 0.85
        vel[slow, 0] *= 0.9
        if s % 2 == 0:
            rec_p.append(pos.copy())
            rec_r.append(rz.copy())
    return dict(p=np.array(rec_p), r=np.array(rec_r), half=half)


DEBRIS = _simulate_debris()


def debris_at(t):
    i = int(np.clip((t - T_COLLAPSE) * 120, 0, len(DEBRIS['p']) - 1))
    return DEBRIS['p'][i], DEBRIS['r'][i], DEBRIS['half']


def _stamp():
    sz = int(round(104 * S))
    f = core._font(core.NOTO + 'NotoSerifCJK-Black.ttc', 2, sz)
    txt = '不可积'
    tw = int(f.getlength(txt))
    padx, pady = int(34 * S), int(22 * S)
    Wd, Hd = tw + 2 * padx + 20, int(sz * 1.35) + 2 * pady + 20
    im = Image.new('L', (Wd, Hd), 0)
    d = ImageDraw.Draw(im)
    bw = max(3, int(9 * S))
    d.rounded_rectangle([10, 10, Wd - 10, Hd - 10], radius=int(16 * S), outline=255, width=bw)
    d.rounded_rectangle([10 + 2.2 * bw, 10 + 2.2 * bw, Wd - 10 - 2.2 * bw, Hd - 10 - 2.2 * bw],
                        radius=int(9 * S), outline=255, width=max(1, bw // 3))
    d.text((Wd / 2, Hd / 2), txt, font=f, fill=255, anchor='mm')
    a = np.asarray(im, np.float32) / 255.0
    rng = np.random.default_rng(9)
    n = rng.random((Hd, Wd)).astype(np.float32)
    n = (n + np.roll(n, 1, 0) + np.roll(n, 1, 1)) / 3
    a = a * np.clip((n - 0.18) * 3.0, 0.25, 1.0)
    rgb = a[:, :, None] * VERM[None, None, :] * 1.05
    return dict(rgb=rgb.astype(np.float32), a=a.astype(np.float32), w=Wd, h=Hd, pad=0, asc=0, chars=[])


STAMP = None


def stamp():
    global STAMP
    if STAMP is None:
        STAMP = _stamp()
    return STAMP


# ---------------------------------------------------------------- Act 4-5: level sets, bills, tower
def _fs_setup():
    P, Hh, C, info = lattice_arrays(U_HOLD)
    ks = sorted(set(k for k, ye, d in info))
    rng = np.random.default_rng(55)
    perm = rng.permutation(len(ks))
    cuts = np.cumsum([0] + [int(round(m * len(ks))) for m in MU_LEVEL])
    cuts[-1] = len(ks)
    lev_of_k = {}
    for r, idx in enumerate(perm):
        lev = int(np.searchsorted(cuts, r, side='right') - 1)
        lev_of_k[ks[idx]] = min(lev, 3)
    lev = np.array([lev_of_k[k] for k, ye, d in info])
    n = len(P)
    u = rng.random(n)
    launch = 32.75 + 0.7 * u
    dur = 0.85 + 0.35 * rng.random(n)
    arrive = launch + dur
    slot = np.zeros(n, int)
    cnt = np.zeros(4, int)
    for lv in range(4):
        idx = np.where(lev == lv)[0]
        idx = idx[np.argsort(arrive[idx])]
        slot[idx] = np.arange(len(idx))
        cnt[lv] = len(idx)
    return dict(P=P, C=C, info=info, lev=lev, n=n, u=u, launch=launch, arrive=arrive, slot=slot, cnt=cnt,
                kick=rng.normal(0, 1, (n, 3)), zr=rng.uniform(-0.6, 0.6, n), spin=rng.uniform(-9, 9, (n, 3)))


FS = _fs_setup()

TS, WS = 150.0, 760.0
SLAB_T = [h * TS for h in H_LEVEL]
SLAB_W = [m * WS for m in MU_LEVEL]
SLAB_D = [max(56.0, w * 0.72) for w in SLAB_W]
BASE_Y = -40.0
SPREAD = 78.0
_yc, _acc = [], BASE_Y
for _t in SLAB_T:
    _yc.append(_acc + _t / 2)
    _acc += _t
SLAB_YC = np.array(_yc)
SLAB_YS = SLAB_YC + SPREAD * np.arange(4)
TOWER_TOP = BASE_Y + sum(SLAB_T)


def slab_yc(t):
    q = ease_in(ramp(43.8, 44.3, t), 2)
    return SLAB_YS * (1 - q) + SLAB_YC * q


def slab_fill(lv, t):
    m = FS['lev'] == lv
    return float(np.clip((t - FS['arrive'][m] + 0.06) / 0.06, 0, 1).sum()) / max(1, FS['cnt'][lv])


def slab_target(i):
    lv = FS['lev'][i]
    j = FS['slot'][i]
    x = -SLAB_W[lv] / 2 + (j + 0.5) / FS['cnt'][lv] * SLAB_W[lv]
    z = (hash01(i, 8) - 0.5) * SLAB_D[lv] * 0.6
    return np.array([x, SLAB_YS[lv], z])


FS_TGT = np.array([slab_target(i) for i in range(FS['n'])])

BILL_DEN = [10] * 7 + [20] * 5 + [50] * 4 + [100] * 3
BILL_COL = {10: COBALT, 20: MUST, 50: GREEN, 100: VERM}
BILL_STACK_X = {10: -255.0, 20: -85.0, 50: 85.0, 100: 255.0}


def _bills():
    rng = np.random.default_rng(77)
    n = len(BILL_DEN)
    order = rng.permutation(n)
    den = [BILL_DEN[i] for i in order]
    p0 = np.stack([rng.uniform(-330, 330, n), rng.uniform(40, 300, n), 150 + rng.uniform(-10, 10, n)], 1)
    r0 = rng.uniform(-0.7, 0.7, n)
    slot = {}
    p1, r1 = [], []
    for d in den:
        s = slot.get(d, 0)
        slot[d] = s + 1
        p1.append((BILL_STACK_X[d] + rng.uniform(-3, 3), 30 + s * 10.0, 150 + s * 0.5))
        r1.append(rng.uniform(-0.05, 0.05))
    return dict(den=den, p0=p0, r0=r0, p1=np.array(p1), r1=np.array(r1), pop=28.5 + 0.4 * rng.random(n),
                go=29.45 + 0.6 * rng.random(n), counts=slot)


BILLS = _bills()


# ---------------------------------------------------------------- Act 6: countable red points
U6 = U0
LAT2_X = -1500.0


def _red_setup():
    ks, x, yel, grow, n = lattice(U6)
    reds = [(int(k), xx) for k, xx, ye in zip(ks, x, yel) if not ye]
    # enumerate like rationals: coarse dyadic levels first, then by numerator
    reds.sort(key=lambda kx: (-_v2(kx[0]) if kx[0] else -99, kx[0] // (1 << _v2(kx[0])) if kx[0] else 0))
    m = len(reds)
    times = {k: 51.95 + 5.1 * (j / max(m - 1, 1)) ** 0.62 for j, (k, xx) in enumerate(reds)}
    order = {k: j for j, (k, xx) in enumerate(reds)}
    return dict(times=times, order=order, xs={k: xx for k, xx in reds}, m=m)


RED = _red_setup()


# ---------------------------------------------------------------- Act 7: mosaic
def _mosaic():
    f = core._font(core.NOTO + 'NotoSansCJK-Black.ttc', 2, 300)
    txt = '几乎处处'
    im = Image.new('L', (1400, 420), 0)
    ImageDraw.Draw(im).text((20, 20), txt, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.where(a > 0.5)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hb, wb = a.shape
    width = 800.0
    best = None
    for c in np.arange(9.0, 26.0, 0.5):
        nc, nr = int(width / c), int(width / c * hb / wb)
        cells = []
        for r in range(nr):
            for q in range(nc):
                y0, y1 = int(r * hb / nr), int((r + 1) * hb / nr)
                x0, x1 = int(q * wb / nc), int((q + 1) * wb / nc)
                if a[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].mean() > 0.42:
                    cells.append((q, r))
        if best is None or abs(len(cells) - N_FRAG * 0.95) < abs(len(best[1]) - N_FRAG * 0.95):
            best = (c, cells, nc, nr)
    c, cells, nc, nr = best
    rng = np.random.default_rng(12)
    P = np.array([(-width / 2 + (q + 0.5) * c, 200 + (nr / 2 - r - 0.5) * c, 0.0) for q, r in cells])
    n = len(P)
    # sources: points inside the compact tower, weighted by slab volume
    vol = np.array([SLAB_W[i] * SLAB_T[i] * SLAB_D[i] for i in range(4)])
    lev = rng.choice(4, n, p=vol / vol.sum())
    src = np.stack([(rng.random(n) - 0.5) * np.array(SLAB_W)[lev], SLAB_YC[lev] + (rng.random(n) - 0.5) *
                    np.array(SLAB_T)[lev], (rng.random(n) - 0.5) * np.array(SLAB_D)[lev]], 1)
    # assign: sort both by x so that flights cross less
    so = np.lexsort((src[:, 1], src[:, 0]))
    to = np.lexsort((-P[:, 1], P[:, 0]))
    tgt = np.empty_like(P)
    tgt[so] = P[to]
    go = 62.8 + 0.55 * (tgt[:, 0] + 400) / 800 + 0.3 * rng.random(n)
    zero = rng.choice(n, 7, replace=False)
    return dict(P=tgt, src=src, lev=lev, go=go, arrive=go + 0.75 + 0.25 * rng.random(n), c=c, n=n, nr=nr,
                zero=set(int(z) for z in zero), kick=rng.normal(0, 1, (n, 3)), spin=rng.uniform(-8, 8, (n, 3)))


MOS = _mosaic()


# ---------------------------------------------------------------- ambient debris and foreground
def _ambient():
    rng = np.random.default_rng(5)
    n = 70
    p = np.stack([rng.uniform(-900, 900, n), rng.uniform(-750, 1150, n), rng.uniform(-200, 200, n)], 1)
    return dict(p=p, h=rng.uniform(3, 9, n), col=rng.choice(4, n), w=rng.uniform(-0.6, 0.6, (n, 3)),
                ph=rng.uniform(0, 6.28, n), dr=rng.normal(0, 6, (n, 3)))


AMB = _ambient()


def _fg():
    rng = np.random.default_rng(8)
    items = []
    for i in range(6):
        side = -1 if i % 2 == 0 else 1
        items.append(dict(x=side * rng.uniform(620, 720), y=rng.uniform(-1000, 1000), vy=rng.uniform(-28, 28),
                          vx=-side * rng.uniform(0, 3), h=rng.uniform(28, 58), col=int(rng.integers(4)),
                          w=rng.uniform(-0.4, 0.4, 3), a0=rng.uniform(0, 6.28, 3)))
    return items


FGI = _fg()
PALETTE = [COBALT, VERM, MUST, GREEN]


def add_ambient(F, t):
    a = AMB
    p = a['p'] + a['dr'] * t + np.stack([np.zeros(len(a['p'])), 10 * np.sin(0.5 * t + a['ph']),
                                         np.zeros(len(a['p']))], 1)
    ang = a['w'] * t
    cols = np.array([PALETTE[c] for c in a['col']]) * 0.42
    F.add_blocks(p, np.repeat(a['h'][:, None], 3, 1), cols, euler(ang[:, 0], ang[:, 1], ang[:, 2]), layer='bg')


def fg_blocks(t, cam):
    """Large out-of-focus cubes for the foreground layer (own camera with strong parallax)."""
    C, Hh, col, ang = [], [], [], []
    for i, it in enumerate(FGI):
        y = (it['y'] + it['vy'] * t + 1100) % 2200 - 1100
        x = it['x'] + it['vx'] * t - (cam.target[0]) * 0.35 - (cam.sx - 470) * 0.25
        C.append((x, y - (cam.target[1] - 160) * 1.2, 0.0))
        Hh.append((it['h'],) * 3)
        col.append(PALETTE[it['col']] * 0.85)
        ang.append(it['a0'] + it['w'] * t)
    ang = np.array(ang)
    return dict(c=np.array(C), h=np.array(Hh), col=np.array(col), R=euler(ang[:, 0], ang[:, 1], ang[:, 2]), flat=False)


# ---------------------------------------------------------------- HUD tags
HUDS = [
    (16.0, 25.3, '黎曼积分 · RIEMANN · 1854', COBALT),
    (25.8, 32.4, '勒贝格 · LEBESGUE · 1902', GREEN),
]


def bezier(p0, p1, p2, p3, s):
    s = s[:, None]
    return ((1 - s) ** 3) * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * p3


# ================================================================= build
def build(t):
    F = Frame(t)
    cam, imp = camera(t)
    F.cam = cam
    F.impact = imp
    add_ambient(F, t)
    F.fg = fg_blocks(t, cam)
    for t0, dur, st in BANDS:
        if t0 <= t <= t0 + dur:
            F.bands.append((-0.2 + 1.4 * (t - t0) / dur, st * max(0.0, math.sin(math.pi * (t - t0) / dur)) ** 0.7))
    for (a, b, txt, col) in HUDS:
        if a - 0.1 <= t <= b + 0.1:
            F.hud.append((txt, col, t - a, b - t))

    if t < T2 + 0.05:
        act1(F, t)
    if T2 - 0.05 <= t < T4 + 1.0:
        act2_3_lattice(F, t)
    if T2 <= t < T3 + 0.2:
        act2_overlays(F, t)
    if T3 - 0.1 <= t < T4 + 1.0:
        act3(F, t)
    if T4 - 0.2 <= t < 36.0:
        act4(F, t)
    if T5 <= t < T7 + 0.4:
        act5_tower(F, t)
    if T6 <= t < T7:
        act6(F, t)
    if t >= T7 - 0.05:
        act7(F, t)
    return F


# ---------------------------------------------------------------- act 1
def act1(F, t):
    pos, half, col, R, te = s1_shards(t)
    F.add_blocks(pos, half, col, R)
    # streaks right after the blast
    if t < 0.9:
        d = S1
        k = 2.3
        a = 1 - t / 0.9
        prev = d['p0'] + d['v'] * (1 - math.exp(-k * max(te - 0.09, 0))) / k
        for i in range(0, d['n'], 3):
            F.trails.append((np.stack([prev[i], pos[i]]), 5.0, COBALT * 1.25 + 0.1, 0.45 * a))
    if t < 0.45:
        x, y = F.cam.screen((0, BAR_Y, 60))
        e = 1 - t / 0.45
        F.sprites.append((x, y, 70, np.array([0.55, 0.65, 1.0]), 0.9 * e * e))
        F.sprites.append((x, y, 22, np.array([1.0, 1.0, 1.0]), 1.6 * e ** 3))


# ---------------------------------------------------------------- act 2-3 lattice (and 4 hover until it hands over)
def act2_3_lattice(F, t):
    if t >= T4 - 0.2:
        return  # act4 takes over the fragments
    u = lat_logmag(t)
    P, Hh, C, info = lattice_arrays(u)
    mult = 1.0 - 0.52 * smooth(16.1, 16.8, t) * (1 - smooth(T4 - 0.6, T4, t))
    F.add_blocks(P, Hh, C * mult)
    add_axes(F, alpha=0.38 * (1 - 0.4 * smooth(16, 17, t)), a_lab=smooth(4.2, 5.0, t) if t < 10 else 1.0)


def act2_overlays(F, t):
    cam = F.cam
    # definition of D(x)
    a = window(t, 4.4, 6.75, 0.5, 0.4)
    if a > 0:
        F.text(text_img('D(x) = 1，x 为无理数', 34, 'math', MUST), 480, 268, 'cm', a)
        F.text(text_img('D(x) = 0，x 为有理数', 34, 'math', VERM), 480, 318, 'cm', a * smooth(4.7, 5.2, t))
    # magnification readout
    u = lat_logmag(t)
    mag = 2 ** (u - U0)
    al = window(t, 10.15, 14.2, 0.2, 0.5)
    if al > 0:
        mval = min(mag, 10000.0)
        s = '×{}'.format(int(round(mval)))
        pop = 1 + 0.25 * max(spring(t - 11.0, 14, 8), spring(t - 12.5, 14, 8), 0) if t > 11 else 1
        F.texts.append(('scaled', text_img(s, 44, 'mathb', CREAM), 820, 250, al, pop))
    # scene cards flipped out of single points
    for ci, (txt, xw, yel, (cx, cy), face, t0) in enumerate(CARD_DEF):
        if t < t0:
            continue
        k, x0 = CARD_ANCH[ci]
        mg = 2 ** (lat_logmag(t) - U0)
        aw = np.array([x0 * mg, ROW1 if yel else 0.0, LAT_DZ])
        ax, ay = cam.screen(aw)
        a0x, a0y = cam.screen(np.array([x0, ROW1 if yel else 0.0, LAT_DZ]))
        q = ramp(t0, t0 + 0.55, t)
        e = ease_back(q, 1.3)
        # follow the anchor when the axis is magnified: cards are flung outwards
        tx, ty = cx + (ax - a0x) * 0.9, cy + (ay - a0y)
        px, py = a0x + (tx - a0x) * e, a0y + (ty - a0y) * e
        flip = math.cos(math.pi * (1 - ease_out(q, 2)))
        sc = 0.25 + 0.75 * e
        out = smooth(14.8, 15.6, t)
        F.cards.append(card_dict(txt, px, py, face, MUST if yel else VERM, flip, sc * (1 - 0.3 * out), 1 - out))
        if q > 0.05:
            F.front_lines.append((np.array([[a0x + (ax - a0x), a0y + (ay - a0y)], [px, py]]), 1.5, CREAM,
                                  0.45 * min(1, q * 2) * (1 - out), 'screen'))
            F.sprites.append((a0x + (ax - a0x), a0y + (ay - a0y), 9, MUST if yel else VERM,
                              0.9 * (1 - q) + 0.25))


# ---------------------------------------------------------------- act 3
def act3(F, t):
    cam = F.cam
    if t < T_COLLAPSE:
        C, Hh, col = [], [], []
        for j in range(NCOL):
            h = col_height(j, t)
            if h <= 2.5:
                continue
            dx = math.sin(43 * t + j * 2.1) * 7 * ramp(18.2, 19.3, t)
            dy = math.sin(51 * t + j) * 3 * ramp(18.6, 19.3, t)
            C.append((COL_X[j] + dx, Y_FLOOR + h / 2 + dy, 0.0))
            Hh.append((COL_HW, h / 2, COL_HD))
            col.append(COBALT * (0.95 + 0.1 * hash01(j, 4)))
        if C:
            F.add_blocks(C, Hh, col)
    elif t < T4 + 1.0:
        p, rz, half = debris_at(t)
        sc = 1 - smooth(T4 + 0.0, T4 + 0.8, t)
        if sc > 0.01:
            R = euler(np.zeros(len(rz)), np.zeros(len(rz)), rz)
            F.add_blocks(p, half * sc, COBALT[None, :] * (0.9 + 0.12 * np.array([hash01(i, 6) for i in range(len(p))]))[:, None]
                         * (1 - 0.35 * smooth(20.5, 21.5, t)), R)
        if t < T_COLLAPSE + 0.4:
            for j in range(NCOL):
                x, y = cam.screen((COL_X[j], ROW1 * 0.5, 0))
                F.sprites.append((x, y, 26, COBALT * 1.4 + 0.2, 0.9 * (1 - (t - T_COLLAPSE) / 0.4)))
    # month labels
    for j in range(NCOL):
        a = smooth(T_RISE0 + j * COL_STAG, T_RISE0 + j * COL_STAG + 0.3, t) * (1 - smooth(T4 - 0.3, T4 + 0.3, t))
        if a <= 0:
            continue
        x, y = cam.screen((COL_X[j], Y_FLOOR - 34, LAT_DZ * 1.6 + 6))
        th = 20.45 + j * 0.14
        hl = smooth(th, th + 0.12, t) * (1 - 0.5 * smooth(22.7, 23.4, t))
        pop = 1 + 0.35 * spring(t - th, 14, 9) if t > th else 1
        F.texts.append(('scaled', text_img(f'{j + 1}月', 22, 'sansm', CREAM * 0.75), x, y, a * (1 - hl), pop))
        if hl > 0:
            F.texts.append(('scaled', text_img(f'{j + 1}月', 22, 'sansb', MUST), x, y, a * hl, pop))
    # the stamp
    if 19.85 <= t <= T4 + 0.2:
        q = ramp(19.85, 20.05, t)
        sc = mix(2.7, 1.0, ease_in(q, 2))
        sq = spring(t - 20.05, 20, 10) if t > 20.05 else 0
        al = smooth(19.85, 19.93, t) * (1 - smooth(T4 - 0.4, T4 + 0.1, t))
        F.images.append((stamp(), 470, 640, al, sc * (1 - 0.08 * sq), 1 + 0.10 * sq, math.radians(-9)))
        if 20.05 <= t < 20.6:
            e = (t - 20.05) / 0.55
            for i in range(16):
                ang = i / 16 * 6.283 + 0.3
                r = 120 + 200 * ease_out(e)
                F.sprites.append((470 + math.cos(ang) * r * 1.2, 640 + math.sin(ang) * r * 0.6, 6,
                                  VERM * 1.2 + 0.1, 0.8 * (1 - e)))


# ---------------------------------------------------------------- act 4
def act4(F, t):
    cam = F.cam
    d = FS
    n = d['n']
    if t < 35.0:
        P = d['P'].copy()
        bob = ramp(T4 - 0.2, T4 + 0.8, t)
        ph = np.array([hash01(i, 3) * 6.28 for i in range(n)])
        P[:, 1] += bob * 7 * np.sin(1.3 * t + ph)
        P[:, 2] += bob * 4 * np.sin(0.9 * t + ph * 1.7)
        q = np.clip((t - (26.9 + 0.55 * d['u'])) / 1.0, 0, 1)
        qe = q * q * (3 - 2 * q)
        ylev = np.array(H_LEVEL)[d['lev']] * ROW1
        P[:, 1] = P[:, 1] * (1 - qe) + (ylev + bob * 7 * np.sin(1.3 * t + ph)) * qe
        lc = np.array([LEVEL_COL[l] for l in d['lev']]) * np.array([cube_var(i, 9) for i in range(n)])[:, None]
        cf = np.clip((qe - 0.35) / 0.3, 0, 1)[:, None]
        C = d['C'] * (1 - cf) + lc * cf
        mult = (1 - 0.52 * (1 - smooth(T4 - 0.6, T4 + 0.2, t))) * (1 - 0.5 * window(t, 28.5, 31.6, 0.3, 0.5))
        ang = np.zeros((n, 3))
        ang[:, 0] = math.pi * qe
        if t < T5:
            F.add_blocks(P, np.full((n, 3), CUBE), C * mult, euler(ang[:, 0], ang[:, 1], ang[:, 2]))
        F._fs_pos = P
        F._fs_col = C
    if t < T5 + 0.6:
        add_axes(F, alpha=0.38 * (1 - smooth(T5, T5 + 0.6, t)), a_lab=1 - smooth(T5, T5 + 0.5, t))
    # cut lines: vertical -> horizontal
    a = smooth(25.9, 26.4, t) * (1 - smooth(32.9, 33.5, t))
    if a > 0:
        rot = ease_io(ramp(27.0, 27.9, t))
        for i, xc in enumerate([-300, -150, 0, 150, 300]):
            if i < 4:
                yc = mix(170, H_LEVEL[i] * ROW1, rot)
                xcc = mix(xc, 0, rot)
                ln = mix(190, 430, rot)
                al = a
            else:
                yc, xcc, ln, al = 170, xc, 190, a * (1 - rot)
            ang = mix(math.pi / 2, 0, rot)
            dxv, dyv = math.cos(ang) * ln, math.sin(ang) * ln
            col = mix(COBALT * 1.3 + 0.08, MUST, smooth(27.6, 28.3, t)) if i < 4 else COBALT * 1.3
            F.front_lines.append((np.array([[xcc - dxv, yc - dyv, LAT_DZ * 2], [xcc + dxv, yc + dyv, LAT_DZ * 2]]),
                                  3.0, col, 0.85 * al, (14, 9)))
    # bills
    b = BILLS
    if 28.4 <= t <= 32.0:
        C, Hh, R, col = [], [], [], []
        for i, den in enumerate(b['den']):
            sc = ease_back(ramp(b['pop'][i], b['pop'][i] + 0.3, t), 2.0)
            q = ease_io(ramp(b['go'][i], b['go'][i] + 0.6, t))
            p = b['p0'][i] * (1 - q) + b['p1'][i] * q
            p = p + np.array([0, 40 * math.sin(math.pi * q), 0])
            rz = b['r0'][i] * (1 - q) + b['r1'][i] * q
            out = ramp(31.15 + 0.03 * i, 31.6 + 0.03 * i, t)
            sc *= 1 - ease_in(out, 2)
            if sc <= 0.01:
                continue
            C.append(p)
            Hh.append((50 * sc, 23 * sc, 3.5 * sc))
            R.append((0.0, 0.0, rz))
            col.append(BILL_COL[den] * (0.95 + 0.1 * hash01(i, 2)))
            if 0.6 < out < 1:
                x, y = cam.screen(p)
                F.sprites.append((x, y, 10, BILL_COL[den] * 1.3 + 0.2, 1.2 * (1 - out)))
        if C:
            R = np.array(R)
            F.add_blocks(C, Hh, col, euler(R[:, 0], R[:, 1], R[:, 2]))
        la = smooth(30.15, 30.45, t) * (1 - smooth(31.1, 31.4, t))
        if la > 0:
            for den, x0 in BILL_STACK_X.items():
                cnt = b['counts'][den]
                x, y = cam.screen((x0, 30 + (cnt - 1) * 10.0, 154))
                F.text(text_img(str(den), 28, 'mathb', CREAM), x, y, 'cm', la)
                x, y = cam.screen((x0, -22, 150))
                F.text(text_img(f'× {cnt}', 26, 'math', CREAM * 0.8), x, y, 'cm', la)


# ---------------------------------------------------------------- act 5 (and the tower in acts 6-7)
def act5_tower(F, t):
    cam = F.cam
    d = FS
    n = d['n']
    yc = slab_yc(t)
    # flying fragments
    if t < 35.0:
        P0 = getattr(F, '_fs_pos', None)
        C0 = getattr(F, '_fs_col', None)
        if P0 is None:
            ylev = np.array(H_LEVEL)[d['lev']] * ROW1
            P0 = d['P'].copy()
            P0[:, 1] = ylev
            C0 = np.array([LEVEL_COL[l] for l in d['lev']])
        s = np.clip((t - d['launch']) / (d['arrive'] - d['launch']), 0, 1)
        se = s * s * (3 - 2 * s)
        live = s < 1
        p1 = P0 + d['kick'] * np.array([60, 50, 60]) + np.array([0, 160, 40])
        tg = FS_TGT
        p2 = tg + np.array([0, 140, 0])
        P = bezier(P0, p1, p2, tg, se)
        ang = d['spin'] * np.sin(np.pi * se)[:, None]
        if live.any():
            idx = np.where(live)[0]
            F.add_blocks(P[idx], np.full((len(idx), 3), CUBE), C0[idx],
                         euler(ang[idx, 0], ang[idx, 1], ang[idx, 2]))
            moving = idx[(s[idx] > 0.02)]
            for i in moving[::1]:
                ss = np.linspace(max(se[i] - 0.22, 0), se[i], 7)
                pts = bezier(P0[i], p1[i], p2[i], tg[i], ss)
                F.trails.append((pts, 7.0, C0[i] * 1.2 + 0.05, 0.55))
    # slabs
    if t >= T7 + 0.15:
        return
    C, Hh, col = [], [], []
    for lv in range(4):
        fill = slab_fill(lv, t)
        if fill <= 0.001:
            continue
        w = SLAB_W[lv] * fill
        sq = spring(t - 44.3, 18, 8) if t > 44.3 else 0.0
        sq2 = spring(t - 34.7, 16, 8) if t > 34.7 else 0.0
        sy = 1 - 0.16 * sq - 0.1 * sq2
        sxz = 1 + 0.06 * sq + 0.04 * sq2
        C.append((-SLAB_W[lv] / 2 + w / 2, yc[lv] - (1 - sy) * SLAB_T[lv] / 2, 0.0))
        Hh.append((w / 2 * sxz, SLAB_T[lv] / 2 * sy, SLAB_D[lv] / 2 * sxz))
        glow = 1.0 + 0.25 * window(t, 35.6 + 1.35 * lv if lv < 3 else 40.7, 43.0, 0.2, 0.8)
        col.append(LEVEL_COL[lv] * glow)
    if C:
        F.add_blocks(C, Hh, col)
    if t < T6 - 0.1:
        tower_labels(F, t, yc)
    else:
        tower_labels(F, t, yc, fade=1 - smooth(T6 - 0.1, T6 + 0.3, t))
    # height ruler
    ra = smooth(46.2, 46.6, t) * (1 - smooth(T7 - 0.2, T7 + 0.2, t))
    if ra > 0:
        g = ease_io(ramp(46.3, 47.6, t))
        x0 = SLAB_W[0] / 2 * 1.0 + 70
        top = BASE_Y + (TOWER_TOP - BASE_Y) * g
        zr = SLAB_D[0] / 2
        F.front_lines.append((np.array([[x0, BASE_Y, zr], [x0, top, zr]]), 3.0, MUST, ra, None))
        F.front_lines.append((np.array([[x0 - 14, BASE_Y, zr], [x0 + 14, BASE_Y, zr]]), 3.0, MUST, ra, None))
        if g > 0.98:
            F.front_lines.append((np.array([[x0 - 14, TOWER_TOP, zr], [x0 + 14, TOWER_TOP, zr]]), 3.0, MUST, ra, None))
            F.front_lines.append((np.array([[-SLAB_W[3] / 2, TOWER_TOP, zr * 0.3], [x0, TOWER_TOP, zr]]), 1.5,
                                  CREAM, 0.45 * ra, (8, 8)))
        x, y = cam.screen((x0, top, zr))
        F.sprites.append((x, y, 8, MUST * 1.2 + 0.2, 1.4 * ra))
        F.sprites.append((x, y, 22, MUST, 0.35 * ra * (1 + window(t, 48.8, 50.2, 0.2, 0.6))))
    # formula
    fa = smooth(44.35, 45.0, t) * (1 - smooth(T6 - 0.2, T6 + 0.2, t))
    if fa > 0:
        fi = formula_img(['∫ f dμ = ', ('lim', '∫', '0', '∞'), ' μ({f > t}) dt'], 46, CREAM)
        pop = 1 + 0.06 * spring(t - 44.35, 10, 6)
        F.texts.append(('scaled', fi, 480, 236, fa, pop))


def tower_labels(F, t, yc, fade=1.0):
    cam = F.cam
    times = [35.6, 37.7, 39.1, 40.7]
    ha = smooth(35.0, 35.6, t) * fade
    for lv in range(4):
        # life label (scene card) on the left, height x measure under it
        a = smooth(times[lv], times[lv] + 0.35, t) * fade
        hm = smooth(35.0 + 0.15 * lv, 35.5 + 0.15 * lv, t) * fade
        ys = cam.screen((0, SLAB_YS[lv], 0))[1]
        y_now = cam.screen((0, yc[lv], 0))[1]
        ly = ys - 6
        lx = 200
        if a > 0:
            e = ease_back(ramp(times[lv], times[lv] + 0.45, t), 1.6)
            F.badges.append((badge_img(LEVEL_LIFE[lv], 22, LEVEL_COL[lv] * 0.92, 'sansb', INK if lv == 3 else CREAM, depth=6, radius=10,
                                       padx=14, pady=8), lx, ly, a, 0.6 + 0.4 * e))
        if hm > 0:
            F.text(text_img(LEVEL_HM[lv], 24, 'math', CREAM * 0.85), lx, ly + 40, 'cm', hm)
        if max(a, hm) > 0:
            ex, ey = cam.screen((-SLAB_W[lv] / 2 * 0.9, yc[lv], SLAB_D[lv] / 2))
            F.front_lines.append((np.array([[lx + 150, ly + 8], [ex - 6, y_now]]), 1.5, CREAM,
                                  0.4 * max(a, hm), 'screen'))
    if ha > 0:
        y0 = cam.screen((0, SLAB_YS[3], 0))[1] - 6 - 56
        F.text(text_img('高度 × 测度', 22, 'sansm', CREAM * 0.7), 200, y0, 'cm', ha)


# ---------------------------------------------------------------- act 6
def act6(F, t):
    cam = F.cam
    ks, x, yel, grow, n = lattice(U6)
    P, Hh, C, info = [], [], [], []
    for k, xx, ye, g in zip(ks, x, yel, grow):
        k = int(k)
        sc, lift = 1.0, 0.0
        if not ye:
            tc = RED['times'][k]
            q = ramp(tc, tc + 0.35, t)
            sc = 1 - ease_in(q, 2)
            lift = 26 * ease_out(q)
            if 0 < q < 1 or (q >= 1 and t - tc < 0.7):
                sx, sy = cam.screen((xx + LAT2_X, lift, LAT_DZ))
                fl = 1 - ramp(tc + 0.3, tc + 0.7, t)
                F.sprites.append((sx, sy, 5, VERM * 1.3 + 0.35, 1.8 * fl * min(1, q * 4)))
                F.sprites.append((sx, sy, 16, VERM, 0.35 * fl * min(1, q * 4)))
            na = window(t, tc, tc + 0.9, 0.08, 0.5)
            if na > 0:
                sx, sy = cam.screen((xx + LAT2_X, 70, LAT_DZ))
                F.texts.append(('scaled', text_img(str(RED['order'][k] + 1), 20, 'mathb', VERM * 1.15),
                                sx, sy - 14 * ramp(tc, tc + 0.9, t), na, 1 + 0.4 * spring(t - tc, 16, 10)))
        if sc <= 0.01:
            continue
        for dd in range(LAT_ND):
            P.append((xx + LAT2_X, (ROW1 if ye else 0.0) + lift, (dd - 1) * LAT_DZ))
            Hh.append((CUBE * g * sc,) * 3)
            C.append((MUST if ye else VERM) * cube_var(k * 7 + n, dd))
    if P:
        F.add_blocks(P, Hh, C)
    add_axes(F, dx=LAT2_X, alpha=0.38)
    # the counter
    ca = smooth(51.8, 52.3, t) * (1 - smooth(57.1, 57.6, t))
    if ca > 0:
        fi = formula_img([('lim', '∫', '0', '1'), ' D(x) dμ'], 40, CREAM)
        F.text(fi, 400, 268, 'rm', ca)
        if t < 54.0:
            fr = int(t * FPS)
            step = 1 + int(4 * ramp(52.5, 54.0, t))
            v = hash01(fr // step, 7)
            s = '= {:.4f}'.format(v)
        else:
            s = '= 1'
        pop = 1 + 0.3 * spring(t - 54.0, 14, 7) if t >= 54.0 else 1
        col = MUST if t >= 54.0 else CREAM
        F.texts.append(('scaled_l', text_img(s, 52, 'mathb', col), 420, 268, ca, pop))
    za = window(t, 55.3, 57.4, 0.4, 0.4)
    if za > 0:
        x, y = cam.screen((LAT2_X, Y_FLOOR - 70, LAT_DZ * 1.6))
        F.text(formula_img(['μ(ℚ ∩ [0,1]) = 0'], 30, VERM * 1.1), x, y, 'cm', za)
    # the lonely fragment
    if 57.7 <= t <= 60.2:
        q = ramp(57.7, 59.15, t)
        px = mix(80, 560, ease_out(q, 2))
        py = mix(560, 610, q) + 10 * math.sin(4 * t)
        shrink = ease_in(ramp(59.05, 59.5, t), 2)
        if shrink < 1:
            F.cards.append(card_dict('那句说错的话', px, py, VERM * 0.9, VERM * 0.6,
                                     math.cos(0.35 * math.sin(2.2 * t)), 1.0 - shrink * 0.97,
                                     smooth(57.7, 57.95, t), size=28))
        if shrink > 0.3:
            fl = window(t, 59.25, 60.2, 0.1, 0.7)
            F.sprites.append((560, 610, 4, CREAM, 3.0 * fl))
            F.sprites.append((560, 610, 18, VERM * 1.2, 0.6 * fl))


# ---------------------------------------------------------------- act 7
def act7(F, t):
    cam = F.cam
    m = MOS
    n = m['n']
    if t < T7 + 0.15:
        return
    sb = np.clip((t - T7 - 0.1) / 0.25, 0, 1)
    s = np.clip((t - m['go']) / (m['arrive'] - m['go']), 0, 1)
    se = s * s * (3 - 2 * s)
    burst = m['src'] + m['kick'] * 70 * ease_out(sb)
    p1 = burst + m['kick'] * 60 + np.array([0, 120, 80])
    p2 = m['P'] + np.array([0, 0, 160])
    P = m['src'] + (burst - m['src']) * ease_out(sb)
    P = np.where((t >= m['go'])[:, None], bezier(burst, p1, p2, m['P'], se), P)
    ang = m['spin'] * (np.sin(np.pi * se)[:, None] + (1 - sb) * 0.0)
    cols = np.array([LEVEL_COL[l] for l in m['lev']]) * np.array([cube_var(i, 11) for i in range(n)])[:, None]
    dim = 1 - 0.5 * smooth(69.0, 69.9, t)
    half = np.full((n, 3), m['c'] * 0.42)
    half[:, 2] = m['c'] * 0.5
    dt = t - m['arrive']
    b = np.where(dt > 0, np.exp(-9 * np.maximum(dt, 0)) * np.cos(15 * np.maximum(dt, 0)), 0.0)
    half[:, 2] *= (1 - 0.5 * b)
    half[:, 0] *= (1 + 0.2 * b)
    half[:, 1] *= (1 + 0.2 * b)
    wave = np.where(s >= 1, 6 * np.sin(2.0 * t - m['P'][:, 0] / 90.0), 0.0)
    P[:, 2] += wave
    for z in m['zero']:
        half[z] = half[z] * mix(1.0, 0.18, smooth(64.3, 64.9, t))
        cols[z] = mix(cols[z], VERM * 1.2, smooth(64.3, 64.9, t))
    F.add_blocks(P, half, cols * dim, euler(ang[:, 0], ang[:, 1], ang[:, 2]))
    moving = np.where((s > 0.02) & (s < 1))[0]
    for i in moving:
        ss = np.linspace(max(se[i] - 0.25, 0), se[i], 7)
        F.trails.append((bezier(burst[i], p1[i], p2[i], m['P'][i], ss), 6.0, cols[i] * 1.2, 0.5))
    ea = smooth(64.4, 65.0, t) * (1 - 0.4 * smooth(69.0, 69.8, t))
    if ea > 0:
        x, y = cam.screen((0, 200 - m['nr'] * m['c'] / 2 - 46, 0))
        F.text(text_img('almost  everywhere', 30, 'math', CREAM * 0.85, spacing=4), x, y, 'cm', ea)
