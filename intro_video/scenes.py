"""The seven scenes. Each draw(P, ov, t, a) adds light to the painter and text to the overlay."""
import math

import numpy as np

from core import (W, H, GOLD, GOLD2, WARM, WHITE, TEAL, BLUEW, NAVY, smooth, window, ease_io, ease_out,
                  hash01, arclen, sub_polyline, point_at, catmull)

DIM = np.array([0.72, 0.74, 0.80], np.float32)


def mix(a, b, u):
    return a * (1 - u) + b * u


def comet(P, pts, cum, s_head, col, t, length=260.0, inten=1.6, width=2.2, npart=60, seed=0, head=1.0):
    sub, ss = sub_polyline(pts, cum, s_head - length, s_head)
    if sub is not None:
        vw = ((ss - (s_head - length)) / length) ** 2
        P.line(sub, col, inten, width, vw=vw)
    rng = np.random.default_rng(seed)
    lag = rng.random(npart) ** 1.4 * length * 1.4
    off = rng.normal(0, 1, npart) * 9
    ph = rng.random(npart) * 6.28
    s = np.clip(s_head - lag, 0, cum[-1])
    x = np.interp(s, cum, pts[:, 0]); y = np.interp(s, cum, pts[:, 1])
    s2 = np.clip(s + 3, 0, cum[-1])
    tx = np.interp(s2, cum, pts[:, 0]) - x; ty = np.interp(s2, cum, pts[:, 1]) - y
    n = np.hypot(tx, ty) + 1e-9
    spread = (lag / length) * (1 + 0.4 * np.sin(t * 3 + ph))
    x = x - ty / n * off * spread
    y = y + tx / n * off * spread
    w = (1 - lag / (length * 1.4)) ** 2 * 0.9 * inten / 1.6
    P.points(x, y, w, mix(col, WARM, 0.4))
    p, _ = point_at(pts, cum, s_head)
    P.sprite(p[0], p[1], 2.2, WARM, 2.2 * head)
    P.sprite(p[0], p[1], 9.0, col, 0.35 * head)


def node(P, x, y, col, inten, t=0.0, ring=0.0):
    P.sprite(x, y, 2.0, WARM, 2.0 * inten)
    P.sprite(x, y, 7.0, col, 0.45 * inten)
    P.sprite(x, y, 20.0, col, 0.08 * inten)
    if ring > 0:
        r = 6 + 70 * ease_out(ring)
        th = np.linspace(0, 2 * np.pi, 200)
        P.line(np.stack([x + r * np.cos(th), y + r * np.sin(th)], 1), col, 0.9 * (1 - ring) * inten, 1.5)


def circle(cx, cy, r, n=None, rx=None, ry=None, rot=0.0):
    rx = r if rx is None else rx
    ry = r if ry is None else ry
    n = n or max(32, int(2 * math.pi * max(rx, ry) / 3))
    th = np.linspace(0, 2 * np.pi, n)
    x, y = rx * np.cos(th), ry * np.sin(th)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


# =====================================================================
# Scene 1 — 初值
K1 = np.array([520.0, 620.0])
_k = 0.35


def _xp(tau):
    return (_k * np.cos(tau) + np.sin(tau)) / (_k * _k + 1)


TAU0 = 2.5
C0 = 1.2
X0 = _xp(TAU0) + C0 * math.exp(-_k * TAU0)


def ode_curve(C, tau):
    x = _xp(tau) + C * np.exp(-_k * (tau - 0.0))
    return np.stack([K1[0] + (tau - TAU0) * 140, K1[1] - (x - X0) * 120], 1)


TAUS = np.linspace(-1.4, 12.8, 900)
FAMILY = [C for C in np.linspace(-7, 7, 46)]


def scene1(P, ov, t, a, cam):
    z = 1.0 + 0.06 * smooth(0, 12, t)

    def S(p):
        return K1 + (p - K1) * z + np.array([0, -6 * smooth(0, 12, t)])

    # family of solution curves (revealed left -> right)
    rev = smooth(0.4, 4.2, t)
    for i, C in enumerate(FAMILY):
        pts = S(ode_curve(C, TAUS))
        cut = int(len(TAUS) * rev)
        if cut < 2:
            continue
        col = GOLD if i % 3 else mix(GOLD, TEAL, 0.6)
        P.line(pts[:cut], col, 0.2 * a * (0.6 + 0.4 * hash01(i, 3)), 1.0)
        # flowing particles with short trails
        cum = arclen(pts[:cut])
        for j in range(2):
            sp = 120 + 60 * hash01(i, j)
            s = (hash01(i, j, 9) * cum[-1] + sp * t) % (cum[-1] + 1)
            sub, ss = sub_polyline(pts[:cut], cum, s - 50, s)
            if sub is not None:
                P.line(sub, WARM, 0.45 * a, 1.0, vw=((ss - (s - 50)) / 50) ** 2)
    # the initial value point
    kp = S(K1)
    ig = smooth(2.6, 3.4, t)
    if ig > 0:
        node(P, kp[0], kp[1], GOLD, 1.6 * a * ig, ring=min(1.0, max(0.0, (t - 2.8) / 1.4)))
    # the unique solution grows out of it in both directions
    g = smooth(3.4, 6.5, t)
    if g > 0:
        pts = S(ode_curve(C0, TAUS))
        i0 = int(np.argmin(np.abs(TAUS - TAU0)))
        n = len(TAUS)
        lo = int(i0 - g * i0)
        hi = int(i0 + g * (n - 1 - i0))
        bright = 1.4 + 0.6 * smooth(6.6, 7.6, t)
        P.line(pts[lo:hi + 1], GOLD, bright * a, 2.0)
        if hi - i0 > 3:
            cp = pts[i0:hi + 1]
            comet(P, cp, arclen(cp), arclen(cp)[-1], GOLD, t, length=200, inten=1.2, seed=1, head=0.8 * (1 - smooth(5.8, 6.6, t)))
        # 'x(t) = 阿白'
        lab = window(t, 7.0, 13.0, 0.8, 0.6)
        if lab > 0:
            j = int(np.argmin(np.abs(pts[:, 0] - 1520)))
            ov.text('x(t) ≡ ', pts[j, 0] - 10, pts[j, 1] - 92, 30, 'math', WARM, a * lab, 'rm', glow=0.4)
            ov.text('阿白', pts[j, 0] - 10, pts[j, 1] - 90, 34, 'serif', GOLD2, a * lab, 'lm', glow=0.6)
            P.sprite(pts[j, 0], pts[j, 1], 2.0, WARM, 1.5 * a * lab)
    # equation
    eq = smooth(0.8, 2.6, t) * (1 - 0.45 * smooth(7, 9, t))
    if eq > 0:
        ov.text('dx/dt = f(t, x),     x(t₀) = x₀', 960, 300 - 4 * z, 58, 'math', WARM, a * eq, 'cm', glow=0.6)
    lab = smooth(4.0, 5.2, t)
    if lab > 0:
        ov.text('昆山', kp[0] - 34, kp[1] + 40, 40, 'serif', GOLD2, a * lab, 'rt', glow=0.6)
        ov.text('KUNSHAN · (t₀, x₀)', kp[0] - 34, kp[1] + 98, 15, 'sans', GOLD, a * lab * 0.9, 'rt', glow=0.2, spacing=3)
        lab2 = smooth(5.0, 6.2, t)
        ov.text('家里做日用品批发，供货给盒马等超市', kp[0] - 34, kp[1] + 128, 20, 'sans', DIM, a * lab2 * 0.9, 'rt', glow=0.1)


# =====================================================================
# Scene 2 — 积分
K2 = K1.copy()
PX = np.arange(380.0, 3300.0, 2.0)


_PC = catmull([[300, 660], [520, 620], [850, 520], [1150, 455], [1500, 450], [1850, 520], [2200, 600],
               [2500, 560], [2800, 470], [3100, 430], [3400, 440]], 80)


def path_y(x):
    return np.interp(x, _PC[:, 0], _PC[:, 1])


PATH = np.stack([PX, path_y(PX)], 1)
PCUM = arclen(PATH)
SUZHOU = np.array([1150.0, float(path_y(1150.0))])
CHAPEL = np.array([2500.0, float(path_y(2500.0))])
BASE_Y = 840.0


def px_of_t(t):
    return 520 + 2160 * ease_io((t - 12.3) / (21.6 - 12.3))


def camx_of_t(t):
    return 1450 * ease_io((t - 12.4) / (22.4 - 12.4))


def s_of_x(x):
    return float(np.interp(x, PX, PCUM))


def scene2(P, ov, t, a, cam):
    z = 1.06 - 0.06 * smooth(12, 14, t)
    cx = camx_of_t(t)

    def S(p):
        p = np.asarray(p, np.float64)
        return K2 + (p - np.array([cx, 0.0]) - K2) * z

    px = px_of_t(t)
    path = S(PATH)
    vis = (path[:, 0] > -80) & (path[:, 0] < W + 80)
    # faint parallel flow lines with particles
    xs = PX
    for k in range(-14, 15):
        if k == 0:
            continue
        off = k * 8.5 * (0.55 + 0.45 * np.sin(xs / 330 + k * 0.7))
        pts = S(np.stack([xs, path_y(xs) + off], 1))[vis]
        if len(pts) < 2:
            continue
        P.line(pts, mix(GOLD, TEAL, 0.5 + 0.5 * math.sin(k)), 0.085 * a, 1.0)
        cum = arclen(pts)
        for j in range(2):
            s = (hash01(k + 40, j) * cum[-1] + (190 + 40 * hash01(k, j)) * t) % (cum[-1] + 1)
            sub, ss = sub_polyline(pts, cum, s - 70, s)
            if sub is not None:
                P.line(sub, WARM, 0.35 * a, 1.0, vw=((ss - (s - 70)) / 70) ** 2)
    # axis + Riemann bars behind the point
    ax = S(np.array([[cx - 100, BASE_Y], [cx + W + 100, BASE_Y]]) + np.array([0, 0]))
    P.line(ax, WARM, 0.25 * a, 1.0)
    for bx in np.arange(540.0, px, 16.0):
        age = px - bx
        f = smooth(0, 120, age)
        top = path_y(bx)
        seg = S(np.array([[bx, BASE_Y], [bx, top]]))
        if seg[0, 0] < -10 or seg[0, 0] > W + 10:
            continue
        P.line(seg, mix(TEAL, GOLD, 0.35), 0.34 * a * f, 1.0, vw=[0.25, 1.0])
        tk = S(np.array([[bx, BASE_Y], [bx, BASE_Y + 8]]))
        P.line(tk, WARM, 0.3 * a * f, 1.0)
    # travelled path bright, future path faint
    sp = s_of_x(px)
    P.line(path[vis], GOLD, 0.16 * a, 1.0)
    done = PATH[:, 0] <= px
    if done.sum() > 2:
        P.line(path[done & vis], GOLD, 1.25 * a, 2.0)
    comet(P, path, PCUM, sp, GOLD, t, length=300, inten=1.8, npart=80, seed=2, head=a * (1 - smooth(21.4, 22.2, t)))
    # Picard iterates converging to the solution near Chapel Hill
    pc = smooth(21.7, 22.4, t)
    if pc > 0:
        xs2 = np.arange(1850.0, 2950.0, 4.0)
        yc = CHAPEL[1]
        base = path_y(xs2)
        for n in range(7):
            show = smooth(21.8 + n * 0.45, 22.3 + n * 0.45, t)
            if show <= 0:
                continue
            yn = base + (yc - base) * (0.55 ** n) + 0.0
            pts = S(np.stack([xs2, yn], 1))
            P.line(pts, mix(TEAL, WARM, n / 7), (0.55 - 0.05 * n) * a * show * pc, 1.0)
        cp = S(CHAPEL)
        al = smooth(22.0, 23.0, t) * a
        ov.text('Picard–Lindelöf 定理', cp[0], 310, 40, 'serif', GOLD2, al, 'ct', glow=0.6)
        ov.text('φₙ₊₁(t) = x₀ + ∫ f(s, φₙ(s)) ds  →  x(t)', cp[0], 372, 26, 'math', WARM, al * 0.9, 'ct', glow=0.3)
    # nodes
    for (pt, name, sub, en, tin) in [(K2, '昆山', '起点', 'KUNSHAN', 12.0),
                                      (SUZHOU, '苏州', '苏州外国语学校 · A-Level', 'SUZHOU', 14.6),
                                      (CHAPEL, 'Chapel Hill', 'UNC · 数学系 · 三年读完 · GPA 3.78', 'NORTH CAROLINA', 19.2)]:
        sp_ = S(pt)
        if sp_[0] < -400 or sp_[0] > W + 400:
            continue
        lit = 1.0 if tin <= 12.0 else smooth(tin, tin + 0.5, t)
        if lit <= 0:
            P.sprite(sp_[0], sp_[1], 3.0, WARM, 0.25 * a)
            continue
        node(P, sp_[0], sp_[1], GOLD, 1.5 * a * lit, ring=0.0 if tin <= 12 else min(1.0, max(0.0, (t - tin) / 1.3)))
        al = a * smooth(tin + 0.1, tin + 0.8, t) if tin > 12 else a
        ov.text(name, sp_[0], sp_[1] + 40, 38, 'serif', GOLD2, al, 'ct', glow=0.6)
        ov.text(en, sp_[0], sp_[1] + 94, 14, 'sans', GOLD, al * 0.85, 'ct', glow=0.2, spacing=3)
        ov.text(sub, sp_[0], sp_[1] + 120, 21, 'sans', DIM, al * 0.95, 'ct', glow=0.1)


# =====================================================================
# Scene 3 — 分支树
ROOT = np.array([360.0, 560.0])
TIPS = [np.array([1230.0, 230.0]), np.array([1300.0, 350.0]), np.array([1250.0, 470.0])]
TIP_NAMES = ['读博？', '港中文', 'NUS']
A_PT = np.array([1060.0, 600.0])
B_PT = np.array([1440.0, 690.0])


def jagged(p0, p1, rng, rough=0.11, depth=6):
    pts = [np.asarray(p0, float), np.asarray(p1, float)]
    for d in range(depth):
        new = [pts[0]]
        for p, q in zip(pts[:-1], pts[1:]):
            m = (p + q) / 2
            dv = q - p
            L = math.hypot(*dv) + 1e-9
            nrm = np.array([-dv[1], dv[0]]) / L
            new += [m + nrm * rng.normal() * L * rough, q]
        pts = new
    return np.array(pts)


def build_tree():
    rng = np.random.default_rng(42)
    br = []
    speed = 620.0

    def add(pts, t0, level, group, inten):
        cum = arclen(pts)
        br.append(dict(pts=pts, cum=cum, t0=t0, level=level, group=group, inten=inten, id=len(br)))
        return br[-1]

    def forks(parent, nf, level, lenr):
        pts, cum = parent['pts'], parent['cum']
        for _ in range(nf):
            i = rng.integers(2, len(pts) - 2)
            d = pts[min(i + 2, len(pts) - 1)] - pts[i - 2]
            ang = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.95)
            L = rng.uniform(*lenr)
            q = pts[i] + L * np.array([math.cos(ang), math.sin(ang)])
            b = add(jagged(pts[i], q, rng, 0.2, 5), parent['t0'] + cum[i] / speed, level, parent['group'],
                    parent['inten'] * 0.45)
            if level < 2:
                forks(b, 2, level + 1, (25, 80))

    trunks = []
    for gi, tip in enumerate(TIPS):
        trunks.append(add(jagged(ROOT, tip, rng), 27.2 + 0.3 * gi, 0, gi, 1.0))
    ta = add(jagged(ROOT, A_PT, rng), 27.5, 0, 'A', 1.0)
    trunks.append(ta)
    tb = add(jagged(A_PT, B_PT, rng), ta['t0'] + ta['cum'][-1] / speed, 0, 'B', 1.0)
    trunks.append(tb)
    for tr in trunks:
        forks(tr, 7, 1, (60, 220))
    # neural fibres
    for k in range(46):
        tgt = np.array([rng.uniform(700, 1500), rng.uniform(210, 820)])
        add(jagged(ROOT, tgt, rng, 0.12, 6), 27.0 + rng.uniform(0, 1.5), 3, 'f', 0.22)
    return br, speed


TREE, TSPEED = build_tree()
CHOSEN = catmull([ROOT, [620, 590], A_PT, [1250, 640], B_PT, [1700, 680], [2050, 640]], 80)
CHOSEN_CUM = arclen(CHOSEN)


def scene3(P, ov, t, a, cam):
    z = 1.0 + 0.05 * smooth(26, 42, t)
    C = np.array([900.0, 540.0])
    drift = np.array([-30 * smooth(26, 42, t), 0])

    def S(p):
        return C + (np.asarray(p) - C) * z + drift

    fi = int(t * 15)
    flick = smooth(30.6, 31.0, t) * (1 - smooth(33.8, 34.2, t))
    dimu = smooth(34.0, 35.6, t)
    for b in TREE:
        L = (t - b['t0']) * TSPEED
        if L <= 0:
            continue
        sub, ss = sub_polyline(b['pts'], b['cum'], 0, L)
        if sub is None:
            continue
        inten = b['inten']
        if flick > 0:
            inten *= 1 - flick + flick * (0.25 + 1.3 * hash01(b['id'], fi))
        g = b['group']
        if g in ('A', 'B'):
            inten *= 1 + 0.2 * dimu
        elif g == 'f':
            inten *= 1 - 0.75 * dimu
        else:
            inten *= 1 - 0.88 * dimu
        col = GOLD if b['level'] < 2 else mix(GOLD, TEAL, 0.55)
        if b['level'] == 3:
            col = mix(TEAL, BLUEW, 0.3)
        width = 1.8 if b['level'] == 0 else 1.0
        # fresh growth front is brighter
        vw = 0.6 + 0.4 * (ss / max(ss[-1], 1))
        P.line(S(sub), col, 0.9 * inten * a, width, vw=vw)
        if L < b['cum'][-1] and b['level'] == 0:
            p = S(sub[-1])
            P.sprite(p[0], p[1], 2.5, WARM, 1.5 * a)
    rp = S(ROOT)
    node(P, rp[0], rp[1], GOLD, 1.3 * a)
    # tips + labels
    for gi, (tip, name) in enumerate(zip(TIPS, TIP_NAMES)):
        tr = TREE[gi]
        reach = tr['t0'] + tr['cum'][-1] / TSPEED
        if t < reach:
            continue
        lit = smooth(reach, reach + 0.4, t)
        f = 1.0
        if flick > 0:
            f = 1 - flick + flick * (0.3 + 1.2 * hash01(gi + 99, fi))
        f *= 1 - 0.8 * dimu
        sp = S(tip)
        node(P, sp[0], sp[1], GOLD, a * lit * f, ring=min(1.0, (t - reach) / 1.2))
        ov.text(name, sp[0] + 26, sp[1], 36, 'serif', GOLD2, a * lit * (1 - 0.65 * dimu) * (0.6 + 0.4 * f), 'lm', glow=0.5)
    for key, pt, name, lofs, anc in [(3, A_PT, '先自学测度论', (-10, 46), 'ct'), (4, B_PT, '成都，教书', (0, 36), 'ct')]:
        tr = TREE[key]
        reach = tr['t0'] + tr['cum'][-1] / TSPEED
        if t < reach:
            continue
        lit = smooth(reach, reach + 0.4, t)
        f = 1.0
        if flick > 0:
            f = 1 - flick + flick * (0.5 + 0.9 * hash01(key + 99, fi))
        sp = S(pt)
        node(P, sp[0], sp[1], GOLD, a * lit * f * (1 + 0.5 * dimu), ring=min(1.0, (t - reach) / 1.2))
        ov.text(name, sp[0] + lofs[0], sp[1] + lofs[1], 36, 'serif', GOLD2, a * lit, anc, glow=0.6 + 0.3 * dimu)
    # the solution: smooth bright curve along the chosen route, extending off-screen
    g = smooth(34.6, 40.5, t)
    if g > 0:
        pts = S(CHOSEN)
        s = g * CHOSEN_CUM[-1]
        sub, ss = sub_polyline(pts, CHOSEN_CUM, 0, s)
        if sub is not None:
            P.line(sub, GOLD, 1.5 * a, 2.2)
            P.line(sub, WARM, 0.5 * a, 6.0)
        comet(P, pts, CHOSEN_CUM, s, GOLD, t, length=280, inten=1.6, npart=70, seed=3, head=a)


# =====================================================================
# Scene 4 — 进动
FOC = np.array([980.0, 530.0])
OA, OE = 230.0, 0.55
OP = OA * (1 - OE * OE)
NU = np.linspace(0, 2 * np.pi, 1700)
OR = OP / (1 + OE * np.cos(NU))
DELTA = math.radians(0.5)
ROT0 = math.radians(-20)
_dx = np.gradient(OR * np.cos(NU)); _dy = np.gradient(OR * np.sin(NU))
ODS = np.hypot(_dx, _dy)
NORB = 365


def n_orbits(t):
    u = max(0.0, t - 43.0)
    c = (NORB - 0.8 * 13.5) / 13.5 ** 3
    return min(NORB, 0.8 * u + c * u ** 3)


def kepler_nu(frac):
    M = 2 * math.pi * frac
    E = M
    for _ in range(8):
        E = E - (E - OE * math.sin(E) - M) / (1 - OE * math.cos(E))
    return 2 * math.atan2(math.sqrt(1 + OE) * math.sin(E / 2), math.sqrt(1 - OE) * math.cos(E / 2)) % (2 * math.pi)


def scene4(P, ov, t, a, cam):
    z = 1.10 - 0.10 * ease_io((t - 43) / 14.0)
    rot_cam = 0.04 * smooth(42, 58, t)

    def S(x, y):
        c, s = math.cos(rot_cam), math.sin(rot_cam)
        return FOC[0] + (c * x - s * y) * z, FOC[1] - (s * x + c * y) * z

    # polar guide grid
    gi = a * smooth(42.0, 43.5, t)
    for r in np.arange(60, 420, 45):
        th = np.linspace(0, 2 * np.pi, 400)
        x, y = S(r * np.cos(th), r * np.sin(th))
        P.line(np.stack([x, y], 1), mix(TEAL, BLUEW, 0.5), 0.09 * gi, 1.0, dash=(6, 3))
    for k in range(24):
        th = k * math.pi / 12
        x, y = S(np.array([40, 420]) * math.cos(th), np.array([40, 420]) * math.sin(th))
        P.line(np.stack([x, y], 1), mix(TEAL, BLUEW, 0.5), 0.06 * gi, 1.0)
    N = n_orbits(t)
    nc = int(math.floor(N))
    # completed orbits, vectorised
    if nc > 0:
        ns = np.arange(nc)
        age = N - (ns + 1)
        inten = (0.05 + 0.55 * np.exp(-age / 3.5)) * a
        if nc > 60:
            inten = inten * (0.6 + 0.4 * 60 / nc)
        inten = inten * np.where(ns % 7 == 6, 3.2, 0.8)
        ang = NU[None, :] + ROT0 + ns[:, None] * DELTA
        r = OR[None, :]
        x, y = S(r * np.cos(ang), r * np.sin(ang))
        u = (ns / NORB)[:, None]
        cols = np.stack([1.0 - 0.8 * u, 0.70 + 0.16 * u, 0.28 + 0.5 * u], -1) * np.ones((1, NU.size, 1))
        w = (inten[:, None] * ODS[None, :]) * np.ones_like(x)
        P.points(x, y, w, cols.reshape(-1, 3).astype(np.float32))
    # first orbit ghost at the end
    if t > 52:
        g = smooth(52.5, 54, t)
        ang = NU + ROT0
        x, y = S(OR * np.cos(ang), OR * np.sin(ang))
        P.line(np.stack([x, y], 1), TEAL, 0.45 * a * g, 1.2, dash=(10, 6))
    # current orbit + planet
    if N < NORB:
        frac = N - nc
        nu = kepler_nu(frac)
        m = NU <= nu
        ang = NU[m] + ROT0 + nc * DELTA
        x, y = S(OR[m] * np.cos(ang), OR[m] * np.sin(ang))
        pts = np.stack([x, y], 1)
        if len(pts) > 2:
            cum = arclen(pts)
            P.line(pts, GOLD, 0.6 * a, 1.2)
            comet(P, pts, cum, cum[-1], GOLD, t, length=min(240, cum[-1]), inten=1.6, npart=50, seed=4, head=a)
    else:
        ang = NU + ROT0 + (NORB - 1) * DELTA
        x, y = S(OR * np.cos(ang), OR * np.sin(ang))
        P.line(np.stack([x, y], 1), GOLD, 1.0 * a, 1.8)
    # sun
    sx, sy = S(0.0, 0.0)
    P.sprite(sx, sy, 3.0, WARM, 2.5 * a)
    P.sprite(sx, sy, 12.0, mix(GOLD, TEAL, 0.3), 0.5 * a)
    P.sprite(sx, sy, 34.0, TEAL, 0.07 * a)
    # accumulated-angle arc at the sun
    tot = min(nc, NORB) * DELTA
    if tot > 0.01:
        th = np.linspace(ROT0, ROT0 + tot, max(3, int(tot * 60)))
        x, y = S(52 * np.cos(th), 52 * np.sin(th))
        P.line(np.stack([x, y], 1), TEAL, 0.9 * a, 1.4)
        for th0 in (ROT0, ROT0 + tot):
            x, y = S(np.array([0, 300]) * math.cos(th0), np.array([0, 300]) * math.sin(th0))
            P.line(np.stack([x, y], 1), TEAL, 0.25 * a, 1.0, dash=(8, 6))
    # corner annotation (top-right)
    al = a * smooth(43.0, 44.0, t)
    ov.text('δφ = 0.5° / 天', 1820, 118, 26, 'math', WARM, al, 'rt', glow=0.3)
    ov.text(f'n = {min(nc, NORB):d} 天', 1820, 160, 26, 'math', WARM, al, 'rt', glow=0.3)
    ov.text(f'Σ δφ = {min(nc, NORB) * 0.5:.1f}°', 1820, 202, 30, 'math', GOLD2, al, 'rt', glow=0.5)
    ov.text('ACCUMULATED PRECESSION', 1820, 252, 13, 'sans', GOLD, al * 0.8, 'rt', glow=0.1, spacing=3)


# =====================================================================
# Scene 5 — 方向场
LABELS5 = [  # name, sub, screen pos, X_k, side
    ('纯数学', '尤其偏纯数的 ODE', (640, 285), -2.4, +1),
    ('古典乐', '听一遍就能弹', (1360, 770), 3.0, -1),
    ('maimai DX', '', (1440, 285), 3.6, +1),
    ('阿白 & 焚烽', '原创角色 · 狼与虎', (560, 770), -3.0, -1),
    ('带风扇的兽装', '自制 · 里面装了散热风扇', (1720, 690), 4.4, -1),
    ('《念张师》1.4×', 'Suno · 慢摇土嗨', (1040, 270), 0.6, +1),
    ('日本', '去过很多次 · 正在学日语', (960, 790), 0.0, -1),
    ('撞色拼接', '', (230, 410), -4.4, +1),
]
T5 = [59.0 + 1.55 * i for i in range(8)]


def proj5(X, Y, Z, t):
    th = -0.10 + 0.20 * ease_io((t - 58) / 16)
    Xr = X * math.cos(th) - (Z - 4) * math.sin(th)
    Zr = X * math.sin(th) + (Z - 4) * math.cos(th) + 4
    hc, zc, ph = 2.2, 2.5, 0.32
    f = 900 + 60 * smooth(58, 74, t)
    yr = Y - hc
    zr = Zr + zc
    yc = yr * math.cos(ph) + zr * math.sin(ph)
    zcam = -yr * math.sin(ph) + zr * math.cos(ph)
    zcam = np.maximum(zcam, 0.3)
    return 960 + f * Xr / zcam, 520 - f * yc / zcam


def warp5(X, Z):
    return -0.35 * np.exp(-((X - 1.5) ** 2 + (Z - 6.0) ** 2) / 3.0) - 0.25 * np.exp(-((X + 2.5) ** 2 + (Z - 3.0) ** 2) / 2.0)


def slope5(X, Z):
    return 0.55 * np.sin(0.9 * X + 0.5 * Z) + 0.15 * (4 - Z)


def curve5_Z(X, t, scale=1.0, base=4.0):
    Z = base + 0.35 * np.sin(0.7 * X)
    for (name, sub, pos, Xk, side), tk in zip(LABELS5, T5):
        u = t - tk
        if u <= 0:
            continue
        A = side * 1.15 * scale * (1 - math.exp(-3.2 * u) * math.cos(8 * u))
        Z = Z + A * np.exp(-(X - Xk) ** 2 / (2 * 0.8 ** 2))
    return Z


def scene5(P, ov, t, a, cam):
    gi = a * smooth(58.0, 59.5, t)
    # curved grid
    Xs = np.linspace(-6, 6, 241)
    for Z in np.linspace(0.6, 10, 26):
        X, Y = proj5(Xs, warp5(Xs, Z), np.full_like(Xs, Z), t)
        fz = 0.32 * (1 - (Z - 0.6) / 11)
        P.line(np.stack([X, Y], 1), mix(NAVY, BLUEW, 0.6), fz * gi, 1.0)
    Zs = np.linspace(0.6, 10, 200)
    for Xg in np.linspace(-6, 6, 31):
        X, Y = proj5(np.full_like(Zs, Xg), warp5(Xg, Zs), Zs, t)
        P.line(np.stack([X, Y], 1), mix(NAVY, BLUEW, 0.6), 0.28 * gi, 1.0, vw=1 - (Zs - 0.6) / 11)
    # direction field
    gx, gz = np.meshgrid(np.linspace(-5.5, 5.5, 34), np.linspace(1.0, 9.0, 17))
    gx, gz = gx.ravel(), gz.ravel()
    sl = slope5(gx, gz)
    d = np.stack([np.ones_like(sl), sl], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    hl = 0.13
    ph = (np.arange(gx.size) * 0.37) % 1.0
    lt = np.linspace(-1, 1, 12)
    for i in range(gx.size):
        X = gx[i] + d[i, 0] * hl * lt
        Z = gz[i] + d[i, 1] * hl * lt
        x, y = proj5(X, warp5(X, Z), Z, t)
        flash = 0.6 + 0.4 * math.sin(t * 2.0 + ph[i] * 6.28 + gx[i])
        P.line(np.stack([x, y], 1), mix(TEAL, GOLD, 0.25), 0.9 * gi * flash * (1 - (gz[i] - 1) / 11), 1.0,
               vw=np.linspace(0.2, 1, 12))
    # neighbour solutions + main curve
    Xc = np.linspace(-6, 6, 600)
    for k in range(-9, 10):
        if k == 0:
            continue
        Zk = curve5_Z(Xc, t, scale=1 - abs(k) / 12, base=4.0 + k * 0.17)
        x, y = proj5(Xc, warp5(Xc, Zk), Zk, t)
        P.line(np.stack([x, y], 1), mix(GOLD, TEAL, abs(k) / 10), 0.16 * gi, 1.0)
    Zm = curve5_Z(Xc, t)
    x, y = proj5(Xc, warp5(Xc, Zm), Zm, t)
    main = np.stack([x, y], 1)
    cum = arclen(main)
    P.line(main, GOLD, 1.5 * gi, 2.2)
    P.line(main, WARM, 0.4 * gi, 5.0)
    for j in range(3):
        s = ((t - 58) * 260 + j * cum[-1] / 3) % cum[-1]
        comet(P, main, cum, s, GOLD, t, length=220, inten=1.3, npart=40, seed=10 + j, head=gi * 0.8)
    # labels
    for i, ((name, sub, pos, Xk, side), tk) in enumerate(zip(LABELS5, T5)):
        if t < tk - 0.05:
            continue
        al = a * smooth(tk, tk + 0.5, t)
        Zk = float(curve5_Z(np.array([Xk]), t)[0])
        Xq = max(-5.5, min(5.5, Xk))
        Zq = float(curve5_Z(np.array([Xq]), t)[0])
        cx, cy = proj5(np.array([Xq]), warp5(np.array([Xq]), np.array([Zq])), np.array([Zq]), t)
        tether = smooth(tk, tk + 0.3, t) * (0.35 + 0.65 * (1 - smooth(tk + 0.6, tk + 2.0, t)))
        ly = pos[1] + (26 if side > 0 else -26) * 0 + (34 if pos[1] < 400 else -30 if pos[1] > 700 else 0)
        P.line(np.array([[pos[0], ly], [cx[0], cy[0]]]), TEAL, 0.55 * a * tether, 1.0, dash=(5, 4, -t * 30))
        P.sprite(cx[0], cy[0], 2.0, WARM, 1.6 * a * tether)
        burst = 1 - smooth(tk, tk + 1.0, t)
        P.sprite(pos[0], pos[1], 26.0, GOLD, 0.25 * a * burst)
        ov.text(name, pos[0], pos[1], 34, 'serif', GOLD2, al, 'cm', glow=0.6)
        if sub:
            ov.text(sub, pos[0], pos[1] + 36, 18, 'sans', DIM, al * 0.9, 'cm', glow=0.1)


# =====================================================================
# Scene 6 — 现在
C6 = np.array([960.0, 470.0])
LAB6 = [('Axler 测度论', '自学中', (520, 330), 76.0),
        ('自己的 agent', '用 Claude Code 搭', (1400, 320), 76.7),
        ('秒判擂台', '高数 PK 对战 app', (1420, 700), 77.4)]
MAIL_PATH = catmull([C6, [945, 360], [915, 230], [880, 100], [840, -60]], 60)
MAIL_CUM = arclen(MAIL_PATH)


def scene6(P, ov, t, a, cam):
    z = 1.08 - 0.08 * ease_io((t - 74) / 10)

    def S(p):
        return C6 + (np.asarray(p, float) - C6) * z

    on = smooth(74.4, 75.2, t)
    # orbit rings
    rng = np.random.default_rng(6)
    for i in range(30):
        r = 70 + i * 11
        rot = rng.uniform(0, math.pi) + t * 0.04 * (1 if i % 2 else -1)
        ratio = rng.uniform(0.28, 0.5)
        pts = S(circle(C6[0], C6[1], r, rx=r, ry=r * ratio, rot=rot))
        grow = smooth(74.2 + i * 0.04, 75.4 + i * 0.04, t)
        col = GOLD if i % 3 else TEAL
        P.line(pts, col, 0.16 * a * grow, 1.0)
        if i in (8, 15, 24):
            th = t * (0.9 - i * 0.02) + i
            x = r * math.cos(th); y = r * ratio * math.sin(th)
            c, s = math.cos(rot), math.sin(rot)
            q = S(C6 + np.array([c * x - s * y, s * x + c * y]))
            P.sprite(q[0], q[1], 2.0, WARM, 1.2 * a * grow)
            P.sprite(q[0], q[1], 7.0, col, 0.25 * a * grow)
    cp = S(C6)
    node(P, cp[0], cp[1], GOLD, 2.0 * a * on, ring=min(1.0, max(0.0, (t - 74.5) / 1.4)))
    ov.text('成都', cp[0], cp[1] + 56, 46, 'serif', GOLD2, a * on, 'ct', glow=0.7)
    al = a * smooth(75.2, 76.0, t)
    ov.text('威学一百 · 教 Edexcel IAL 数学', cp[0], cp[1] + 124, 22, 'sans', WARM, al, 'ct', glow=0.2)
    # course palette swatches
    for k, (nm, colr) in enumerate([('藏蓝', NAVY * 2.2), ('青绿', TEAL), ('金', GOLD)]):
        x = cp[0] - 120 + k * 120
        y = cp[1] + 178
        P.sprite(x - 26, y + 1, 4.5, colr, 1.6 * al)
        ov.text(nm, x - 14, y, 18, 'sans', DIM, al, 'lm', glow=0.1)
    ov.text('课件配色', cp[0], cp[1] + 210, 13, 'sans', GOLD, al * 0.7, 'ct', glow=0.0, spacing=3)
    # satellites: projects
    for (name, sub, pos, tk) in LAB6:
        g = smooth(tk, tk + 0.6, t)
        if g <= 0:
            continue
        p = np.array(pos, float)
        P.line(np.array([cp, cp + (p - cp) * ease_out((t - tk) / 0.8) * 0.86]), TEAL, 0.55 * a * g, 1.0,
               dash=(6, 5, -t * 25))
        P.sprite(p[0], p[1], 26.0, GOLD, 0.12 * a * g)
        ov.text(name, p[0], p[1], 36, 'serif', GOLD2, a * g, 'cm', glow=0.6)
        ov.text(sub, p[0], p[1] + 40, 19, 'sans', DIM, a * g * 0.9, 'cm', glow=0.1)
    # the email to the advisor
    if t > 79.6:
        u = (t - 79.6) / 3.2
        s = ease_io(u) * MAIL_CUM[-1]
        pts = S(MAIL_PATH)
        comet(P, pts, MAIL_CUM, s, TEAL, t, length=260, inten=1.4, npart=60, seed=7, head=a)
        p, tg = point_at(pts, MAIL_CUM, s)
        # envelope icon
        nx, ny = -tg[1], tg[0]
        hw, hh = 15, 10
        cx_, cy_ = p[0] + nx * 26, p[1] + ny * 26
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
        env = np.array([[cx_ + c[0], cy_ + c[1]] for c in corners])
        P.line(env, WARM, 1.1 * a, 1.0)
        P.line(np.array([[cx_ - hw, cy_ - hh], [cx_, cy_ + 2], [cx_ + hw, cy_ - hh]]), WARM, 1.1 * a, 1.0)
        al = a * window(t, 80.0, 84.5, 0.6, 0.6)
        ov.text('致 · 本科导师', 990, 236, 24, 'serif', WARM, al, 'lm', glow=0.4)
        ov.text('隔了一年', 990, 272, 16, 'sans', DIM, al * 0.9, 'lm', glow=0.1)


# =====================================================================
# Scene 7 — 结尾
PTS7_X = np.linspace(330, 1590, 10)
_rng7 = np.random.default_rng(77)
_y7 = 700 - 0.00011 * (PTS7_X - 900) ** 2 * 1.0 - 0.06 * (PTS7_X - 960)
PTS7_Y = _y7 + np.array([18, -26, 22, -14, 28, -24, 12, -30, 20, -10])
TAGS7 = ['昆山', '苏外', 'UNC', 'Picard–Lindelöf', '日语', '古典乐', 'maimai DX', '焚烽', '成都', '测度论']
_c2 = np.polyfit(PTS7_X / 1000, PTS7_Y, 2)
_c9 = np.polyfit(PTS7_X / 1000, PTS7_Y, 9)


def scene7(P, ov, t, a, cam):
    z = 1.0 + 0.03 * smooth(84, 92, t)

    def S(x, y):
        return 960 + (x - 960) * z, 600 + (y - 600) * z

    # points
    for i, (x, y) in enumerate(zip(PTS7_X, PTS7_Y)):
        ti = 84.2 + i * 0.12
        g = smooth(ti, ti + 0.4, t)
        if g <= 0:
            continue
        sx, sy = S(x, y)
        node(P, sx, sy, GOLD if i % 2 == 0 else mix(GOLD, TEAL, 0.5), 1.2 * a * g)
        dy = 30 if PTS7_Y[i] < _y7[i] else -30
        dy = -dy
        ov.text(TAGS7[i], sx, sy + (34 if dy > 0 else -34), 17, 'sans', DIM, a * g * 0.85 * (1 - 0.5 * smooth(89, 91, t)), 'cm', glow=0.1)
    # overfit curve flickers then fades
    xs = np.linspace(260, 1660, 700)
    ov9 = window(t, 84.9, 86.6, 0.3, 0.9)
    if ov9 > 0:
        y9 = np.polyval(_c9, xs / 1000)
        y9 = 660 + 240 * np.tanh((y9 - 660) / 240)
        fl = 0.6 + 0.4 * hash01(int(t * 20), 5)
        x, y = S(xs, y9)
        P.line(np.stack([x, y], 1), TEAL, 0.5 * a * ov9 * fl, 1.0)
    # simple smooth curve, solid then dashed
    g = smooth(85.6, 87.8, t)
    if g > 0:
        xe = 260 + (1640 - 260) * g
        xs2 = np.linspace(260, xe, 500)
        x, y = S(xs2, np.polyval(_c2, xs2 / 1000))
        pts = np.stack([x, y], 1)
        P.line(pts, GOLD, 1.5 * a, 2.0)
        P.line(pts, WARM, 0.3 * a, 6.0)
    d = smooth(87.6, 90.5, t)
    if d > 0:
        xe = 1640 + (2000 - 1640) * d
        xs3 = np.linspace(1640, xe, 300)
        x, y = S(xs3, np.polyval(_c2, xs3 / 1000))
        P.line(np.stack([x, y], 1), GOLD, 1.2 * a, 1.8, dash=(14, 9), vw=np.linspace(1, 0.5, 300))
    # title
    al = a * smooth(86.4, 87.6, t)
    ov.text('阿白', 960, 395, 128, 'serifb', GOLD2, al, 'cm', glow=0.9)
    al2 = a * smooth(87.4, 88.4, t)
    ov.text('存在，且唯一。 ∎', 960, 505, 36, 'serif', WARM, al2, 'cm', glow=0.4)


SCENES = [
    # start, end, fn, hud
    (0.0, 12.0, scene1, ('初值', 'INITIAL VALUE', 'x(t₀) = x₀ · 昆山')),
    (12.0, 26.0, scene2, ('积分', 'INTEGRAL', 'x(t) = x₀ + ∫ f(s, x(s)) ds')),
    (26.0, 42.0, scene3, ('可能性', 'BRANCHES', '读博？ · 港中文 · NUS · 测度论 · 成都')),
    (42.0, 58.0, scene4, ('进动', 'PRECESSION', '每天 30 分钟')),
    (58.0, 74.0, scene5, ('方向场', 'DIRECTION FIELD', 'dy/dx = f(x, y)')),
    (74.0, 84.0, scene6, ('t = 2026 · 成都', 'THE PRESENT', '威学一百 · Edexcel IAL')),
    (84.0, 92.0, scene7, ('存在，且唯一', 'Q.E.D.', 'Picard–Lindelöf')),
]

SUBS = [
    (1.2, 5.6, '给定初值，解【存在】，且【唯一】。'),
    (6.4, 11.4, '这个解，叫【阿白】。'),
    (12.7, 16.4, '苏外，【A-Level】，数学开始变得【认真】。'),
    (16.9, 21.1, '【UNC Chapel Hill】，三年读完，GPA 【3.78】。'),
    (21.6, 25.6, '人生第一个数学展示，讲的就是【这条定理】。'),
    (27.0, 32.4, '每个路口，都有【很多条可能】。'),
    (34.2, 41.2, '但解只有一个——是【自己走出来】的那一条。'),
    (43.0, 47.0, '【Genki II】，每天【三十分钟】。'),
    (47.6, 51.6, '每一圈，只偏【一点点】。'),
    (52.2, 57.4, '一年以后，整条轨道【换了方向】。'),
    (59.6, 73.0, '让曲线【拐弯】的，是这些东西。'),
    (74.8, 78.8, '在成都，教 【IAL 数学】。'),
    (79.3, 83.5, '以及，隔了一年，重新写给导师的【那封邮件】。'),
    (84.5, 87.7, '不需要一条复杂的曲线，去解释【每一个点】。'),
    (88.1, 91.4, '定理只保证【局部存在】，往后的，要自己【积出来】。'),
]
