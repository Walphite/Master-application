"""Two-voice subtitles: math line (cream, smaller, cobalt bar on the left) above,
life line (larger, bold, keywords in mustard) below. Characters pop in one by one.

Entries: (start, end, kind, text). '\n' breaks lines, 【】 marks highlights, '‖' adds a pause.
Layout keeps everything above the bottom 25 % and left of the right 15 % of the frame.
"""
import math

import numpy as np
from PIL import Image

import core
from core import S, CREAM, MUST, COBALT, VERM, INK, smooth, ease_out, ease_back, text_img, blit_arr

SUBS = [
    # 0-4 hook
    (0.20, 4.00, 'm', '数学里有个函数，\n碎到没人能算出它的面积。'),
    (2.45, 4.00, 'l', '就像你的【生活】。'),
    # 4-16 shattered
    (4.15, 10.05, 'm', '狄利克雷函数：\n在任意小的一段里，都同时取两个值。'),
    (6.85, 8.75, 'l', '【七点四十】的地铁，\n被叫进办公室的【五分钟】，'),
    (8.75, 10.05, 'l', '被夸了一句的【下午】——'),
    (10.20, 15.90, 'm', '放大一百倍，分不开。‖\n放大一万倍，还是分不开。'),
    (12.90, 15.90, 'l', '你以为过段时间就能理清，\n其实【越看越乱】。'),
    # 16-25.6 Riemann
    (16.10, 18.70, 'm', '黎曼积分的办法：\n按顺序把区间切开，一段一段加。'),
    (18.70, 25.45, 'm', '对这个函数，它失败了。'),
    (20.30, 22.60, 'l', '我们回头看一年，\n也是从【一月】数到【十二月】。'),
    (22.60, 25.45, 'l', '越数越乱，最后只剩一句：\n这一年，好像【什么都没干成】。'),
    # 25.6-32.6 turn
    (25.80, 28.55, 'm', '1902 年，勒贝格换了个方向：\n不竖着切，横着切。'),
    (28.55, 32.45, 'm', '他说，这就像数钱：\n先把同样面额的放在一起，再数。'),
    (30.90, 32.45, 'l', '不按【时间】算，按【分量】算。'),
    # 32.6-51.4 climax
    (32.80, 40.50, 'm', '同样高度的点，\n不管散落在哪里，都归进同一层。'),
    (35.40, 37.60, 'l', '妈妈的 【60 秒语音】，\n散在几十个晚上；'),
    (37.60, 39.00, 'l', '一年只见【三次】的朋友；'),
    (39.00, 40.50, 'l', '加班后【那碗面】——'),
    (40.60, 43.55, 'l', '它们被时间切碎了，\n在这里，被放回【同一层】。'),
    (43.70, 51.20, 'm', '把每一层叠起来，就是积分。'),
    (46.00, 48.40, 'l', '你以为这一年\n什么都没剩下——'),
    (48.40, 51.20, 'l', '它们叠起来，有【这么高】。'),
    # 51.4-62.6 measure zero
    (51.70, 54.55, 'm', '那个不可积的函数，\n勒贝格算出来了：答案是 1。'),
    (54.55, 62.40, 'm', '因为那些零散的点，一个一个数得过来，\n加起来的测度是零。'),
    (57.70, 59.80, 'l', '那句说错的话，\n那次没忍住的【眼泪】——'),
    (59.80, 62.40, 'l', '你反复回想的那一个瞬间，\n改变不了【整个积分】。'),
    # 62.6-72 ending
    (62.80, 69.00, 'm', '数学家管这叫：几乎处处。\n除了测度为零的地方，处处成立。'),
    (66.30, 69.00, 'l', '你不需要每一刻都好。\n【几乎处处】，就够了。'),
    (69.30, 72.30, 'q', '你的生活里，\n【最厚的那一层】是什么？'),
]

CX = 486.0          # centre of the subtitle column (design px); safe area is x < 918
Y_M_BOTTOM = 1214.0
Y_L_TOP = 1246.0
STYLE = {
    'm': dict(size=40, kind='sansm', lh=58, cps=0.042),
    'l': dict(size=58, kind='sansb', lh=80, cps=0.052),
    'q': dict(size=64, kind='sansb', lh=90, cps=0.075),
}
POP = 0.26
PAUSE = 0.55


def _lines(text):
    """Split into lines; returns list of (line_text, [char start offsets])."""
    out, tcur = [], 0.0
    for raw in text.split('\n'):
        starts, clean = [], ''
        for ch in raw:
            if ch == '‖':
                tcur += PAUSE
                continue
            clean += ch
            if ch not in '【】':
                starts.append(tcur)
                tcur += 1.0
        out.append((clean, starts))
    return out


def _scaled(rgb, a, scale):
    nz = a > 1e-4
    col = np.zeros_like(rgb)
    col[nz] = rgb[nz] / a[nz][:, None]
    im = Image.fromarray(np.clip(col * 255, 0, 255).astype(np.uint8), 'RGB')
    im.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    w, h = im.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(im, np.float32) / 255.0
    aa = arr[:, :, 3]
    return arr[:, :, :3] * aa[:, :, None], aa


def _draw_line(img, ti, starts, t0, cps, x_left, y_top, t, alpha_out, dy_out):
    """Draw one line with per-character pop. x_left / y_top in real pixels (glyph box)."""
    rgb, a, pad = ti['rgb'], ti['a'], ti['pad']
    chars = ti['chars']
    n = len(chars)
    prog = [((t - (t0 + s * cps)) / POP) for s in starts]
    first = next((i for i, p in enumerate(prog) if p < 1.0), n)
    X0, Y0 = x_left - pad, y_top - pad + dy_out
    # settled prefix as one crop
    if first > 0:
        xe = int(chars[first][0] - 1) if first < n else rgb.shape[1]
        blit_arr(img, rgb[:, :xe], a[:, :xe], X0, Y0, alpha_out)
    for i in range(first, n):
        p = prog[i]
        if p <= 0:
            continue
        q = min(p, 1.0)
        sc = 0.45 + 0.55 * ease_back(q, 2.2)
        al = min(1.0, q * 2.2) * alpha_out
        dy = (1 - ease_out(q)) * 20 * S
        x0 = int(chars[i][0] - 2)
        x1 = int(chars[i][0] + chars[i][1] + 2) if i + 1 >= n else int(chars[i + 1][0] - 1)
        x1 = max(x1, x0 + 2)
        cr, ca = rgb[:, x0:x1], a[:, x0:x1]
        if abs(sc - 1) < 0.01:
            blit_arr(img, cr, ca, X0 + x0, Y0 + dy, al)
        else:
            sr, sa = _scaled(np.ascontiguousarray(cr), np.ascontiguousarray(ca), sc)
            cx, cy = X0 + (x0 + x1) / 2, Y0 + rgb.shape[0] / 2 + dy
            blit_arr(img, sr, sa, cx - sr.shape[1] / 2, cy - sr.shape[0] / 2, al)


def draw(img, t):
    for (a, b, kind, text) in SUBS:
        if not (a <= t <= b + 0.05):
            continue
        st = STYLE[kind]
        lines = _lines(text)
        tis = [text_img(l, st['size'], st['kind'], CREAM, MUST, shadow=0.85) for l, _ in lines]
        out = smooth(b - 0.28, b, t) if kind != 'q' else 0.0
        alpha_out = 1 - out
        dy_out = -14 * S * out
        lh = st['lh'] * S
        if kind == 'm':
            bw = max(ti['w'] for ti in tis)
            nl = len(tis)
            top = Y_M_BOTTOM * S - nl * lh
            xl = CX * S - bw / 2 + 14 * S
            # cobalt bar
            g = ease_out((t - a) / 0.35)
            bh = (nl * lh - 14 * S) * g
            ym = top + nl * lh / 2 + 4 * S + dy_out
            y0, y1 = int(ym - bh / 2), int(ym + bh / 2)
            x0 = int(xl - 22 * S)
            x1 = x0 + max(2, int(round(5 * S)))
            if y1 > y0:
                reg = img[max(y0, 0):y1, max(x0, 0):x1]
                reg *= 1 - 0.95 * alpha_out
                reg += COBALT * 1.08 * 0.95 * alpha_out
            for li, (ti, (l, starts)) in enumerate(zip(tis, lines)):
                y = top + li * lh + (lh - ti['h']) / 2
                _draw_line(img, ti, starts, a + 0.08, st['cps'], xl, y, t, alpha_out, dy_out)
        else:
            top = (Y_L_TOP if kind == 'l' else 1180.0) * S
            for li, (ti, (l, starts)) in enumerate(zip(tis, lines)):
                y = top + li * lh + (lh - ti['h']) / 2
                xl = CX * S - ti['w'] / 2
                _draw_line(img, ti, starts, a + 0.05, st['cps'], xl, y, t, alpha_out, dy_out)
    # closing note in the top-left corner
    na = smooth(69.4, 69.9, t)
    if na > 0:
        ti = text_img('测度论笔记 · 01', 26, 'sansm', CREAM * 0.85, shadow=0.5)
        x, y = 64 * S, 168 * S
        reg = img[int(y + 4 * S):int(y + ti['h'] - 2 * S), int(x - 16 * S):int(x - 12 * S)]
        reg *= 1 - na
        reg += COBALT * na
        blit_arr(img, ti['rgb'], ti['a'], x - ti['pad'], y - ti['pad'], na)
