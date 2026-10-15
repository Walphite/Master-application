"""Frame compositor, contact sheets, chunked resumable rendering and muxing.

  LB_SCALE=0.5 python3 render.py keys s1 0 1 2 3.5     # contact sheet build_540/keys_s1.png
  LB_SCALE=0.5 python3 render.py all 540               # render chunks for up to 540 s, rerun to resume
  LB_SCALE=0.5 python3 render.py mux                   # concat chunks (+ music) -> build_540/*.mp4
  python3 render.py all 540 && python3 render.py mux   # 1080x1920 final, writes the two mp4s next to this file

LB_NFRAG sets the fragment budget (default 400), NPROC the worker count.
"""
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import core
from core import (W, H, S, SS, FPS, NFRAMES, DUR, CREAM, INK, COBALT, Cam, Textures, draw_blocks, poly_strip,
                  add_sprite, blit_arr, blit_centered, anchor_xy, text_img, badge_img, smooth, ease_out, ease_back)
import scenes
import subs

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = S >= 0.999
OUT = os.path.join(HERE, 'build_%d' % W)
CHUNK = 60
_tex = None


def textures():
    global _tex
    if _tex is None:
        _tex = Textures()
    return _tex


def to8(c, a=1.0):
    c = np.clip(np.asarray(c, np.float64) * a, 0, 1)
    return (int(c[0] * 255 + 0.5), int(c[1] * 255 + 0.5), int(c[2] * 255 + 0.5))


def _line_pts(cam, pts, kind):
    if isinstance(kind, str) and kind == 'screen':
        return np.asarray(pts, np.float64) * S * SS
    xs, ys, _ = cam.project(np.asarray(pts, np.float64))
    return np.stack([xs, ys], 1) * SS


def draw_lines(cam, lines, dr, dm):
    for pts, width, col, alpha, dash in lines:
        if alpha <= 0.003:
            continue
        P = _line_pts(cam, pts, dash)
        wpx = max(1, int(round(width * S * SS)))
        segs = []
        if isinstance(dash, tuple):
            on, off = dash[0] * S * SS, dash[1] * S * SS
            for a, b in zip(P[:-1], P[1:]):
                L = float(np.hypot(*(b - a)))
                s = 0.0
                while s < L:
                    e = min(s + on, L)
                    segs.append([tuple(a + (b - a) * s / L), tuple(a + (b - a) * e / L)])
                    s += on + off
        else:
            segs.append([tuple(p) for p in P])
        for sg in segs:
            dr.line(sg, fill=to8(col, alpha), width=wpx)
            dm.line(sg, fill=int(alpha * 255), width=wpx)


def draw_card(c, dr, dm, ds):
    ti = text_img(c['text'], c['size'], 'sansb', CREAM)
    sc = c['scale']
    w = (ti['w'] / S + 36) * sc
    h = (ti['h'] / S + 22) * sc
    fx = abs(c['flip'])
    wv = max(w * fx, 2.0)
    face = c['face'] if c['flip'] >= 0 else c['back']
    a = c['alpha']
    k = S * SS
    x0, y0 = (c['x'] - wv / 2) * k, (c['y'] - h / 2) * k
    x1, y1 = (c['x'] + wv / 2) * k, (c['y'] + h / 2) * k
    r = max(1, int(min(11 * sc * k, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1)))
    dep = 7 * sc * k
    dark = to8(face * 0.5, a)
    steps = max(1, int(dep / 2))
    for i in range(steps, 0, -1):
        o = dep * i / steps
        dr.rounded_rectangle([x0 + o * 0.5, y0 + o, x1 + o * 0.5, y1 + o], radius=r, fill=dark)
        dm.rounded_rectangle([x0 + o * 0.5, y0 + o, x1 + o * 0.5, y1 + o], radius=r, fill=int(a * 255))
    dr.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=to8(face, a))
    dm.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=int(a * 255))
    if (x1 - x0) > 4 * r:
        dr.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=to8(face * 1.35 + 0.06, a), width=max(1, int(k * 1.5)))
    q = 0.25 / SS
    ds.rounded_rectangle([(x0 + 12 * k) * q, (y0 + 24 * k) * q, (x1 + 12 * k) * q, (y1 + 24 * k) * q],
                         radius=max(1, int(r * q)), fill=int(170 * a))
    return ti, fx


def render_frame(t):
    tex = textures()
    F = scenes.build(t)
    cam = F.cam
    Wc, Hc = W * SS, H * SS
    rgb = Image.new('RGB', (Wc, Hc))
    msk = Image.new('L', (Wc, Hc))
    dr, dm = ImageDraw.Draw(rgb), ImageDraw.Draw(msk)
    sh = Image.new('L', (W // 4 + 1, H // 4 + 1))
    ds = ImageDraw.Draw(sh)
    # far debris (own camera, weak parallax)
    if F.bgblocks:
        bgcam = Cam(yaw=cam.yaw * 0.4 - 0.15, pitch=cam.pitch, zoom=0.78 + 0.22 * cam.zoom,
                    target=(cam.target[0] * 0.3, (cam.target[1] - 160) * 0.3 + 160, 0), sx=540, sy=960 - 160 * 0.3,
                    roll=cam.roll * 0.5)
        draw_blocks(bgcam, F.bgblocks, dr, dm)
    draw_lines(cam, F.back_lines, dr, dm)
    for pts, width, col, alpha in F.trails:
        poly = poly_strip(cam, pts, width * S * cam.zoom)
        if poly:
            dr.polygon(poly, fill=to8(col, alpha))
            dm.polygon(poly, fill=int(255 * alpha))
    off = (14 * S * cam.zoom / 4, 24 * S * cam.zoom / 4)
    draw_blocks(cam, F.blocks, dr, dm, ds, off)
    # front canvas: dashed cut lines, leader lines, cards
    front = None
    card_txt = []
    if F.front_lines or F.cards:
        frgb = Image.new('RGB', (Wc, Hc))
        fmsk = Image.new('L', (Wc, Hc))
        fdr, fdm = ImageDraw.Draw(frgb), ImageDraw.Draw(fmsk)
        draw_lines(cam, F.front_lines, fdr, fdm)
        for c in F.cards:
            if c['alpha'] > 0.003 and c['scale'] > 0.02:
                ti, fx = draw_card(c, fdr, fdm, ds)
                if c['flip'] > 0.12:
                    card_txt.append((ti, c['x'], c['y'], c['alpha'] * min(1, (c['flip'] - 0.12) * 4), c['scale'], fx))
        front = (np.asarray(frgb.reduce(SS), np.float32) / 255.0, np.asarray(fmsk.reduce(SS), np.float32) / 255.0)
    blk = np.asarray(rgb.reduce(SS), np.float32) / 255.0
    m = np.asarray(msk.reduce(SS), np.float32) / 255.0
    shf = np.asarray(sh.filter(ImageFilter.GaussianBlur(4.5 * S)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
    img = tex.bg * (1 - 0.6 * shf)[:, :, None]
    m3 = m[:, :, None]
    img = img * (1 - m3) + blk * tex.surface
    for (x, y, r, col, inten) in F.sprites:
        add_sprite(img, x, y, r, col, inten)
    mtot = m
    if front is not None:
        fm = front[1][:, :, None]
        img = img * (1 - fm) + front[0] * tex.surface
        mtot = np.maximum(m, front[1])
    for ti, x, y, a, sc, fx in card_txt:
        blit_centered(img, ti, x, y + 1, a, sc, fx)
    for it in F.texts:
        if it[0] == 'scaled':
            _, ti, x, y, a, sc = it
            blit_centered(img, ti, x, y, a, sc)
        elif it[0] == 'scaled_l':
            _, ti, x, y, a, sc = it
            blit_centered(img, ti, x + ti['w'] / S / 2, y, a, sc)
        else:
            ti, x, y, anchor, a = it
            xx, yy = anchor_xy(ti['w'], ti['h'], x * S, y * S, anchor)
            blit_arr(img, ti['rgb'], ti['a'], xx - ti['pad'], yy - ti['pad'], a)
    for ti, x, y, a, sc in F.badges:
        blit_centered(img, ti, x, y, a, sc)
    for ti, x, y, a, sc, sx, rot in F.images:
        blit_centered(img, ti, x, y, a, sc, sx, rot)
    # foreground out-of-focus cubes
    if F.fg is not None:
        q = Image.new('RGB', (W // 4 + 1, H // 4 + 1))
        qm = Image.new('L', (W // 4 + 1, H // 4 + 1))
        fgcam = Cam(yaw=-0.35 + 0.3 * cam.yaw, pitch=0.3, zoom=1.0, target=(0, 0, 0), sx=540, sy=960, roll=cam.roll)
        draw_blocks(fgcam, [F.fg], ImageDraw.Draw(q), ImageDraw.Draw(qm), ss=0.25)
        rad = 3.0 * S
        qa = np.asarray(q.filter(ImageFilter.GaussianBlur(rad)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
        qma = np.asarray(qm.filter(ImageFilter.GaussianBlur(rad)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
        qma = qma[:, :, None] * 0.72
        img = img * (1 - qma) + qa * 0.72 * tex.surface
    # light bands
    for u, st in F.bands:
        b = np.exp(-((tex.band_d - u) / 0.045) ** 2)
        img += (b * st * (0.22 + 0.78 * mtot))[:, :, None] * np.array([1.0, 0.95, 0.84], np.float32) * 0.55
    # impact: chromatic aberration
    k = int(round(6 * S * F.impact))
    if k > 0:
        img[:, :, 0] = np.roll(img[:, :, 0], k, 1)
        img[:, :, 2] = np.roll(img[:, :, 2], -k, 1)
        img[:, :, 1] = np.roll(img[:, :, 1], k // 3, 0)
    img *= tex.paper
    img *= tex.vig
    # HUD tags (screen space)
    for txt, col, since, until in F.hud:
        e = ease_back(since / 0.55, 1.5) if since < 0.55 else 1.0
        a = smooth(0, 0.25, since) * smooth(0, 0.35, until)
        bi = badge_img(txt, 30, col, 'sansb', CREAM, depth=8, radius=12, padx=20, pady=11)
        x = 60 + bi['w'] / S / 2 - 70 * (1 - e)
        blit_centered(img, bi, x, 238, a, 0.85 + 0.15 * e)
    subs.draw(img, t)
    img *= F.fade
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- tools
def keys(tag, times):
    os.makedirs(OUT, exist_ok=True)
    tiles = []
    tw, th = 360, 640
    for t in times:
        t0 = time.time()
        fr = render_frame(t)
        print(f't={t:.2f} {time.time() - t0:.2f}s', flush=True)
        im = Image.fromarray(fr).resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.text((6, 4), f'{t:.2f}', fill=(255, 80, 80))
        # safe-area guides: bottom 25 %, right 15 %
        d.line([(0, th * 0.75), (tw, th * 0.75)], fill=(255, 0, 255))
        d.line([(tw * 0.85, 0), (tw * 0.85, th)], fill=(255, 0, 255))
        tiles.append(im)
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (tw * cols, th * rows), (0, 0, 0))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    p = os.path.join(OUT, f'keys_{tag}.png')
    sheet.save(p)
    print(p)


def frame_png(t):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'frame_{t:.2f}.png')
    Image.fromarray(render_frame(t)).save(p)
    print(p)


def render_chunk(ci):
    os.makedirs(os.path.join(OUT, 'chunks'), exist_ok=True)
    dst = os.path.join(OUT, 'chunks', f'c{ci:03d}.mp4')
    if os.path.exists(dst):
        return ci, 0.0
    tmp = dst + '.part.mp4'
    f0, f1 = ci * CHUNK, min(NFRAMES, (ci + 1) * CHUNK)
    t0 = time.time()
    enc = ['-preset', 'slow', '-crf', '18'] if FINAL else ['-preset', 'veryfast', '-crf', '22']
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', *enc, '-pix_fmt', 'yuv420p', '-threads', '1', tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(render_frame(f / FPS).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, dst)
    return ci, time.time() - t0


def render_all(budget, nproc):
    from multiprocessing import Pool
    n = (NFRAMES + CHUNK - 1) // CHUNK
    todo = [i for i in range(n) if not os.path.exists(os.path.join(OUT, 'chunks', f'c{i:03d}.mp4'))]
    print(f'{W}x{H}: {len(todo)} / {n} chunks to render', flush=True)
    if not todo:
        return
    t0 = time.time()
    done = 0
    pool = Pool(nproc)
    try:
        for ci, dt in pool.imap_unordered(render_chunk, todo):
            done += 1
            print(f'chunk {ci} done in {dt:.0f}s  ({done}/{len(todo)}, {time.time() - t0:.0f}s)', flush=True)
            if time.time() - t0 > budget and done < len(todo):
                print('time budget reached; rerun to resume', flush=True)
                pool.terminate()
                break
    finally:
        pool.close()
        pool.join()
    for f in os.listdir(os.path.join(OUT, 'chunks')):
        if f.endswith('.part.mp4'):
            os.remove(os.path.join(OUT, 'chunks', f))
    left = [i for i in range(n) if not os.path.exists(os.path.join(OUT, 'chunks', f'c{i:03d}.mp4'))]
    print(f'{len(left)} chunks left', flush=True)


def mux():
    n = (NFRAMES + CHUNK - 1) // CHUNK
    lst = os.path.join(OUT, 'chunks.txt')
    with open(lst, 'w') as f:
        for i in range(n):
            f.write(f"file 'chunks/c{i:03d}.mp4'\n")
    wav = os.path.join(HERE, 'build_audio', 'music.wav')
    dst_dir = HERE if FINAL else OUT
    with_music = os.path.join(dst_dir, '最温柔的词.mp4')
    silent = os.path.join(dst_dir, '最温柔的词_无音乐.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                    '-c:v', 'copy', '-an', '-movflags', '+faststart', silent], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest',
                    with_music], check=True)
    print(with_music)
    print(silent)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'keys':
        keys(sys.argv[2], [float(x) for x in sys.argv[3:]])
    elif cmd == 'frame':
        for x in sys.argv[2:]:
            frame_png(float(x))
    elif cmd == 'all':
        render_all(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9, int(os.environ.get('NPROC', os.cpu_count() or 4)))
    elif cmd == 'chunk':
        print(render_chunk(int(sys.argv[2])))
    elif cmd == 'mux':
        mux()
