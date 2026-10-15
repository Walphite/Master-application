"""Frame compositor + keyframe contact sheets + chunked parallel rendering.

  python3 render.py keys 1 2.0 6.0 10.0     # contact sheet for scene 1 at given times
  python3 render.py all                     # render all chunks (resumable) with multiprocessing
  python3 render.py mux                     # concat chunks + music -> 阿白.mp4
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import core
from core import W, H, FPS, NFRAMES, smooth, Painter, Overlay, Sky, glow, blit, draw_hud, draw_subtitle
from scenes import SCENES, SUBS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
CHUNK = 120
_sky = None
_vig = None
_noise = None


def init():
    global _sky, _vig, _noise
    if _sky is None:
        _sky = Sky()
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        _vig = (1 - 0.30 * r2 ** 1.3).clip(0.45, 1)[:, :, None].astype(np.float32)
        # darken bottom band slightly for subtitle legibility
        band = 1 - 0.35 * np.clip((yy - 880) / 160, 0, 1) ** 1.5
        _vig = _vig * band[:, :, None]
        _noise = (np.random.default_rng(3).random((H + 64, W + 64, 3)).astype(np.float32) - 0.5) / 255.0


def scene_alpha(t, a, b):
    fi = 0.0 if a <= 0 else 0.45
    fo = 0.0 if b >= core.DUR else 0.45
    return smooth(a - fi, a + fi, t) * (1 - smooth(b - fo, b + fo, t)) if fi or fo else 1.0


def render_frame(t):
    init()
    P = Painter()
    ov = Overlay()
    _sky.stars(P, t)
    huds = []
    for (a, b, fn, hud) in SCENES:
        if a - 0.5 <= t <= b + 0.5:
            al = scene_alpha(t, a, b)
            if al > 0.001:
                fn(P, ov, t, al, None)
            hal = smooth(a + 0.25, a + 1.0, t) * (1 - smooth(b - 0.6, b - 0.05, t)) if b < core.DUR else smooth(a + 0.25, a + 1.0, t)
            if a == 0.0:
                hal = smooth(0.5, 1.4, t) * (1 - smooth(b - 0.6, b - 0.05, t))
            huds.append((hud, hal))
    L = P.flush()
    for (ti, x, y, al, gl) in ov.items:
        if gl > 0:
            blit(L, ti, x, y, al * gl, add=True)
    draw_subtitle(None, L, SUBS, t)
    g1, g2 = glow(L)
    img = _sky.background(t) + L + 0.85 * g1 + 1.9 * g2
    img = 1.0 - np.exp(-img)
    img *= _vig
    for (ti, x, y, al, gl) in ov.items:
        blit(img, ti, x, y, al)
    for hud, hal in huds:
        draw_hud(img, None, hud, hal)
    draw_subtitle(img, None, SUBS, t)
    fade = smooth(0.0, 0.9, t) * (1 - smooth(core.DUR - 0.9, core.DUR - 0.05, t))
    img *= fade
    ox, oy = int(t * 997) % 64, int(t * 571) % 64
    img += _noise[oy:oy + H, ox:ox + W]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def keys(tag, times):
    os.makedirs(OUT, exist_ok=True)
    tiles = []
    for t in times:
        t0 = time.time()
        fr = render_frame(t)
        print(f't={t:.2f} {time.time() - t0:.2f}s', flush=True)
        tiles.append(Image.fromarray(fr).resize((960, 540), Image.LANCZOS))
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (960 * cols, 540 * rows), (0, 0, 0))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 960, (i // cols) * 540))
    p = os.path.join(OUT, f'keys_{tag}.png')
    sheet.save(p)
    Image.fromarray(render_frame(times[-1])).save(os.path.join(OUT, f'full_{tag}.png')) if len(sys.argv) > 99 else None
    print(p)


def render_chunk(ci):
    init()
    os.makedirs(os.path.join(OUT, 'chunks'), exist_ok=True)
    dst = os.path.join(OUT, 'chunks', f'c{ci:03d}.mp4')
    if os.path.exists(dst):
        return ci, 0.0
    tmp = dst + '.part.mp4'
    f0, f1 = ci * CHUNK, min(NFRAMES, (ci + 1) * CHUNK)
    t0 = time.time()
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15',
           '-pix_fmt', 'yuv420p', '-threads', '1', tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(render_frame(f / FPS).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, dst)
    return ci, time.time() - t0


def render_all(nproc=4):
    from multiprocessing import Pool
    n = (NFRAMES + CHUNK - 1) // CHUNK
    todo = [i for i in range(n) if not os.path.exists(os.path.join(OUT, 'chunks', f'c{i:03d}.mp4'))]
    print(f'{len(todo)} / {n} chunks to render', flush=True)
    with Pool(nproc) as pool:
        for ci, dt in pool.imap_unordered(render_chunk, todo):
            print(f'chunk {ci} done in {dt:.0f}s', flush=True)


def mux():
    n = (NFRAMES + CHUNK - 1) // CHUNK
    lst = os.path.join(OUT, 'chunks.txt')
    with open(lst, 'w') as f:
        for i in range(n):
            f.write(f"file 'chunks/c{i:03d}.mp4'\n")
    wav = os.path.join(OUT, 'music.wav')
    final = os.path.join(os.path.dirname(OUT), '阿白.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wav,
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-movflags', '+faststart', '-shortest', final], check=True)
    print(final)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'keys':
        keys(sys.argv[2], [float(x) for x in sys.argv[3:]])
    elif cmd == 'all':
        render_all(int(os.environ.get('NPROC', os.cpu_count() or 4)))
    elif cmd == 'chunk':
        print(render_chunk(int(sys.argv[2])))
    elif cmd == 'mux':
        mux()
