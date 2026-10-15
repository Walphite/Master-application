"""Score for 《数学里最温柔的一个词》: numpy-synthesised piano.

Sparse single notes with lots of air for the first half, chords and a bass line
in the climax (32.6-51.4 s), and a long final chord that fades out.
"""
import os
import wave

import numpy as np

SR = 44100
DUR = 72.0


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def piano(freq, dur, vel, bright=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    B = 0.0004
    for k in range(1, 10):
        fk = freq * k * np.sqrt(1 + B * k * k)
        if fk > SR / 2.2:
            break
        amp = 1.0 / k ** (1.9 - 0.4 * bright)
        decay = 0.9 + 0.75 * k + freq / 500
        out += amp * np.exp(-decay * t * 0.5) * np.sin(2 * np.pi * fk * t + 0.37 * k)
    # slow second stage of the decay (the "ring" of a held piano note)
    out = out * (0.75 + 0.25 * np.exp(-t * 0.4))
    att = np.minimum(t / 0.005, 1.0)
    rel = np.clip((dur - t) / 0.25, 0, 1)
    hammer = np.exp(-t * 70) * 0.12 * np.sin(2 * np.pi * freq * 5.03 * t)
    return (out + hammer) * att * rel * vel


# --------------------------------------------------------------- score (time, midi, length, velocity)
NOTES = []


def N(t, m, d=2.8, v=0.30):
    NOTES.append((t, m, d, v))


def chord(t, ms, d=3.0, v=0.17, spread=0.025):
    for i, m in enumerate(ms):
        N(t + i * spread, m, d, v)


# 0-4 hook: one struck low A with a high answer
N(0.00, 45, 4.5, 0.42); N(0.00, 57, 4.0, 0.30); N(0.02, 76, 3.0, 0.24)
N(1.60, 72, 2.6, 0.22); N(2.55, 69, 3.0, 0.24)
# 4-16 shattered: single notes, air between them
for t, m in [(4.1, 64), (5.3, 67), (6.2, 72), (6.9, 71), (7.6, 69), (8.8, 67), (9.4, 64),
             (10.3, 76), (11.8, 79), (13.0, 72), (14.4, 71), (15.2, 67)]:
    N(t, m, 2.6, 0.22 + 0.04 * (m > 74))
N(4.1, 45, 4.0, 0.20); N(10.3, 52, 3.0, 0.16)
# 16-25.6 Riemann: steady counting notes that get unsettled, a low cluster at the stamp
for j in range(12):
    N(16.25 + j * 0.19, [57, 60, 64, 67][j % 4] + 12, 0.9, 0.12)
N(18.7, 63, 1.6, 0.16); N(19.35, 46, 2.5, 0.22)
N(20.05, 33, 4.0, 0.42); N(20.05, 40, 4.0, 0.28); N(20.07, 46, 3.5, 0.18)
for t, m in [(20.4, 69), (21.3, 67), (22.6, 64), (23.8, 62), (24.7, 60)]:
    N(t, m, 2.4, 0.18)
# 25.6-32.6 the turn: quiet, then three light notes while the bills sort
for t, m, v in [(25.9, 72, 0.22), (27.4, 76, 0.2), (28.6, 74, 0.18), (29.55, 79, 0.15), (29.8, 81, 0.15),
                (30.05, 84, 0.15), (30.95, 72, 0.22), (31.8, 76, 0.2)]:
    N(t, m, 2.6, v)
N(25.9, 41, 5.0, 0.18)
# 32.6-51.4 climax: F - G - Am - C (twice), bass, broken chords and a top line
PROG = [(41, [53, 57, 60, 64]), (43, [55, 59, 62, 67]), (45, [57, 60, 64, 69]), (48, [55, 60, 64, 67])]
LEN = (51.4 - 32.6) / 8
TOP = [72, 74, 76, 79, 77, 74, 76, 72]
for i in range(8):
    t0 = 32.6 + i * LEN
    bass, ch = PROG[i % 4]
    N(t0, bass - 12, LEN + 1.5, 0.30)
    N(t0, bass, LEN + 1.2, 0.20)
    chord(t0 + 0.02, ch, LEN + 0.8, 0.12)
    for k, m in enumerate([ch[0], ch[2], ch[3] + 12 - 12, ch[1] + 12]):
        N(t0 + LEN * (0.25 + 0.17 * k), m, 1.8, 0.10)
    N(t0 + 0.01, TOP[i], LEN + 0.6, 0.20)
    N(t0 + LEN * 0.62, TOP[i] - 3 if i % 2 else TOP[i] + 2, 1.6, 0.14)
N(44.3, 29, 4.0, 0.30)  # the layers slam together
N(49.0, 81, 3.0, 0.18)
# 51.4-62.6 measure zero: thin again; a soft chord when the answer lands on 1
for t, m in [(51.7, 69), (52.6, 72), (53.3, 74)]:
    N(t, m, 2.4, 0.18)
chord(54.0, [45, 57, 64, 69, 72], 4.5, 0.15)
for t, m in [(55.6, 76), (56.8, 74), (57.8, 72), (58.9, 69)]:
    N(t, m, 2.6, 0.18)
N(59.5, 88, 2.8, 0.10)  # the fragment goes out
chord(59.9, [41, 53, 60, 64], 4.0, 0.12)
N(61.2, 67, 2.4, 0.16)
# 62.6-72 ending: chords that settle on one long chord and fade
chord(62.8, [45, 57, 60, 64, 72], 3.0, 0.14)
chord(64.1, [41, 53, 57, 64, 69, 76], 3.5, 0.16)
N(64.1, 29, 4.0, 0.25)
chord(66.3, [48, 55, 64, 67, 72], 3.2, 0.13)
N(67.6, 79, 2.5, 0.14)
chord(69.2, [36, 48, 55, 62, 64, 67, 71, 76], 8.0, 0.15, spread=0.06)


def build():
    total = int((DUR + 3) * SR)
    L = np.zeros(total)
    R = np.zeros(total)
    rng = np.random.default_rng(3)
    for (t, m, d, v) in NOTES:
        sig = piano(hz(m), d, v * (0.92 + 0.16 * rng.random()), bright=1.0 if m > 60 else 0.6)
        i = int(max(t + rng.normal(0, 0.003), 0) * SR)
        j = min(total, i + len(sig))
        pan = np.clip(0.5 + (m - 60) / 60, 0.2, 0.8)
        L[i:j] += sig[:j - i] * np.sqrt(1 - pan)
        R[i:j] += sig[:j - i] * np.sqrt(pan)
    irn = int(3.2 * SR)
    ti = np.arange(irn) / SR
    out = []
    for ch, seed in ((L, 1), (R, 2)):
        r = np.random.default_rng(seed)
        ir = r.normal(0, 1, irn) * np.exp(-ti / 0.95)
        ir = np.convolve(ir, np.ones(8) / 8, mode='same')
        ir[: int(0.015 * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        nfft = 1 << int(np.ceil(np.log2(len(ch) + irn)))
        wet = np.fft.irfft(np.fft.rfft(ch, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(ch)]
        out.append(0.72 * ch + 0.5 * wet)
    st = np.stack(out, 1)[: int(DUR * SR)]
    t = np.arange(len(st)) / SR
    env = np.clip((DUR - t) / 3.2, 0, 1) ** 1.5
    st *= env[:, None]
    st *= 0.72 / np.abs(st).max()
    return st


if __name__ == '__main__':
    st = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_audio', 'music.wav')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with wave.open(out, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    print(out, st.shape[0] / SR)
