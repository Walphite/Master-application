"""Gentle piano arpeggio, D – A – Bm – G loop, with simple convolution reverb."""
import os
import wave

import numpy as np

SR = 44100
DUR = 92.0
CHORD_LEN = 2.875           # 4 chords = 11.5 s, 8 loops = 92 s
STEP = CHORD_LEN / 8        # eighth notes


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def piano(freq, dur, vel):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    B = 0.0004  # slight inharmonicity
    for k in range(1, 9):
        fk = freq * k * np.sqrt(1 + B * k * k)
        if fk > SR / 2.2:
            break
        amp = (1.0 / k ** 1.6) * (1.0 if k > 1 else 1.0)
        decay = 1.6 + 0.9 * k + freq / 400
        out += amp * np.exp(-decay * t * 0.55) * np.sin(2 * np.pi * fk * t + 0.3 * k)
    att = np.minimum(t / 0.006, 1.0)
    rel = np.minimum((dur - t) / 0.08, 1.0).clip(0, 1)
    hammer = np.exp(-t * 60) * 0.15 * np.sin(2 * np.pi * freq * 6.1 * t)
    return (out + hammer) * att * rel * vel


# chords as MIDI: root(bass), and the arpeggio voicing
CHORDS = [
    (38, [50, 57, 62, 66, 69, 66, 62, 57]),   # D
    (45, [45, 52, 57, 61, 64, 61, 57, 52]),   # A
    (47, [47, 54, 59, 62, 66, 62, 59, 54]),   # Bm
    (43, [43, 50, 55, 59, 62, 59, 55, 50]),   # G
]
# a soft top line that enters in the middle loops
MELODY = [74, 73, 71, 67]


def build():
    total = int(DUR * SR) + SR * 4
    L = np.zeros(total)
    R = np.zeros(total)
    rng = np.random.default_rng(5)

    def add(sig, start, pan):
        i = max(0, int(start * SR))
        j = min(total, i + len(sig))
        L[i:j] += sig[:j - i] * np.sqrt(1 - pan)
        R[i:j] += sig[:j - i] * np.sqrt(pan)

    nchords = int(round(DUR / CHORD_LEN))
    for c in range(nchords):
        loop = c // 4
        bass, arp = CHORDS[c % 4]
        t0 = c * CHORD_LEN
        add(piano(midi_hz(bass), CHORD_LEN + 1.5, 0.30), t0, 0.42)
        for k, m in enumerate(arp):
            vel = (0.23 if k == 0 else 0.17) * (0.9 + 0.2 * rng.random())
            add(piano(midi_hz(m), 2.6, vel), t0 + k * STEP + rng.normal(0, 0.004), 0.30 + 0.45 * (m - 45) / 25)
        if 2 <= loop <= 6:
            mv = 0.13 if loop in (2, 6) else 0.16
            add(piano(midi_hz(MELODY[c % 4]), CHORD_LEN + 1.0, mv), t0 + STEP * 0.02, 0.6)
            if c % 2 == 1:
                add(piano(midi_hz(MELODY[c % 4] - 3 if c % 4 != 3 else 69), 1.8, mv * 0.7), t0 + 5 * STEP, 0.62)

    # reverb: decaying filtered noise impulse response (different per channel)
    irn = int(2.8 * SR)
    ti = np.arange(irn) / SR
    out = []
    for ch, seed in ((L, 1), (R, 2)):
        r = np.random.default_rng(seed)
        ir = r.normal(0, 1, irn) * np.exp(-ti / 0.75)
        ir = np.convolve(ir, np.ones(6) / 6, mode='same')
        ir[: int(0.012 * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        nfft = 1 << int(np.ceil(np.log2(len(ch) + irn)))
        wet = np.fft.irfft(np.fft.rfft(ch, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(ch)]
        out.append(0.78 * ch + 0.42 * wet)
    st = np.stack(out, 1)[: int(DUR * SR)]
    t = np.arange(len(st)) / SR
    env = np.minimum(t / 2.5, 1.0) * np.clip((DUR - t) / 4.5, 0, 1)
    st *= env[:, None]
    st *= 0.7 / np.abs(st).max()
    return st


if __name__ == '__main__':
    st = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'music.wav')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with wave.open(out, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    print(out, st.shape[0] / SR)
