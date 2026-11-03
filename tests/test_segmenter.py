import numpy as np

from prodtrans.config import VadConfig
from prodtrans.segmenter import ChannelSegmenter
from prodtrans.vad import SileroVad

SR = 16000


def fake_vad(chunk):
    return 1.0 if np.abs(chunk).max() > 0.1 else 0.0


def tone(seconds, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


def run(*parts, cfg=None):
    jobs = []
    seg = ChannelSegmenter(3, cfg or VadConfig(), fake_vad, jobs.append, clock=lambda: 1000.0)
    for p in parts:
        # karmimy kawałkami o nierównej długości, jak z prawdziwego wejścia
        for i in range(0, len(p), 700):
            seg.feed(p[i : i + 700])
    return jobs


def finals(jobs):
    return [j for j in jobs if j.final]


def test_single_utterance():
    jobs = run(silence(0.5), tone(2.0), silence(1.0))
    f = finals(jobs)
    assert len(f) == 1
    assert f[0].channel == 3
    assert abs(len(f[0].audio) / SR - 2.8) <= 0.1
    assert f[0].t_end > f[0].t_start >= 1000.0


def test_partials_emitted():
    jobs = run(silence(0.5), tone(2.0), silence(1.0))
    partials = [j for j in jobs if not j.final]
    assert len(partials) >= 1
    assert all(p.segment_id == finals(jobs)[0].segment_id for p in partials)


def test_short_blip_dropped():
    jobs = run(silence(0.5), tone(0.1), silence(1.0))
    assert finals(jobs) == []


def test_long_speech_split_without_loss():
    jobs = run(tone(30.0), silence(1.0))
    f = finals(jobs)
    assert len(f) == 3
    assert len({j.segment_id for j in f}) == 3
    assert sum(len(j.audio) for j in f) / SR >= 30.0


def test_noise_below_threshold_ignored():
    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.05, 0.05, 10 * SR)).astype(np.float32)
    assert run(noise) == []


def test_silero_loads():
    assert SileroVad().prob(np.zeros(512, np.float32)) < 0.5
