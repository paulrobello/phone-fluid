# Audio provenance

Generated 2026-10-12 with ElevenLabs text-to-sound-effects (`eleven_text_to_sound_v2`) via the
`elevenlabs` skill, then processed with ffmpeg (mono 44.1 kHz, AAC 128k).

| File | Prompt | Params | Processing |
| --- | --- | --- | --- |
| `Audio/slosh_loop.m4a` | gentle liquid sloshing and gurgling in a glass bottle rocked slowly, soft watery movement, close microphone, warm and low, continuous | 8 s, loop, influence 0.45 | 3× concat → lowpass 5.5 kHz, highpass 60 Hz, +18 dB, limiter at 0.7 → middle copy (seam-safe) |
| `Audio/splash_small.m4a` | small splash of water hitting the inside wall of a glass container, short watery slap, close-up | 1 s, influence 0.6 | lowpass 7 kHz, highpass 80 Hz, +12 dB, 0.15 s fade-out |
| `Audio/splash_big.m4a` | wave of water slamming against the inside of a sealed glass tank, heavy watery impact with a deep slosh afterwards, close-up | 2 s, influence 0.6 | lowpass 7.5 kHz, highpass 60 Hz, −1 dB, 0.3 s fade-out |

Rejected: a brighter loop candidate ("water sloshing inside a small sealed glass jar…", spectral
centroid 3.5 kHz vs 2.8 kHz for the chosen one).
