"""LiveAvatar: real-time streaming talking-head video generation.

Pipeline: TTS/mic PCM (16kHz mono S16LE) → MuseTalk lip-sync inference
→ ``AvatarFrame`` (BGR24) → LiveKit video track, with epoch-based
cancellation (interrupt stops video within one frame) and a
MuseTalk → static-frame → audio-only degradation chain.
"""

__version__ = "0.4.0"
