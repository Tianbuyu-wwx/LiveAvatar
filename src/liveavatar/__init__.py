# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""LiveAvatar: real-time streaming talking-head video generation.

Pipeline: TTS/mic PCM (16kHz mono S16LE) → MuseTalk lip-sync inference
→ ``AvatarFrame`` (BGR24) → video sink (self-developed WS transport), with epoch-based
cancellation (interrupt stops video within one frame) and a
MuseTalk → static-frame → audio-only degradation chain.
"""

__version__ = "0.4.0"
