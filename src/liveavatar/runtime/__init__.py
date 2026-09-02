# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Session runtime hub — orchestrates VAD / ASR / TTS / Avatar spokes.

Migrated from WisdomVII ``RealtimeWorker``. The hub owns the epoch clock:
every spoke (VAD, ASR, TTS, avatar adapter) is advanced/cancelled together
so a barge-in stops audio AND video within one frame.
"""
