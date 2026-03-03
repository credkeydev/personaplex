# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import collections
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import re
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
import hashlib
import pickle
import threading



# CredKey: Directory for prompt_id disk loading
PROMPT_DIR = Path("/opt/credkey/moshi-prompts")

# CredKey: Token caching for fast handshakes (prompt tokenization is CPU-bound)
_PROMPT_TOKEN_CACHE = {}  # {cache_key: (tokens, token_count)}
_PROMPT_TOKEN_CACHE_LOCK = threading.Lock()
_PROMPT_TOKEN_CACHE_DIR = PROMPT_DIR / ".cache"

def _credkey_get_prompt_tokens_cached(text_prompt, prompt_id, text_tokenizer, wrap_fn, clog):
    if not text_prompt:
        return None

    prompt_hash = hashlib.sha256(text_prompt.encode("utf-8")).hexdigest()
    cache_key = f"{prompt_id or 'inline'}:{prompt_hash[:16]}"

    v = _PROMPT_TOKEN_CACHE.get(cache_key)
    if v is not None:
        tokens, token_count = v
        try: clog.log("info", f"TOKEN_CACHE_HIT mem prompt_id={prompt_id} sha={prompt_hash[:8]} tokens={token_count}")
        except Exception: pass
        return tokens

    with _PROMPT_TOKEN_CACHE_LOCK:
        v = _PROMPT_TOKEN_CACHE.get(cache_key)
        if v is not None:
            return v[0]

        if prompt_id:
            try:
                _PROMPT_TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
                cache_file = _PROMPT_TOKEN_CACHE_DIR / f"{prompt_id}.{prompt_hash[:16]}.tokens"
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        tokens = pickle.load(f)
                    token_count = len(tokens) if hasattr(tokens, "__len__") else "?"
                    _PROMPT_TOKEN_CACHE[cache_key] = (tokens, token_count)
                    try: clog.log("info", f"TOKEN_CACHE_HIT disk prompt_id={prompt_id} sha={prompt_hash[:8]} tokens={token_count}")
                    except Exception: pass
                    return tokens
            except Exception as e:
                try: clog.log("warning", f"TOKEN_CACHE disk read failed {cache_key}: {e}")
                except Exception: pass

        t0 = time.time()
        tokens = text_tokenizer.encode(wrap_fn(text_prompt))
        ms = int((time.time() - t0) * 1000)
        token_count = len(tokens) if hasattr(tokens, "__len__") else "?"
        _PROMPT_TOKEN_CACHE[cache_key] = (tokens, token_count)

        if prompt_id:
            try:
                _PROMPT_TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
                cache_file = _PROMPT_TOKEN_CACHE_DIR / f"{prompt_id}.{prompt_hash[:16]}.tokens"
                with open(cache_file, "wb") as f:
                    pickle.dump(tokens, f)
            except Exception as e:
                try: clog.log("warning", f"TOKEN_CACHE disk write failed {cache_key}: {e}")
                except Exception: pass

        try: clog.log("info", f"TOKEN_CACHE_MISS tokenized prompt_id={prompt_id} bytes={len(text_prompt)} sha={prompt_hash[:8]} tokenize_ms={ms} tokens={token_count}")
        except Exception: pass

        return tokens

logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# Phase 2: Inline text injection — force script tokens into Moshi generation
CREDKEY_INLINE_TEXT = os.environ.get("CREDKEY_INLINE_TEXT", "0") == "1"
CREDKEY_INLINE_EOS = os.environ.get("CREDKEY_INLINE_EOS", "0") == "1"       # append EOS after line (OFF by default — corrupts model state)
CREDKEY_IDLE_PAD = os.environ.get("CREDKEY_IDLE_PAD", "1") == "1"           # force PAD when no injection queued (suppresses hallucination)
CREDKEY_INLINE_PROSODY = os.environ.get("CREDKEY_INLINE_PROSODY", "0") == "1"
CREDKEY_POST_PAD_FRAMES = int(os.environ.get("CREDKEY_POST_PAD_FRAMES", "0"))
CREDKEY_INJECT_STRIDE = int(os.environ.get("CREDKEY_INJECT_STRIDE", "1"))  # frames between token injections  # extra PAD frames after line ends
CREDKEY_MODEL_GUIDED_PACE = os.environ.get("CREDKEY_MODEL_GUIDED_PACE", "0") == "1"
CREDKEY_PACE_MAX_HOLD = int(os.environ.get("CREDKEY_PACE_MAX_HOLD", "8"))
CREDKEY_PACE_MIN_HOLD = int(os.environ.get("CREDKEY_PACE_MIN_HOLD", "1"))
INLINE_TOKEN_CAP = 250  # hard cap: drop tokens beyond this
TEXT_PAD_ID = 3   # <pad> — blank/no-text token
# Prosody: pause frames after punctuation (only when CREDKEY_INLINE_PROSODY=1)
_PUNCT_PAUSE = {261: 2, 263: 4, 330: 4, 430: 4}  # comma=2, period/question/excl=4


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self._pending_inline_text = None           # Ramp B: inline text from shim (kind==2)
        self._inline_text_tokens = collections.deque()  # tokenized queue, fed one-per-step
        self._inline_pause_frames = 0             # prosody: remaining pause frames after punctuation
        self._inline_inject_active = False        # True while queue is draining
        self._inline_postpad_remaining = 0        # post-line PAD hold frames
        self._inline_idle_pad_on = False          # True when idle PAD suppression is active (for logging)
        self._inline_forced_log = None            # comparison: forced token accumulator
        self._inline_emitted_log = None           # comparison: emitted token accumulator
        self._inline_frames_in_inject = 0         # for rate-limited logging
        self._inline_hold_frames = 0                # model-guided: PAD frames since last content token
        self._inline_awaiting_model_advance = False  # model-guided: True = holding on PAD, watching model signal
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path
                
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        # CredKey: Load text_prompt from disk via prompt_id (if provided)
        text_prompt = request.query.get("text_prompt", "")
        prompt_id = request.query.get("prompt_id")
        if prompt_id:
            if not re.match(r"^[A-Za-z0-9_-]{1,64}$", prompt_id):
                try: clog.log("error", f"Invalid prompt_id format: {prompt_id}")
                except Exception: pass
            else:
                prompt_file = PROMPT_DIR / f"{prompt_id}.txt"
                try:
                    if prompt_file.exists() and prompt_file.is_file():
                        text_prompt = prompt_file.read_text(encoding="utf-8")
                        try: clog.log("info", f"Loaded prompt_id={prompt_id} ({len(text_prompt)} bytes)")
                        except Exception: pass
                    else:
                        try: clog.log("warning", f"prompt_id={prompt_id} file not found: {prompt_file}")
                        except Exception: pass
                except Exception as e:
                    try: clog.log("error", f"Failed reading prompt_id={prompt_id}: {e}")
                    except Exception: pass

        self.lm_gen.text_prompt_tokens = await asyncio.to_thread(_credkey_get_prompt_tokens_cached, text_prompt, prompt_id, self.text_tokenizer, wrap_with_system_tags, clog)
        self.lm_gen.text_prompt = text_prompt
        seed = int(request["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            clog.log("info", "P0_DEBUG recv_loop STARTED")
            recv_frame_count = 0
            try:
                async for message in ws:
                    recv_frame_count += 1
                    if recv_frame_count == 1:
                        clog.log("info", f"P0_DEBUG recv_loop FIRST_MESSAGE type={message.type}")
                    if recv_frame_count % 200 == 0:
                        clog.log("info", f"P0_DEBUG recv_loop frames={recv_frame_count}")
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    elif kind == 2:  # inline text from shim (Ramp B)
                        text = message[1:].decode("utf-8", errors="replace")
                        clog.log("info", f"INLINE_TEXT_RECEIVED len={len(text)}: {text[:100]}")
                        self._pending_inline_text = text
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", f"P0_DEBUG recv_loop ENDED frames={recv_frame_count}")
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None
            clog.log("info", "P0_DEBUG opus_loop STARTED")
            opus_frames_processed = 0
            opus_writes_to_writer = 0
            opus_text_tokens = 0
            opus_loop_start_time = time.time()

            # [CredKey] Phase 1.6: Generation-Level Gate
            CREDKEY_GEN_GATE = os.environ.get("CREDKEY_GEN_GATE", "0") == "1"
            CREDKEY_GEN_GATE_N = int(os.environ.get("CREDKEY_GEN_GATE_N", "25"))
            gen_gate_released = not CREDKEY_GEN_GATE
            skipped_gen_steps = 0
            if CREDKEY_GEN_GATE:
                clog.log("info", f"[CredKey] GEN_GATE_ON n_required={CREDKEY_GEN_GATE_N}")

            while True:
                if close:
                    clog.log("info", f"P0_DEBUG opus_loop ENDED close=True frames_processed={opus_frames_processed} opus_writes={opus_writes_to_writer} text_tokens={opus_text_tokens}")
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm is None:
                    await asyncio.sleep(0.01)
                    continue
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    opus_frames_processed += 1
                    if opus_frames_processed == 1:
                        clog.log("info", f"P0_DEBUG opus_loop FIRST_FRAME elapsed={time.time()-opus_loop_start_time:.3f}s")
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)

                    # [CredKey] Phase 1.6: Generation Gate - prevent step() during gate window
                    if not gen_gate_released:
                        if opus_frames_processed >= CREDKEY_GEN_GATE_N:
                            gen_gate_released = True
                            elapsed_ms = (time.time() - opus_loop_start_time) * 1000
                            clog.log("info",
                                f"[CredKey] GEN_GATE_RELEASED inbound_frames={opus_frames_processed} "
                                f"ms_since_start={elapsed_ms:.1f} skipped_steps={skipped_gen_steps}")
                        else:
                            skipped_gen_steps += 1
                            continue  # Skip generation entirely - no step(), no decode(), no emit

                    for c in range(codes.shape[-1]):
                        codes_slice = codes[:, :, c: c + 1]

                        # ── Phase 2 §1: Consume pending text → tokenize → queue ──
                        if CREDKEY_INLINE_TEXT and self._pending_inline_text is not None:
                            try:
                                inline_text = self._pending_inline_text
                                self._pending_inline_text = None
                                new_tokens = self.text_tokenizer.encode(inline_text)
                                if len(new_tokens) > INLINE_TOKEN_CAP:
                                    clog.log("warning",
                                        f"INLINE_TEXT_TOKEN_CAP "
                                        f"{len(new_tokens)} -> {INLINE_TOKEN_CAP}")
                                    new_tokens = new_tokens[:INLINE_TOKEN_CAP]
                                eos_appended = 0
                                if CREDKEY_INLINE_EOS:
                                    eos = getattr(self.text_tokenizer, 'eos_id', None)
                                    eos_val = eos() if callable(eos) else eos
                                    if eos_val is not None and eos_val >= 0:
                                        new_tokens.append(eos_val)
                                        eos_appended = 1
                                self._inline_text_tokens.clear()
                                self._inline_text_tokens.extend(new_tokens)
                                self._inline_pause_frames = 0
                                self._inline_postpad_remaining = 0
                                self._inline_inject_active = True
                                self._inline_forced_log = []
                                self._inline_emitted_log = []
                                self._inline_frames_in_inject = 0
                                self._inline_hold_frames = 0
                                self._inline_awaiting_model_advance = False
                                if self._inline_idle_pad_on:
                                    clog.log("info", "INLINE_IDLE_PAD_OFF")
                                    self._inline_idle_pad_on = False
                                clog.log("info",
                                    f"INLINE_TEXT_TOKENIZED tokens={len(new_tokens)} "
                                    f"eos_enabled={1 if CREDKEY_INLINE_EOS else 0} "
                                    f"eos_appended={eos_appended} "
                                    f"text={inline_text[:80]}")
                            except Exception as e:
                                clog.log("error", f"INLINE_TEXT_TOKENIZE_FAILED: {e}")

                        # ── Phase 2 §2: Pick forced text_token for this frame ──
                        forced_text_token = None

                        # A) Active injection: force content or PAD tokens
                        if CREDKEY_INLINE_TEXT and self._inline_inject_active:
                            self._inline_frames_in_inject += 1

                            if self._inline_pause_frames > 0:
                                forced_text_token = TEXT_PAD_ID
                                self._inline_pause_frames -= 1
                            elif self._inline_text_tokens:
                                # Phase 3: model-guided pacing or legacy stride
                                if CREDKEY_MODEL_GUIDED_PACE:
                                    if not self._inline_awaiting_model_advance:
                                        # Emit next content token immediately
                                        forced_text_token = self._inline_text_tokens.popleft()
                                        if self._inline_forced_log is not None and forced_text_token not in (0, 1, 2, 3):
                                            self._inline_forced_log.append(forced_text_token)
                                        if CREDKEY_INLINE_PROSODY and forced_text_token in _PUNCT_PAUSE:
                                            self._inline_pause_frames = _PUNCT_PAUSE[forced_text_token]
                                        self._inline_awaiting_model_advance = True
                                        self._inline_hold_frames = 0
                                        if self._inline_frames_in_inject == 1:
                                            clog.log("info",
                                                f"INLINE_INJECT_START first_tok={forced_text_token} "
                                                f"mode=model_guided max_hold={CREDKEY_PACE_MAX_HOLD} "
                                                f"min_hold={CREDKEY_PACE_MIN_HOLD} "
                                                f"qlen={len(self._inline_text_tokens)}")
                                    else:
                                        # Hold on PAD, check model signal
                                        forced_text_token = TEXT_PAD_ID
                                        self._inline_hold_frames += 1
                                        last_sampled = self.lm_gen._last_sampled_text_token  # Python int, no GPU sync
                                        model_wants_advance = (last_sampled is not None and last_sampled != TEXT_PAD_ID)
                                        past_min = self._inline_hold_frames >= CREDKEY_PACE_MIN_HOLD
                                        past_max = self._inline_hold_frames >= CREDKEY_PACE_MAX_HOLD
                                        if (model_wants_advance and past_min) or past_max:
                                            if past_max and not model_wants_advance:
                                                clog.log("warning", f"PACE_MAX_HOLD_HIT hold={self._inline_hold_frames}")
                                            self._inline_awaiting_model_advance = False  # next frame pops next token
                                else:
                                    # Legacy stride mode
                                    if CREDKEY_INJECT_STRIDE <= 1 or self._inline_frames_in_inject == 1 or (self._inline_frames_in_inject % CREDKEY_INJECT_STRIDE) == 1:
                                        forced_text_token = self._inline_text_tokens.popleft()
                                        if self._inline_forced_log is not None and forced_text_token not in (0, 1, 2, 3):
                                            self._inline_forced_log.append(forced_text_token)
                                        if CREDKEY_INLINE_PROSODY and forced_text_token in _PUNCT_PAUSE:
                                            self._inline_pause_frames = _PUNCT_PAUSE[forced_text_token]
                                        if self._inline_frames_in_inject == 1:
                                            clog.log("info",
                                                f"INLINE_INJECT_START first_tok={forced_text_token} "
                                                f"stride={CREDKEY_INJECT_STRIDE} "
                                                f"qlen={len(self._inline_text_tokens)}")
                                    else:
                                        forced_text_token = TEXT_PAD_ID  # PAD on off-stride frames
                            else:
                                # Queue drained — injection complete
                                self._inline_inject_active = False
                                # Start post-line PAD hold
                                if CREDKEY_POST_PAD_FRAMES > 0:
                                    self._inline_postpad_remaining = CREDKEY_POST_PAD_FRAMES
                                    clog.log("info", f"INLINE_POSTPAD_START frames={CREDKEY_POST_PAD_FRAMES}")
                                # Log comparison
                                if self._inline_forced_log is not None:
                                    forced_text = self.text_tokenizer.decode(self._inline_forced_log)
                                    emitted_text = self.text_tokenizer.decode(
                                        self._inline_emitted_log or [])
                                    match_pct = 0
                                    if forced_text:
                                        common = sum(1 for a, b in zip(forced_text, emitted_text) if a == b)
                                        match_pct = int(100 * common / len(forced_text))
                                    clog.log("info",
                                        f"INLINE_INJECT_DONE "
                                        f"forced={len(self._inline_forced_log)}tok "
                                        f"emitted={len(self._inline_emitted_log or [])}tok "
                                        f"frames={self._inline_frames_in_inject} "
                                        f"char_match={match_pct}%")
                                    clog.log("info", f"INLINE_FORCED:  {forced_text[:150]}")
                                    clog.log("info", f"INLINE_EMITTED: {emitted_text[:150]}")
                                self._inline_forced_log = None
                                self._inline_emitted_log = None
                                # Fall through to idle pad below

                        # B) Post-line PAD hold (brief silence after scripted line)
                        if forced_text_token is None and self._inline_postpad_remaining > 0:
                            forced_text_token = TEXT_PAD_ID
                            self._inline_postpad_remaining -= 1
                            if self._inline_postpad_remaining == 0:
                                clog.log("info", "INLINE_POSTPAD_END")

                        # C) Idle PAD suppression: prevent hallucination between injections
                        if forced_text_token is None and CREDKEY_INLINE_TEXT and CREDKEY_IDLE_PAD:
                            if not self._inline_inject_active and not self._inline_text_tokens:
                                forced_text_token = TEXT_PAD_ID
                                if not self._inline_idle_pad_on:
                                    self._inline_idle_pad_on = True
                                    clog.log("info", "INLINE_IDLE_PAD_ON")

                        # ── Phase 2 §3: Call lm_gen.step — always real codes ──
                        if forced_text_token is not None:
                            tokens = self.lm_gen.step(
                                codes_slice,
                                text_token=forced_text_token,
                            )
                        else:
                            tokens = self.lm_gen.step(codes_slice)
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        pcm_data = main_pcm[0, 0].numpy()
                        opus_writer.append_pcm(pcm_data)
                        opus_writes_to_writer += 1
                        if opus_writes_to_writer == 1:
                            clog.log("info", f"P0_DEBUG FIRST_OPUS_WRITE pcm_samples={len(pcm_data)} elapsed={time.time()-opus_loop_start_time:.3f}s")
                        if opus_writes_to_writer % 100 == 0:
                            clog.log("info", f"P0_DEBUG opus_writes={opus_writes_to_writer} frames_processed={opus_frames_processed}")
                        text_token = tokens[0, 0, 0].item()

                        # ── Phase 2 §4: Track emitted tokens during injection ──
                        if CREDKEY_INLINE_TEXT and self._inline_emitted_log is not None:
                            if text_token not in (0, 1, 2, 3):
                                self._inline_emitted_log.append(text_token)

                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

        async def send_loop():
            clog.log("info", "P0_DEBUG send_loop STARTED")
            send_count = 0
            send_bytes_total = 0
            send_empty_polls = 0
            send_start_time = time.time()
            last_empty_log = time.time()
            while True:
                if close:
                    clog.log("info", f"P0_DEBUG send_loop ENDED close=True sends={send_count} bytes={send_bytes_total} empty_polls={send_empty_polls}")
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    send_count += 1
                    send_bytes_total += len(msg)
                    if send_count == 1:
                        clog.log("info", f"P0_DEBUG FIRST_SEND_TO_SHIM bytes={len(msg)} elapsed={time.time()-send_start_time:.3f}s")
                    if send_count % 100 == 0:
                        clog.log("info", f"P0_DEBUG send_loop sends={send_count} bytes_total={send_bytes_total}")
                    await ws.send_bytes(b"\x01" + msg)
                else:
                    send_empty_polls += 1
                    now = time.time()
                    if now - last_empty_log >= 2.0:
                        clog.log("info", f"P0_DEBUG send_loop EMPTY_OPUS_BUFFER empty_polls={send_empty_polls} sends={send_count} elapsed={now-send_start_time:.3f}s")
                        last_empty_log = now

        clog.log("info", "accepted connection")
        if CREDKEY_INLINE_TEXT:
            clog.log("info", f"INLINE_TEXT_ON token_cap={INLINE_TOKEN_CAP}")
        if len(text_prompt) > 0:
            clog.log("info", f"text prompt: {text_prompt[:200] + '...' if len(text_prompt) > 200 else text_prompt}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # FIX-C: Reset per-session injection state to prevent cross-session contamination
            self._pending_inline_text = None
            self._inline_text_tokens.clear()
            self._inline_pause_frames = 0
            self._inline_inject_active = False
            self._inline_postpad_remaining = 0
            self._inline_idle_pad_on = False
            self._inline_forced_log = None
            self._inline_emitted_log = None
            self._inline_frames_in_inject = 0
            self._inline_hold_frames = 0
            self._inline_awaiting_model_advance = False
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # FIX-B: Bail early if shim already disconnected while we waited on the lock
            # Saves 31-43 seconds of wasted GPU compute per dead connection
            if ws.closed:
                clog.log("warning", "CLIENT_GONE_BEFORE_PROMPTS ws.closed=True -- skipping system prompts")
                return ws
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                clog.log("info", f"P0_DEBUG ws_closed={ws.closed} close_flag={close}")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                # P0 DEBUG: Add done callbacks to catch silent task deaths
                task_names = ['recv_loop', 'opus_loop', 'send_loop']
                for task, name in zip(tasks, task_names):
                    def make_cb(n):
                        def cb(t):
                            if t.cancelled():
                                clog.log("info", f"P0_DEBUG TASK_CANCELLED {n}")
                            elif t.exception():
                                clog.log("error", f"P0_DEBUG TASK_DIED {n} exception={t.exception()}")
                            else:
                                clog.log("info", f"P0_DEBUG TASK_FINISHED {n}")
                        return cb
                    task.add_done_callback(make_cb(name))

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
    )
    logger.info("warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
