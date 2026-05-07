"""
Enigma Engine - GUI Media & Voice Logic
==========================================

Mixin providing media rendering (images, GIFs, videos, URLs)
and voice I/O (TTS output + speech-to-text input).
Split from gui_logic.py to keep files under 800 lines.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from tkinter import filedialog

from enigma_engine.gui.widgets import C_RED, C_SURFACE, C_TEXT_DIM

logger = logging.getLogger(__name__)


class LogicMediaMixin:
    """Mixin providing media rendering and voice I/O.

    Expects the host class to have chat_display, mic_btn,
    _chat_images, _chat_gif_animations, _chat_media_refs,
    voice_enabled, user_name, etc.
    """

    # ================================================================
    # Logic - Chat media rendering
    # ================================================================

    def _process_media_in_text(self, text: str, tag: str = "assistant"):
        """Scan completed text for media refs and render inline.

        Called after typewriter finishes. Detects image paths,
        GIF paths, video paths, and URLs. Renders images/GIFs
        inline and makes URLs clickable.
        """
        from enigma_engine.gui.media import (
            detect_media_refs, detect_urls,
        )
        # Render media references found in the text
        refs = detect_media_refs(text)
        for ref in refs:
            if ref["type"] == "image":
                self._insert_media(ref["path"], ref["source"])
            elif ref["type"] == "gif":
                self._insert_gif(ref["path"], ref["source"])
            elif ref["type"] == "video":
                self._insert_video_thumbnail(
                    ref["path"], ref["source"])

        # Make plain URLs clickable (non-media URLs)
        urls = detect_urls(text)
        media_urls = {r["path"] for r in refs if r["source"] == "url"}
        for url in urls:
            if url not in media_urls:
                self._make_url_clickable(url)

    def _insert_media(self, path_str: str, source: str = "file"):
        """Insert an image inline in the chat display.

        URL downloads run in a background thread to avoid freezing
        the GUI (up to 10s timeout per image).
        """
        from enigma_engine.gui.media import (
            load_chat_image, load_url_image, get_media_chat_width,
        )
        try:
            chat_w = self.chat_display.winfo_width()
            max_w = get_media_chat_width(chat_w)
        except Exception:
            max_w = 400

        if source == "url":
            # Network download — offload to thread to avoid GUI freeze
            def _download_and_insert():
                photo = load_url_image(path_str, max_width=max_w)
                self.after(0, lambda: self._insert_media_widget(
                    path_str, photo))

            threading.Thread(
                target=_download_and_insert, daemon=True).start()
            return

        photo = load_chat_image(path_str, max_width=max_w)
        self._insert_media_widget(path_str, photo)

    def _insert_media_widget(self, path_str: str, photo):
        """Insert an already-loaded image into the chat display."""
        if photo is None:
            name = Path(path_str).name if "/" in path_str or \
                "\\" in path_str else path_str
            if len(name) > 60:
                name = name[:57] + "..."
            tb = self.chat_display._textbox
            tb.insert("end", f"\n  [Image not available: {name}]\n",
                      "media_caption")
            return

        # Keep reference to prevent garbage collection
        if not hasattr(self, "_chat_images"):
            self._chat_images = []
        self._chat_images.append(photo)
        self._trim_chat_images()

        tb = self.chat_display._textbox
        tb.insert("end", "\n")
        tb.image_create("end", image=photo, padx=12, pady=4)
        tb.insert("end", "\n")
        # Caption with path
        name = Path(path_str).name if "/" in path_str or \
            "\\" in path_str else path_str
        if len(name) > 60:
            name = name[:57] + "..."
        tb.insert("end", f"  {name}\n", "media_caption")
        self._auto_resize_chat()
        self._scroll_chat_to_bottom()

    def _insert_gif(self, path_str: str, source: str = "file"):
        """Insert an animated GIF inline in the chat display.

        URL downloads run in a background thread to avoid GUI freeze.
        """
        from enigma_engine.gui.media import (
            extract_gif_frames, get_media_chat_width,
        )
        try:
            chat_w = self.chat_display.winfo_width()
            max_w = get_media_chat_width(chat_w)
        except Exception:
            max_w = 400

        if source == "url":
            # Network download — offload to thread
            def _download_and_insert():
                frames = extract_gif_frames(path_str, max_width=max_w)
                self.after(0, lambda: self._insert_gif_widget(
                    path_str, source, frames, max_w))

            threading.Thread(
                target=_download_and_insert, daemon=True).start()
            return

        frames = extract_gif_frames(path_str, max_width=max_w)
        self._insert_gif_widget(path_str, source, frames, max_w)

    def _insert_gif_widget(
            self, path_str: str, source: str,
            frames: list | None, max_w: int):
        """Insert already-loaded GIF frames into the chat display."""
        if not frames:
            # Fall back to static image
            self._insert_media(path_str, source)
            return

        # Keep all frame references
        if not hasattr(self, "_chat_images"):
            self._chat_images = []
        self._chat_images.extend([f[0] for f in frames])
        self._trim_chat_images()

        tb = self.chat_display._textbox
        tb.insert("end", "\n")
        # Create image with first frame
        tb.image_create("end", image=frames[0][0], padx=12, pady=4)
        # Get the mark for this image to animate it
        img_index = tb.index("end - 1 char")
        tb.insert("end", "\n")
        name = Path(path_str).name if "/" in path_str or \
            "\\" in path_str else path_str
        tb.insert("end", f"  {name} (GIF)\n", "media_caption")
        self._auto_resize_chat()
        self._scroll_chat_to_bottom()

        # Start animation cycle
        anim_data = {
            "frames": frames,
            "index": img_index,
            "current": 0,
            "active": True,
        }
        self._chat_gif_animations.append(anim_data)
        self._animate_gif(anim_data)

    def _animate_gif(self, anim_data: dict):
        """Cycle through GIF frames using after() scheduling."""
        if getattr(self, '_shutting_down', False):
            return
        if not anim_data.get("active", False):
            return
        frames = anim_data["frames"]
        idx = anim_data["current"]
        next_idx = (idx + 1) % len(frames)
        anim_data["current"] = next_idx
        photo, duration = frames[next_idx]
        try:
            tb = self.chat_display._textbox
            # Update the image at the stored position
            tb.image_configure(
                anim_data["index"], image=photo)
        except Exception:
            anim_data["active"] = False
            return
        self.after(duration, lambda: self._animate_gif(anim_data))

    def _insert_video_thumbnail(
            self, path_str: str, source: str = "file"):
        """Insert a video thumbnail with play overlay in chat."""
        from enigma_engine.gui.media import (
            extract_video_thumbnail, get_media_chat_width,
        )
        try:
            chat_w = self.chat_display.winfo_width()
            max_w = get_media_chat_width(chat_w)
        except Exception:
            max_w = 400

        photo = extract_video_thumbnail(path_str, max_width=max_w)
        if photo is None:
            # Show as text link instead
            self._chat_append(
                "video_link",
                f"\n  \u25b6 VIDEO: {Path(path_str).name}\n")
            # Store path for click handling
            self._chat_media_refs[Path(path_str).name] = path_str
            return

        if not hasattr(self, "_chat_images"):
            self._chat_images = []
        self._chat_images.append(photo)
        self._trim_chat_images()

        tb = self.chat_display._textbox
        tb.insert("end", "\n")
        tb.image_create("end", image=photo, padx=12, pady=4)
        tb.insert("end", "\n")
        name = Path(path_str).name
        tb.insert("end", f"  \u25b6 {name} ", "video_link")
        tb.insert("end", "(click to play)\n", "media_caption")
        self._auto_resize_chat()
        self._scroll_chat_to_bottom()
        # Store path for click handling
        self._chat_media_refs[name] = path_str

    def _make_url_clickable(self, url: str):
        """Tag url text in chat display and store for click lookup.

        Uses unique per-URL tags (link_0, link_1, ...) so
        _on_link_click can identify exactly which URL was clicked.
        Falls back to searching only the last portion of the
        widget to avoid re-tagging old messages.
        """
        tb = self.chat_display._textbox
        if not hasattr(self, "_link_urls"):
            self._link_urls = {}
        # Assign a unique tag id for this URL
        link_id = len(self._link_urls)
        tag_name = f"link_{link_id}"
        self._link_urls[tag_name] = url
        # Search only the last 200 lines to avoid re-tagging old text
        total_lines = int(tb.index("end").split(".")[0])
        search_start = f"{max(1, total_lines - 200)}.0"
        start = search_start
        while True:
            pos = tb.search(url, start, stopindex="end")
            if not pos:
                break
            end_pos = f"{pos}+{len(url)}c"
            # Apply both the general "link" tag (for styling)
            # and the unique tag (for URL lookup)
            tb.tag_add("link", pos, end_pos)
            tb.tag_add(tag_name, pos, end_pos)
            start = end_pos

    def _on_link_click(self, event=None):
        """Handle click on a clickable link tag in chat.

        Checks which link_N tag is at the click position and
        looks up the URL from _link_urls dict.
        """
        tb = self.chat_display._textbox
        try:
            idx = tb.index(f"@{event.x},{event.y}")
            # Check all tags at the click position
            tags = tb.tag_names(idx)
            link_urls = getattr(self, "_link_urls", {})
            for tag in tags:
                if tag in link_urls:
                    self._open_link(link_urls[tag])
                    return
            # Fallback: parse the line for URLs
            line_start = tb.index(f"{idx} linestart")
            line_end = tb.index(f"{idx} lineend")
            line_text = tb.get(line_start, line_end)
            from enigma_engine.gui.media import detect_urls
            urls = detect_urls(line_text)
            if urls:
                self._open_link(urls[0])
        except Exception as exc:
            logger.debug("Link click handler failed: %s", exc)

    def _on_video_click(self, event=None):
        """Handle click on a video link tag in chat."""
        tb = self.chat_display._textbox
        try:
            idx = tb.index(f"@{event.x},{event.y}")
            line_start = tb.index(f"{idx} linestart")
            line_end = tb.index(f"{idx} lineend")
            line_text = tb.get(line_start, line_end).strip()
            # Extract filename from "▶ filename.mp4" pattern
            for name, path in self._chat_media_refs.items():
                if name in line_text:
                    self._open_video(path)
                    return
        except Exception:
            pass

    def _insert_file_link(self, path_str: str):
        """Insert a clickable OPEN FILE line in the chat display."""
        p = Path(path_str)
        name = p.name or path_str
        if not hasattr(self, "_chat_file_refs"):
            self._chat_file_refs = {}
        self._chat_file_refs[name] = str(p)
        self.chat_display._textbox.insert(
            "end", f"  OPEN FILE: {name}\n", "file_link")
        self._auto_resize_chat()
        self._scroll_chat_to_bottom()

    def _on_file_click(self, event=None):
        """Handle click on a file link tag in chat."""
        tb = self.chat_display._textbox
        try:
            idx = tb.index(f"@{event.x},{event.y}")
            line_start = tb.index(f"{idx} linestart")
            line_end = tb.index(f"{idx} lineend")
            line_text = tb.get(line_start, line_end).strip()
            prefix = "OPEN FILE:"
            if prefix in line_text:
                name = line_text.split(prefix, 1)[1].strip()
                path = getattr(self, "_chat_file_refs", {}).get(name)
                if path:
                    self._open_file(path)
        except Exception as exc:
            logger.debug("File click handler failed: %s", exc)

    def _open_file(self, path_str: str):
        """Open a local file in the default app."""
        from enigma_engine.gui.media import _resolve_path
        resolved = _resolve_path(path_str)
        if resolved is None:
            self._chat_error(f"File not found: {path_str}")
            return
        try:
            os.startfile(str(resolved))
            self._chat_system(f"Opened file: {resolved}")
        except Exception as exc:
            self._chat_error(f"Failed to open file: {exc}")

    def _open_link(self, url: str):
        """Open a URL in the default browser."""
        import webbrowser
        try:
            webbrowser.open(url)
            self._chat_system(f"Opened: {url}")
        except Exception as exc:
            self._chat_error(f"Failed to open link: {exc}")

    def _open_video(self, path_str: str):
        """Open a video file in the default media player."""
        from enigma_engine.gui.media import _resolve_path
        resolved = _resolve_path(path_str)
        if resolved is None:
            self._chat_error(f"Video not found: {path_str}")
            return
        try:
            os.startfile(str(resolved))
            self._chat_system(f"Playing: {resolved.name}")
        except Exception as exc:
            self._chat_error(f"Failed to open video: {exc}")

    def _attach_image(self, path: str | None = None):
        """Display an image/GIF/video inline in chat.

        If no path is given, opens a file picker.
        Called by _attach_file when a media file is selected.
        """
        if path is None:
            path = filedialog.askopenfilename(
                title="Send Image",
                filetypes=[
                    ("Images",
                     "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                    ("GIFs", "*.gif"),
                    ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All files", "*.*"),
                ])
            if not path:
                return
        # TEACH-1b: carry image provenance into the next user/assistant
        # exchange so correction capture can tag modality=vision.
        self._pending_correction_image_path = str(path)
        ext = Path(path).suffix.lower()
        name = Path(path).name
        ts = time.strftime("%H:%M")
        # Show as user-sent media
        self._chat_append(
            "timestamp", f"\n  {ts} ",
            "user_prefix", f"{self.user_name}  ",
            "user", f"[sent {name}]\n")
        if ext == ".gif":
            self._insert_gif(path, "file")
        elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            self._insert_video_thumbnail(path, "file")
        else:
            self._insert_media(path, "file")

    # ================================================================
    # Logic - Voice toggle
    # ================================================================

    def _on_voice_toggle(self, is_on: bool):
        self.voice_enabled = is_on
        state = "ON" if is_on else "OFF"
        self._chat_system(f"Voice output {state}")
        # Stop any in-progress TTS when toggled off
        if not is_on:
            self._tts_stop()

    def _tts_clean_text(self, text: str) -> str:
        """Strip markdown, code blocks, URLs, and [CMD] blocks.

        Returns plain readable text suitable for SAPI5 TTS.
        """
        import re
        # Remove fenced code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', ' code block ', text)
        # Remove [CMD]...[/CMD] blocks
        text = re.sub(r'\[CMD\].*?\[/CMD\]', '', text, flags=re.DOTALL)
        # Remove inline backticks (keep content)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # Remove markdown bold/italic markers
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'_{1,3}', ' ', text)
        # Replace URLs with "link"
        text = re.sub(
            r'https?://\S+', 'link', text)
        # Replace markdown links [text](url) with just text
        text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
        # Remove markdown image syntax ![alt](url)
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
        # Collapse excessive whitespace
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tts_chunk_text(self, text: str,
                        max_len: int = 180) -> list[str]:
        """Split text into sentence-sized chunks for safe TTS.

        pyttsx3 SAPI5 hangs on text longer than ~300 chars.
        Splitting at sentence boundaries keeps each chunk short.
        """
        import re
        # Split on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if current and len(current) + len(part) + 1 > max_len:
                chunks.append(current.strip())
                current = part
            else:
                current = f"{current} {part}" if current else part
        if current.strip():
            chunks.append(current.strip())

        # Safety — break any remaining chunks that are still too long
        final: list[str] = []
        for chunk in chunks:
            while len(chunk) > max_len:
                # Find last space within limit
                idx = chunk.rfind(' ', 0, max_len)
                if idx <= 0:
                    idx = max_len
                final.append(chunk[:idx].strip())
                chunk = chunk[idx:].strip()
            if chunk:
                final.append(chunk)
        return final if final else [text[:max_len]]

    def _tts_speak(self, text: str):
        """Speak text aloud using pyttsx3 on a dedicated TTS thread.

        Uses a single persistent thread with a queue to avoid
        COM threading issues on Windows.  The pyttsx3 engine is
        initialised once and reused for every utterance.

        Text is cleaned (markdown/code stripped) and chunked into
        short sentences before being queued — pyttsx3 SAPI5 hangs
        on long text (~300+ chars).

        Stopping is done via a threading.Event checked inside a
        ``started-word`` callback so that ``engine.stop()`` is
        always called on the **worker** thread — never cross-thread.
        SAPI5 COM objects have thread affinity; calling them from
        another thread crashes on Windows.
        """
        if not getattr(self, 'voice_enabled', False):
            return
        if not text or not text.strip():
            return

        # Clean and chunk text before queuing
        cleaned = self._tts_clean_text(text)
        if not cleaned:
            return
        chunks = self._tts_chunk_text(cleaned)

        # Lazily create the TTS worker thread + queue
        if not getattr(self, '_tts_queue', None):
            import queue as _queue
            self._tts_queue = _queue.Queue(maxsize=100)
            self._tts_alive = True

            def _tts_worker():
                """Persistent thread: init engine once, process queue."""
                try:
                    import pyttsx3
                except ImportError:
                    self.after(0, lambda: self._chat_system(
                        "pyttsx3 not installed — "
                        "run: pip install pyttsx3"))
                    return
                try:
                    engine = pyttsx3.init()
                    self._tts_engine_ref = engine
                except Exception as exc:
                    self.voice_enabled = False
                    err = str(exc)
                    self.after(0, lambda e=err: self._chat_system(
                        f"Voice engine failed to start: {e}"))
                    btn = getattr(self, 'voice_btn', None)
                    if btn and hasattr(btn, 'set_state'):
                        self.after(0, lambda: btn.set_state(False))
                    return

                # NOTE: We intentionally do NOT use the
                # 'started-word' callback with engine.stop().
                # On Windows SAPI5, calling engine.stop() from
                # within a runAndWait() callback corrupts the
                # COM engine state, causing it to go silent
                # after the first word. Instead, we check the
                # stop event BETWEEN chunks (each <=180 chars,
                # so the wait is negligible).
                stop_evt = self._tts_stop_event

                while self._tts_alive:
                    try:
                        msg = self._tts_queue.get(timeout=1.0)
                    except Exception:
                        # queue.Empty — loop back and check alive
                        continue
                    if msg is None:
                        # Poison pill — shut down
                        break
                    if stop_evt.is_set():
                        stop_evt.clear()
                        continue
                    try:
                        engine.say(msg)
                        engine.runAndWait()
                    except Exception:
                        # Engine error — reinitialise
                        try:
                            engine = pyttsx3.init()
                            self._tts_engine_ref = engine
                        except Exception:
                            break
                    # Clear stop flag after each utterance so it
                    # doesn't linger and skip the next message.
                    stop_evt.clear()
                try:
                    engine.stop()
                except Exception:
                    pass
                self._tts_engine_ref = None

            t = threading.Thread(target=_tts_worker, daemon=True)
            t.start()

        for chunk in chunks:
            self._tts_queue.put(chunk)

    def _tts_stop(self):
        """Stop any in-progress TTS playback.

        Sets the stop event so the worker skips remaining chunks.
        Also drains the queue to prevent queued chunks from playing
        after the current utterance finishes.
        """
        self._tts_stop_event.set()
        # Drain pending chunks so they don't play later
        q = getattr(self, '_tts_queue', None)
        if q:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

    def _tts_shutdown(self):
        """Shut down the TTS worker thread (called on app close)."""
        self._tts_alive = False
        q = getattr(self, '_tts_queue', None)
        if q:
            try:
                q.put(None)  # Poison pill
            except Exception:
                pass
        self._tts_stop()

    # ================================================================
    # Logic - Voice input (speech-to-text)
    # ================================================================

    def _toggle_voice_input(self):
        """Start or stop continuous voice input.

        Works like a conversation — when the mic is on, each
        recognised phrase is automatically sent as a chat message.
        Click the mic button again (or press Escape) to stop.
        Uses listen_in_background for non-blocking recording.
        """
        # If already recording, stop it
        if getattr(self, "_voice_recording", False):
            self._voice_stop_listening()
            return

        try:
            import speech_recognition as sr
        except ImportError:
            self._chat_error(
                "speech_recognition not installed. "
                "Run: pip install SpeechRecognition")
            return

        self._voice_recording = True
        self.mic_btn.configure(
            fg_color=C_RED, text_color="#ffffff")
        self._chat_system(
            "Voice conversation started — speak naturally. "
            "Click mic again to stop.")

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.0

        def _on_audio(rec, audio):
            """Called each time a phrase is detected."""
            if not getattr(self, '_voice_recording', False):
                return
            try:
                text = rec.recognize_google(audio)
                if text and text.strip():
                    self.after(0, lambda t=text: self._on_voice_text(t))
            except Exception:
                # Recognition failed for this phrase — ignore and
                # keep listening for the next one
                pass

        try:
            mic = sr.Microphone()
            # listen_in_background returns a stopper callable;
            # each phrase triggers _on_audio without blocking
            self._voice_stopper = recognizer.listen_in_background(
                mic, _on_audio, phrase_time_limit=30)
        except Exception as exc:
            self._voice_recording = False
            self._voice_input_done()
            err = str(exc) if str(exc) else type(exc).__name__
            self._chat_error(f"Microphone error: {err}")

    def _voice_stop_listening(self):
        """Stop background voice listening."""
        self._voice_recording = False
        stopper = getattr(self, "_voice_stopper", None)
        if stopper:
            try:
                stopper(wait_for_stop=False)
            except Exception:
                pass
            self._voice_stopper = None
        self._voice_input_done()
        self._chat_system("Voice conversation ended.")

    def _on_voice_text(self, text: str):
        """Auto-send transcribed speech as a chat message.

        Instead of inserting into the input box, this sends
        directly so voice works like a real conversation.
        """
        if not text:
            return
        # Don't send if AI is already generating
        if getattr(self, '_is_generating', False):
            return
        self._chat_system(f'Voice: "{text}"')
        # Put text into input and send immediately
        self.chat_input.delete("1.0", "end")
        self.chat_input.insert("1.0", text)
        self._send_message()

    def _voice_input_done(self):
        """Reset mic button after recording ends."""
        try:
            self.mic_btn.configure(
                fg_color=C_SURFACE, text_color=C_TEXT_DIM)
        except Exception:
            pass
