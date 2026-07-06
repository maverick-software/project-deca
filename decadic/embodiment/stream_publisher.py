"""Live spectator-camera streaming for the embodied agent.

Renders a spectator/follow camera of the running MuJoCo world at a steady
frame rate and pushes it as an H.264 RTMP stream to any ingest endpoint --
YouTube Live, Twitch, or a self-hosted media server (MediaMTX / nginx-rtmp).
Because it renders offscreen through the same headless-capable path as the
agent's egocentric eye (``MUJOCO_GL=egl``), it runs anywhere the body runs:
your desktop or a rented headless GPU box. Nothing about the stream depends on
WHERE the agent is deployed -- set an ingest URL (or a YouTube stream key) and
it goes live.

Design:
  * Own thread, own ``mujoco.Renderer`` (MuJoCo GL contexts are thread-local;
    the streamer never touches the body's egocentric renderer). It reads the
    shared ``MjData`` each frame -- an occasional torn read during a physics
    step is at worst one cosmetically glitched frame, never a crash.
  * Raw RGB frames are piped to an ``ffmpeg`` subprocess that encodes video +
    a silent audio track (YouTube requires an audio stream) and publishes FLV
    over RTMP.
  * Self-healing: if ffmpeg exits (a network blip on the RTMP link), it is
    relaunched with backoff, mirroring the body's own reconnect discipline.
  * Fully opt-in and side-effect-free when disabled: constructed only when an
    ingest target is configured, so existing runs and the test suite are
    byte-identical.

Enable via environment (preferred for deployment -- the launcher just sets
these) or the adapter's ``--stream`` flag:
  * ``DECADIC_STREAM_RTMP``     full ingest URL, e.g.
      ``rtmp://a.rtmp.youtube.com/live2/xxxx-xxxx-xxxx-xxxx-xxxx``
  * ``DECADIC_YT_STREAM_KEY``   just the YouTube stream key; the standard
      YouTube ingest URL is built around it. (Ignored if RTMP is set.)
  * ``DECADIC_STREAM_CAMERA``   spectator camera name (default: first non-eye
      camera in the scene; falls back to the free overview camera).
  * ``DECADIC_STREAM_SIZE``     ``WxH`` (default ``1280x720``; dims forced even).
  * ``DECADIC_STREAM_FPS``      target frame rate (default ``30``).
  * ``DECADIC_STREAM_BITRATE``  video bitrate, e.g. ``4500k`` (default 720p30).
  * ``DECADIC_STREAM_VCODEC``   ``libx264`` (default) or ``h264_nvenc`` to
      offload encoding to the GPU on a CUDA box.
  * ``DECADIC_FFMPEG``          ffmpeg binary path (default: ``ffmpeg`` on PATH).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

YT_INGEST_BASE = "rtmp://a.rtmp.youtube.com/live2"


def resolve_ingest_url() -> str | None:
    """The configured RTMP ingest URL, or None if streaming is not requested.

    A raw ``DECADIC_STREAM_RTMP`` wins (any provider); otherwise a YouTube
    stream key is wrapped in the standard YouTube ingest URL."""
    rtmp = os.environ.get("DECADIC_STREAM_RTMP", "").strip()
    if rtmp:
        return rtmp
    key = os.environ.get("DECADIC_YT_STREAM_KEY", "").strip()
    if key:
        return f"{YT_INGEST_BASE}/{key}"
    return None


def _redact(url: str) -> str:
    """Never log a stream key: keep the host, mask the key path."""
    try:
        head, _, _ = url.rpartition("/")
        return f"{head}/****"
    except Exception:
        return "****"


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _parse_size(default_w: int = 1280, default_h: int = 720) -> tuple[int, int]:
    raw = os.environ.get("DECADIC_STREAM_SIZE", "").strip().lower()
    if raw and "x" in raw:
        try:
            w_s, h_s = raw.split("x", 1)
            return _even(int(w_s)), _even(int(h_s))
        except ValueError:
            logger.warning("bad DECADIC_STREAM_SIZE=%r; using %dx%d", raw, default_w, default_h)
    return default_w, default_h


class StreamPublisher:
    """Renders a spectator camera and publishes it as RTMP via ffmpeg.

    Parameters
    ----------
    mj : the imported ``mujoco`` module (the sim already holds it).
    model, data : the LIVE ``MjModel`` / ``MjData`` the physics loop is driving.
    ingest_url : full RTMP URL (see ``resolve_ingest_url``).
    camera : spectator camera name, or None to auto-pick / free-cam.
    """

    def __init__(
        self,
        mj: Any,
        model: Any,
        data: Any,
        *,
        ingest_url: str,
        camera: str | None = None,
    ) -> None:
        self._mj = mj
        self._model = model
        self._data = data
        self._ingest_url = ingest_url
        self._w, self._h = _parse_size()
        self._fps = max(1, int(os.environ.get("DECADIC_STREAM_FPS", "30")))
        self._bitrate = os.environ.get("DECADIC_STREAM_BITRATE", "4500k").strip()
        self._vcodec = os.environ.get("DECADIC_STREAM_VCODEC", "libx264").strip()
        self._ffmpeg = os.environ.get("DECADIC_FFMPEG", "ffmpeg").strip() or "ffmpeg"
        self._camera = self._resolve_camera(camera)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._renderer: Any = None
        self.frames_sent = 0
        self.ffmpeg_restarts = 0

    # -- camera resolution --------------------------------------------------
    def _resolve_camera(self, requested: str | None) -> str | int:
        want = (requested or os.environ.get("DECADIC_STREAM_CAMERA", "")).strip()
        names = []
        for c in range(self._model.ncam):
            nm = self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_CAMERA, c) or ""
            if nm:
                names.append(nm)
        if want and want in names:
            return want
        if want:
            logger.warning("stream camera %r not in %s; falling back", want, names)
        # Prefer any named spectator camera over the agent's eye; else free cam.
        for nm in names:
            if nm != "egocentric":
                return nm
        return -1  # MuJoCo free/overview camera

    # -- lifecycle ----------------------------------------------------------
    @classmethod
    def maybe_create(cls, mj: Any, model: Any, data: Any) -> "StreamPublisher | None":
        """Build a publisher iff an ingest target is configured AND ffmpeg is
        available. Never raises: a misconfiguration disables streaming and
        leaves the body untouched."""
        url = resolve_ingest_url()
        if not url:
            return None
        if shutil.which(os.environ.get("DECADIC_FFMPEG", "ffmpeg")) is None:
            logger.error(
                "streaming requested but ffmpeg not found on PATH "
                "(set DECADIC_FFMPEG or install ffmpeg); streaming disabled"
            )
            return None
        try:
            return cls(mj, model, data, ingest_url=url)
        except Exception:
            logger.exception("stream publisher init failed; streaming disabled")
            return None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="mujoco-streamer", daemon=True
        )
        self._thread.start()
        cam = self._camera if isinstance(self._camera, str) else "(free overview)"
        print(
            f"[stream] LIVE -> {_redact(self._ingest_url)} | camera={cam} "
            f"{self._w}x{self._h}@{self._fps} {self._vcodec} {self._bitrate}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None
        self._kill_proc()
        print(f"[stream] stopped ({self.frames_sent} frames sent)", flush=True)

    # -- ffmpeg -------------------------------------------------------------
    def _ffmpeg_cmd(self) -> list[str]:
        # rawvideo (rgb24) in on stdin + a silent stereo track (YouTube needs
        # audio); H.264 + AAC out over FLV/RTMP. zerolatency keeps the live
        # delay short; the 2s keyframe interval matches YouTube's guidance.
        gop = str(self._fps * 2)
        cmd = [
            self._ffmpeg,
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self._w}x{self._h}",
            "-r", str(self._fps),
            "-i", "-",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", self._vcodec,
            "-pix_fmt", "yuv420p",
            "-b:v", self._bitrate,
            "-maxrate", self._bitrate,
            "-bufsize", self._bitrate,
            "-g", gop,
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-f", "flv",
            self._ingest_url,
        ]
        # Encoder-specific latency flags.
        if self._vcodec == "libx264":
            cmd[cmd.index("-c:v") + 2 : cmd.index("-c:v") + 2] = [
                "-preset", "veryfast", "-tune", "zerolatency",
            ]
        return cmd

    def _spawn_proc(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                self._ffmpeg_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,  # let ffmpeg warnings surface in the body log
            )
            return True
        except Exception:
            logger.exception("failed to launch ffmpeg; streaming paused")
            self._proc = None
            return False

    def _kill_proc(self) -> None:
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
            p.wait(timeout=3.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    # -- render/publish loop ------------------------------------------------
    def _run(self) -> None:
        try:
            self._renderer = self._mj.Renderer(self._model, height=self._h, width=self._w)
        except Exception:
            logger.exception("stream renderer init failed; streaming disabled")
            return

        backoff = 1.0
        frame_dt = 1.0 / float(self._fps)
        next_frame = time.perf_counter()
        while not self._stop.is_set():
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:  # it died -> restart with backoff
                    self.ffmpeg_restarts += 1
                    logger.warning("ffmpeg exited; restarting stream in %.0fs", backoff)
                    self._kill_proc()
                    if self._stop.wait(backoff):
                        break
                    backoff = min(15.0, backoff * 2.0)
                if not self._spawn_proc():
                    if self._stop.wait(backoff):
                        break
                    backoff = min(15.0, backoff * 2.0)
                    continue
                backoff = 1.0  # healthy launch resets the backoff

            try:
                self._renderer.update_scene(self._data, camera=self._camera)
                pixels = self._renderer.render()  # HxWx3 uint8 RGB
                self._proc.stdin.write(pixels.tobytes())
                self.frames_sent += 1
            except BrokenPipeError:
                self._kill_proc()  # ffmpeg went away; loop restarts it
                continue
            except Exception:
                logger.exception("stream frame failed (continuing)")

            # Pace to the target fps without drifting.
            next_frame += frame_dt
            sleep = next_frame - time.perf_counter()
            if sleep > 0:
                if self._stop.wait(sleep):
                    break
            else:
                next_frame = time.perf_counter()  # fell behind; resync

        try:
            if self._renderer is not None:
                self._renderer.close()
        except Exception:
            pass
