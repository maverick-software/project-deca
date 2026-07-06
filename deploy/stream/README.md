# Live-streaming the agent's world (YouTube / any RTMP)

This streams a third-person view of the MuJoCo environment around the agent to
YouTube Live (or Twitch, or your own media server) and gives you a webpage that
embeds it. It works **wherever the body runs** — your desktop or a rented GPU
box — because the frame is rendered offscreen (the same headless-capable path as
the agent's eye) and pushed straight out over RTMP. Outbound only: no inbound
ports, no tunnels.

Pipeline: `mujoco.Renderer` (spectator camera) → raw frames → `ffmpeg` (H.264 +
silent AAC) → RTMP ingest. Implemented in
[`decadic/embodiment/stream_publisher.py`](../../decadic/embodiment/stream_publisher.py),
wired into the body adapter, auto-enabled by an env var.

Requirement everywhere: **ffmpeg on PATH** (already added to the vast provisioner;
on your desktop, `winget install Gyan.FFmpeg` or grab a build from ffmpeg.org).

---

## Step 1 — Create the live stream on your YouTube channel (one time)

1. Go to **YouTube Studio → Create → Go Live** (your channel must be verified;
   first-time live enablement can take ~24h).
2. Choose **Streaming software** (not webcam). Give it a title.
3. In **Stream settings**, copy the **Stream key** (looks like
   `xxxx-xxxx-xxxx-xxxx-xxxx`). This is the only secret you need. Treat it like a
   password — anyone with it can stream to your channel.

That key is permanent/reusable, so you set it once.

---

## Step 2 — Point the agent at your channel

Set your stream key in the environment. Either works:

```powershell
# Windows / PowerShell (local run)
$env:DECADIC_YT_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"
```

```bash
# Linux / the vast box
export DECADIC_YT_STREAM_KEY="xxxx-xxxx-xxxx-xxxx-xxxx"
```

That's the whole connection. The standard YouTube ingest URL is built around the
key automatically. (For a non-YouTube target, set `DECADIC_STREAM_RTMP` to a full
`rtmp://…` URL instead — Twitch, Restream, MediaMTX, etc.)

Optional tuning (sane defaults shown):

| Env var | Default | Meaning |
|---|---|---|
| `DECADIC_STREAM_CAMERA` | first spectator cam | scene camera to broadcast |
| `DECADIC_STREAM_SIZE` | `1280x720` | resolution |
| `DECADIC_STREAM_FPS` | `30` | frame rate |
| `DECADIC_STREAM_BITRATE` | `4500k` | video bitrate |
| `DECADIC_STREAM_VCODEC` | `libx264` | use `h264_nvenc` on a CUDA box to offload encoding to the GPU |
| `DECADIC_FFMPEG` | `ffmpeg` | ffmpeg binary path |

---

## Step 3 — Run it

### Local (your machine)

Start the server as usual, then run the body with the stream on:

```powershell
$env:DECADIC_YT_STREAM_KEY = "xxxx-xxxx-xxxx-xxxx-xxxx"
.venv\Scripts\python.exe scripts\mujoco_decadic_adapter.py --port 8765 --stream
```

Setting the key alone also auto-enables the stream (the `--stream` flag is just
an explicit switch). You'll see a `[stream] LIVE -> rtmp://.../****` line.

### On a rented GPU box (vast.ai)

ffmpeg is installed by the provisioner, and the body inherits the server's
environment, so you only need the key present when the server launches. Set
`DECADIC_YT_STREAM_KEY` in the environment that invokes `deploy/vast/run_remote.sh`
(it is passed straight through to the body). Nothing else changes — deploy as
normal and it goes live.

---

## Step 4 — Confirm it's live

Back in YouTube Studio's Go Live page you'll see the incoming feed and stream
health within a few seconds. Click **Go Live** to make it public. (YouTube adds
~5–20 s of buffering latency; that's normal for RTMP. Use "Low-latency" or
"Ultra-low-latency" in stream settings to reduce it.)

---

## Step 5 — Put it on your website

Open [`watch.html`](./watch.html), set **one** constant at the top, and host the
file anywhere static (Netlify, Cloudflare Pages, GitHub Pages, an S3 bucket, or
your existing site):

- `YT_CHANNEL_ID` — **recommended.** Your channel ID (`UC…`, from
  youtube.com/account_advanced). The page then always shows whatever is currently
  live on your channel, so you never touch it again between streams.
- `YT_VIDEO_ID` — pin one specific broadcast instead.
- `HLS_URL` — only if you self-host via a media server rather than YouTube.

That page is a complete, dependency-free embed (plus an `hls.js` fallback for the
self-hosted path).

---

## Which path for the website?

- **YouTube embed (this setup):** zero infra, free CDN, scales to any audience,
  ~10 s latency. Best default — the stream is on YouTube Live *and* your site at
  once, which is exactly the two-places-at-once you wanted.
- **Self-hosted (HLS/WebRTC via MediaMTX):** your own domain, no YouTube chrome,
  sub-second latency possible, but you run and scale the server. Point
  `DECADIC_STREAM_RTMP` at the media server and set `HLS_URL` in `watch.html`.
  Reach for this only if you specifically need low latency or no third party.
