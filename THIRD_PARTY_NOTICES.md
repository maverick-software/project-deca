# Third-Party License Notices

Project Deca (the *Decadic Cycle Cognitive Architecture*) is distributed under
[AGPL-3.0-or-later](./LICENSE) with a [commercial option](./LICENSING.md). It depends on
third-party open-source packages, **all under permissive licenses** (MIT, BSD, Apache-2.0,
HPND). This file records the attribution obligations those licenses carry. It applies to
**every** distribution of Deca — source or binary, open-source or commercial.

> **Not legal advice.** This is a maintainer's good-faith inventory. For any commercial
> distribution, have counsel confirm the obligations below and verify each dependency's
> current license (versions change licenses rarely, but they can).

## Obligations at a glance

- **No copyleft.** None of Deca's dependencies is under GPL/LGPL/MPL or any other copyleft
  license, so none imposes source-disclosure obligations on downstream users. Deca's own
  copyleft (AGPL §13) comes from Deca's license, not from any dependency. A **commercial
  licensee of Deca inherits no copyleft from these dependencies** — only the attribution
  duties listed here.
- **Attribution travels with distribution.** MIT, BSD, and HPND require that their copyright
  notice and license text be preserved in distributions. Apache-2.0 additionally requires
  reproducing any `NOTICE` file the dependency ships and stating that changes were made (Deca
  makes none to these dependencies — it consumes them unmodified).
- **License compatibility.** Apache-2.0 is one-way compatible into AGPL-3.0-or-later, and
  MIT/BSD/HPND compose freely, so bundling these dependencies with AGPL-licensed Deca is sound.
- **No vendored third-party source.** Deca does not copy third-party source into its own tree;
  dependencies are consumed as installed packages. There are therefore no foreign copyright
  headers to preserve inside `decadic/`.

## Runtime dependencies and their licenses

Snapshot of the declared dependencies in `pyproject.toml` (verify current text via the
generated appendix — see below).

| Package | Typical license | Role |
|---|---|---|
| FastAPI | MIT | HTTP/WebSocket server framework |
| Starlette (via FastAPI) | BSD-3-Clause | ASGI toolkit |
| Uvicorn | BSD-3-Clause | ASGI server |
| Pydantic | MIT | Data validation |
| NumPy | BSD-3-Clause | Numerics |
| PyTorch (`torch`) | BSD-3-Clause* | Trainable neural stack |
| Hugging Face Transformers | Apache-2.0 | Frozen CLIP/Whisper front-end loader |
| Pillow | HPND (MIT-style) | Image handling |
| httpx | BSD-3-Clause | HTTP client |
| vastai | *verify* | Optional cloud-GPU deploy client |
| LanceDB | Apache-2.0 | Episodic vector store |
| Kuzu | MIT | Semantic knowledge graph |
| PyArrow | Apache-2.0 | Columnar data (LanceDB schema) |
| MuJoCo | Apache-2.0 | Optional physical body simulation |
| websockets | BSD-3-Clause | WebSocket client (body/tests) |
| matplotlib | Matplotlib (BSD-compatible) | Optional plotting (dev) |
| pytest, pytest-asyncio | MIT / Apache-2.0 | Test suite (dev) |

\* PyTorch's top-level license is BSD-3-Clause; it bundles third-party components under their
own permissive licenses, reproduced in the PyTorch distribution's own notices.

### Downloaded model weights (not redistributed)

When `DECADIC_ENCODER_MODE=hf`, Deca downloads frozen encoder weights at runtime from
Hugging Face — `openai/clip-vit-base-patch32` and `openai/whisper-small`, both under the **MIT
license** (OpenAI). Deca does **not** redistribute these weights; they are fetched by the end
user's own environment. **If you package or ship a distribution that bundles these weights**,
you must also carry their MIT license notices. `zeros` encoder mode uses no downloaded weights.

## The verbatim license-text appendix

The full copyright and license text of every installed package is generated (not hand-written)
into `THIRD_PARTY_NOTICES.txt`. Regenerate it after any dependency change:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\gen_third_party_notices.ps1
```

```bash
# Unix (equivalent)
.venv/bin/python -m pip install pip-licenses
.venv/bin/python -m piplicenses --with-license-file --no-license-path --with-urls \
  --with-authors --format=plain-vertical --output-file THIRD_PARTY_NOTICES.txt
```

`THIRD_PARTY_NOTICES.txt` is the authoritative, machine-generated record; this Markdown file
is the durable human-readable summary. Ship both with any distribution.
