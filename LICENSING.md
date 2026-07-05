# Licensing

Copyright (C) 2026 Charles Richard Wayne Sears. All rights reserved.

The Decadic Cycle of Expression — the cognitive framework this software
implements — originates in the book *The Architecture of Awareness: Decoding
Consciousness* by Charles R.W. Sears (ISBN 1963092066). Any use of this
software should credit both the software and the originating work; see
[How to cite](#how-to-cite).

## Open-source license (default)

This project — the Decadic Cycle Cognitive Architecture — is released under the
**GNU Affero General Public License, version 3 or later (AGPL-3.0-or-later)**.
The full text is in [`LICENSE`](./LICENSE).

In plain terms, the AGPL lets anyone use, study, modify, and share this
software, on one central condition: **if you run a modified version as a
network service, you must make your modified source available to the users of
that service.** Ordinary GPL only triggers on distribution; the AGPL's
Section 13 closes the "software-as-a-service" gap. Because this system is
designed to be served over an API (REST + WebSocket), that clause is the whole
point — it prevents a third party from taking the architecture, running it as
a closed hosted product, and giving nothing back.

If you deploy a modified version as a network service, you must offer its users
a way to obtain the corresponding source (for example, a visible "Source" link
in any dashboard or UI that points to your published, matching source tree).

## Commercial license (alternative)

The AGPL's copyleft and network-source obligations are not acceptable to every
user. If you want to build on this software **without** the AGPL's
requirements — for instance, to embed it in a proprietary or closed-source
product or hosted service — a separate **commercial license is available from
the copyright holder**.

This is a dual-licensing model: the same code is offered under AGPL-3.0 to the
community and under a negotiated commercial license to those who need different
terms.

To inquire about a commercial license, please reach out to
**Charles Richard Wayne Sears**:

- **Contact form (preferred):** https://www.charlesrsears.com/#connect
- Email: charles.r.sears@gmail.com
- LinkedIn: https://www.linkedin.com/in/charlesrsears/

Because the project is offered under two licenses, all outside contributions
require a signed **Contributor License Agreement (CLA)** assigning sufficient
rights to the copyright holder; otherwise the dual-licensing offer could not be
maintained. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) if present, or ask.

## Third-party dependencies

This software depends on third-party open-source packages, **all under permissive licenses**
(MIT, BSD, Apache-2.0, HPND) — none copyleft. Their attribution obligations, a per-package
license inventory, and the note on runtime-downloaded model weights (CLIP/Whisper, MIT) are
recorded in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md); the verbatim license texts are
generated into `THIRD_PARTY_NOTICES.txt` by `scripts/gen_third_party_notices.ps1`. Because none
of these dependencies is copyleft, **a commercial licensee inherits no source-disclosure
obligation from them** — only the standard attribution duties, which travel with any
distribution. Ship both notices files (or their contents) with any distribution, open or commercial.

## Trademarks (not covered by the code license)

**"Decadic Cycle of Expression", "Project Deca", and "Deca"**, and any
associated logos, are trademarks of Charles Richard Wayne Sears. The AGPL grants rights to
the *code*; it does **not** grant any right to use these names or marks. You may
state factually that your work is "based on" or "compatible with" the Decadic
architecture, but you may not name a fork or derivative product in a way that
implies it is the original or is endorsed by the author. (This mirrors how,
e.g., the Firefox source is open while the name is controlled.)

## How to cite

If you use this work in research or writing, please cite **both** the software
and the originating book. A machine-readable citation for both is provided in
[`CITATION.cff`](./CITATION.cff).

- **Software:** Sears, Charles Richard Wayne. *Decadic Cycle Cognitive
  Architecture* (software), 2026. Licensed AGPL-3.0-or-later.
- **Originating framework:** Sears, Charles R.W. *The Architecture of Awareness:
  Decoding Consciousness.* ISBN 1963092066.

The architecture and the underlying Decadic Cycle of Expression are the
intellectual contribution of the author, first set out in the book above;
citation, not the software license, is what secures that credit.
