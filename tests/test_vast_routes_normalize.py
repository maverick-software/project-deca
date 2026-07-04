"""_normalize_offer(): raw Vast.ai offer dict -> the shape the dashboard renders.

Covers the GPU-offer-search redesign (docs/gpu_offer_search_wbs.md, phase B1):
the newly-added fields (total_flops, bandwidth, CPU info, port count, host
IDs, datacenter flag, days remaining), plus the None-fallback paths for
fields real offers won't always populate. Uses a synthetic raw dict shaped
like Vast's own documented example (docs.vast.ai/api/search-offers) rather
than live data - the sandbox has no Vast API key/CLI to capture a real one.
"""

from __future__ import annotations

import time

from decadic.api.vast.routes import (
    _days_remaining,
    _is_datacenter,
    _mb_to_gb,
    _mbps,
    _normalize_offer,
)

# A coherent (internally-consistent) synthetic offer, in the spirit of Vast's
# documented example but with numbers that actually add up for a single-GPU
# offer, since the docs' own example data is illustrative/fake and not
# internally consistent (e.g. its num_gpus/gpu_total_ram don't multiply out).
RAW_OFFER: dict = {
    "id": 12345678,
    "gpu_name": "RTX_4090",
    "num_gpus": 1,
    "gpu_ram": 24564,  # MB -> ~24 GB
    "gpu_total_ram": 24564,
    "cpu_ram": 65536,  # MB -> 64 GB
    "dph_total": 0.345,
    "dlperf": 55.5,
    "dlperf_per_dphtotal": 160.9,
    "cuda_max_good": 12.4,
    "geolocation": "US",
    "reliability2": 0.9954,
    "verification": "verified",
    "total_flops": 82.6,
    "gpu_mem_bw": 1008.0,
    "inet_up": 850.0,  # MB/s (per docs) -> 6800 Mbps
    "inet_down": 900.0,  # MB/s -> 7200 Mbps
    "cpu_name": "AMD EPYC 7413",
    "cpu_cores": 32,
    "direct_port_count": 8,
    "host_id": 555,
    "machine_id": 888,
    "hosting_type": 1,
    "end_date": time.time() + 5 * 86400.0,  # ~5 days out
}


def test_normalize_offer_maps_every_new_field():
    n = _normalize_offer(RAW_OFFER)
    assert n["total_flops"] == 82.6
    assert n["gpu_mem_bw_gbps"] == 1008.0
    assert n["inet_up_mbps"] == 6800.0
    assert n["inet_down_mbps"] == 7200.0
    assert n["cpu_name"] == "AMD EPYC 7413"
    assert n["cpu_cores"] == 32
    assert n["direct_port_count"] == 8
    assert n["host_id"] == 555
    assert n["machine_id"] == 888
    assert n["is_datacenter"] is True
    assert n["days_remaining"] is not None
    assert 4.9 < n["days_remaining"] <= 5.0


def test_normalize_offer_still_maps_existing_fields():
    n = _normalize_offer(RAW_OFFER)
    assert n["gpu_name"] == "RTX_4090"
    assert n["gpu_ram_gb"] == _mb_to_gb(24564)
    assert n["verified"] is True


def test_missing_optional_fields_fall_back_to_none_not_guesses():
    minimal = {"id": 1, "gpu_name": "RTX_3090", "dph_total": 0.2}
    n = _normalize_offer(minimal)
    assert n["total_flops"] is None
    assert n["gpu_mem_bw_gbps"] is None
    assert n["inet_up_mbps"] is None
    assert n["cpu_name"] is None
    assert n["direct_port_count"] is None
    assert n["host_id"] is None
    assert n["is_datacenter"] is None  # unknown, not "community" or "datacenter"
    assert n["days_remaining"] is None


def test_mbps_conversion():
    assert _mbps(100.0) == 800.0
    assert _mbps(0) == 0.0
    assert _mbps(None) is None
    assert _mbps("not a number") is None
    assert _mbps(-5) is None


def test_is_datacenter_reads_hosting_type_then_datacenter_key():
    assert _is_datacenter({"hosting_type": 0}) is False
    assert _is_datacenter({"hosting_type": 1}) is True
    assert _is_datacenter({"hosting_type": 2}) is True
    assert _is_datacenter({"datacenter": True}) is True
    assert _is_datacenter({"datacenter": False}) is False
    assert _is_datacenter({}) is None


def test_days_remaining_none_when_past_or_missing():
    assert _days_remaining(None) is None
    assert _days_remaining("garbage") is None
    assert _days_remaining(time.time() - 1000.0) is None  # already ended
    future = _days_remaining(time.time() + 2 * 86400.0)
    assert future is not None and 1.9 < future <= 2.0
