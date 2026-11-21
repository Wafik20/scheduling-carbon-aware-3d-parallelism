"""Offline safeguards for trace retrieval and recovered workshop artifacts."""
import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def generator(monkeypatch):
    spec = importlib.util.spec_from_file_location("trace_generator", ROOT / "scripts/generate_good_month_traces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "EMAPS_API_KEY", "offline-test")
    monkeypatch.setattr(module, "EMAPS_ZONE", "TEST-ZONE")
    return module


def test_explicit_zone_and_response_match(generator, monkeypatch):
    captured = {}
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"zone": "TEST-ZONE", "data": [{"carbonIntensity": 42}]}
    def get(url, **kwargs):
        captured.update(kwargs)
        return Response()
    monkeypatch.setattr(generator.requests, "get", get)
    start, end = generator.month_range_utc(2024, 5)
    assert generator.fetch_chunk(start, end) == [{"carbonIntensity": 42}]
    assert captured["params"]["zone"] == "TEST-ZONE"
    assert captured["params"]["disableCallerLookup"] == "true"
    assert captured["params"]["temporalGranularity"] == "hourly"
    monkeypatch.setattr(generator, "EMAPS_ZONE", "OTHER")
    with pytest.raises(ValueError, match="received"):
        generator.fetch_chunk(start, end)
    monkeypatch.setattr(Response, "json", lambda self: {"data": []})
    with pytest.raises(ValueError, match="received"):
        generator.fetch_chunk(start, end)
    monkeypatch.setattr(generator, "EMAPS_ZONE", None)
    with pytest.raises(ValueError, match="EMAPS_ZONE"):
        generator.fetch_chunk(start, end)


def test_hourly_range_validation(generator, monkeypatch):
    start, _ = generator.month_range_utc(2024, 5)
    end = start + generator.timedelta(hours=2)
    monkeypatch.setattr(generator.time, "sleep", lambda _: None)
    entries = [{"datetime": (start + generator.timedelta(hours=i)).isoformat(), "carbonIntensity": 1} for i in range(3)]
    monkeypatch.setattr(generator, "fetch_chunk", lambda *_: entries)
    assert len(generator.fetch_range(start, end)) == 2
    entries.pop(0)
    with pytest.raises(ValueError, match="Expected"):
        generator.fetch_range(start, end)


def test_recovered_historical_traces():
    expected = {
        "us-east-2": "5857c58e4f8f022908e0b0b524c2e7e48415175c3eea5e00269190fc2ba9df15",
        "us-west-1": "8fb0da0d267112b666d726298746f4391804ca619b3924a9a1dec279666db96e",
        "us-west-2": "626fd4b194dc07ff3c3c2f2e15c8f52f34e37f215b14430532978722ab0acc64",
    }
    for region, digest in expected.items():
        path = ROOT / "data/recovered_workshop_may2024" / f"carbon_2024_05_hourly_aws_{region}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with path.open() as stream:
            assert len(list(csv.DictReader(stream))) == 744
