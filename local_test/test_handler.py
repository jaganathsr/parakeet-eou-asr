#!/usr/bin/env python3
"""
Local integration test for the Parakeet EOU handler.

Generates test WAV audio (sine waves), then hits /health and /predict
on localhost:8080.

Usage:
    python3 test_handler.py
"""

import base64, io, json, struct, sys, math
import urllib.request


BASE_URL = "http://localhost:8080"


def generate_wav_bytes(freq=440.0, duration=1.0, sample_rate=16000):
    """Generate a sine-wave WAV (16-bit PCM, mono, 16kHz) in memory."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = 0.3 * math.sin(2 * math.pi * freq * t)
        val += 0.2 * math.sin(2 * math.pi * (freq * 1.5) * t)
        samples.append(int(val * 32767))

    buf = io.BytesIO()
    data_size = n_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for s in samples:
        buf.write(struct.pack("<h", s))
    return buf.getvalue()


def test_health():
    print("=" * 60)
    print("TEST: GET /health")
    print("=" * 60)
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            print(f"  Status code: {resp.status}")
            print(f"  Response:    {body}")
            assert resp.status == 200
            assert body.get("status") == "healthy"
            print("  PASSED")
            return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_predict_single():
    print("\n" + "=" * 60)
    print("TEST: POST /predict (single instance)")
    print("=" * 60)
    wav_bytes = generate_wav_bytes(duration=2.0)
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    payload = json.dumps({"instances": [{"audio_base64": audio_b64}]}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            print(f"  Status code: {resp.status}")
            print(f"  Response:    {json.dumps(body, indent=4)}")
            assert resp.status == 200
            assert "predictions" in body
            assert len(body["predictions"]) == 1
            pred = body["predictions"][0]
            assert "transcription" in pred
            assert "end_of_utterance_detected" in pred
            print("  PASSED")
            return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_predict_batch():
    print("\n" + "=" * 60)
    print("TEST: POST /predict (batch of 3)")
    print("=" * 60)
    instances = []
    for freq in [300, 440, 600]:
        wav_bytes = generate_wav_bytes(freq=freq, duration=1.5)
        instances.append({"audio_base64": base64.b64encode(wav_bytes).decode("utf-8")})
    payload = json.dumps({"instances": instances}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
            print(f"  Status code: {resp.status}")
            print(f"  Response:    {json.dumps(body, indent=4)}")
            assert resp.status == 200
            assert len(body["predictions"]) == 3
            print("  PASSED")
            return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_predict_missing_field():
    print("\n" + "=" * 60)
    print("TEST: POST /predict (missing audio_base64)")
    print("=" * 60)
    payload = json.dumps({"instances": [{"wrong_field": "abc"}]}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            print(f"  Status code: {resp.status}")
            print(f"  Response:    {json.dumps(body, indent=4)}")
            assert "error" in body["predictions"][0]
            print("  PASSED")
            return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_predict_empty():
    print("\n" + "=" * 60)
    print("TEST: POST /predict (empty instances)")
    print("=" * 60)
    payload = json.dumps({"instances": []}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"  FAILED: expected 400, got {resp.status}")
            return False
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print(f"  Status code: {e.code}")
        print(f"  Response:    {json.dumps(body, indent=4)}")
        assert e.code == 400
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    print("Parakeet EOU Handler — Local Integration Tests")
    print(f"Target: {BASE_URL}")
    print(f"Base image: nvcr.io/nvidia/pytorch:24.01-py3")
    print()

    results = [
        ("Health check",      test_health()),
        ("Single prediction", test_predict_single()),
        ("Batch prediction",  test_predict_batch()),
        ("Missing field",     test_predict_missing_field()),
        ("Empty instances",   test_predict_empty()),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
