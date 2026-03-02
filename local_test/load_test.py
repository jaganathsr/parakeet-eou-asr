#!/usr/bin/env python3
"""
Load testing script for the Parakeet ASR /predict endpoint.

Measures throughput and latency under concurrent load. Supports single
concurrency tests, multi-level sweeps, and JSON output for comparing
implementations (e.g. Flask+gunicorn vs Triton).

Usage:
    # Quick test: 10 requests at concurrency 2
    python3 load_test.py --concurrency 2 --total-requests 10

    # Sweep mode: test multiple concurrency levels
    python3 load_test.py --sweep --requests-per-level 30

    # Save JSON for later comparison
    python3 load_test.py --sweep --output-json results.json --label flask-gunicorn

Third-party alternatives for more advanced load testing:

    Locust      pip install locust          Python-native, web UI, custom protocols
    k6          https://k6.io              High concurrency, CI integration
    vegeta      go install                 Constant-rate testing, HDR histograms
    wrk2        build from source          Raw throughput, constant-rate
"""

import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Audio generation (same as test_handler.py)
# ---------------------------------------------------------------------------

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


def build_payload(duration: float) -> bytes:
    """Build the JSON request body once; return bytes ready to POST."""
    wav_bytes = generate_wav_bytes(duration=duration)
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return json.dumps({"instances": [{"audio_base64": audio_b64}]}).encode("utf-8")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    success: bool
    response_size_bytes: int
    error: Optional[str] = None
    transcription: Optional[str] = None


@dataclass
class TestStatistics:
    concurrency: int
    audio_duration_sec: float
    total_requests: int
    successes: int
    failures: int
    error_rate_pct: float
    total_duration_sec: float
    throughput_rps: float
    latency_min_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_stddev_ms: float
    avg_response_size_bytes: float


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get_vertex_ai_token() -> Optional[str]:
    """Get an access token for Vertex AI endpoint authentication."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def send_single_request(
    url: str,
    payload: bytes,
    timeout: float = 120.0,
    auth_token: Optional[str] = None,
) -> RequestResult:
    """Send one POST request and measure latency."""
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    req = urllib.request.Request(url, data=payload, headers=headers)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            data = json.loads(body)
            transcription = None
            if "predictions" in data and data["predictions"]:
                transcription = data["predictions"][0].get("transcription")

            return RequestResult(
                latency_ms=elapsed_ms,
                status_code=resp.status,
                success=True,
                response_size_bytes=len(body),
                transcription=transcription,
            )
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        body = e.read()
        return RequestResult(
            latency_ms=elapsed_ms,
            status_code=e.code,
            success=False,
            response_size_bytes=len(body),
            error=f"HTTP {e.code}: {body[:200].decode('utf-8', errors='replace')}",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(
            latency_ms=elapsed_ms,
            status_code=0,
            success=False,
            response_size_bytes=0,
            error=str(e),
        )


def check_health(base_url: str, auth_token: Optional[str] = None) -> bool:
    """Send a GET /health and return True if healthy."""
    health_url = f"{base_url}/health"
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    req = urllib.request.Request(health_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if resp.status == 200 and body.get("status") == "healthy":
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Load test runner
# ---------------------------------------------------------------------------

def run_load_test(
    predict_url: str,
    concurrency: int,
    total_requests: int,
    payload: bytes,
    ramp_up_seconds: float = 0.0,
    auth_token: Optional[str] = None,
    timeout: float = 120.0,
) -> Tuple[List[RequestResult], float]:
    """Run a load test at a single concurrency level.

    Returns (results, wall_clock_duration_sec).
    """
    results: List[RequestResult] = []
    completed = 0

    print(f"\n  Sending {total_requests} requests at concurrency={concurrency} ...")

    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        if ramp_up_seconds > 0 and concurrency > 1:
            # Stagger submission to ramp up gradually
            delay_per_worker = ramp_up_seconds / concurrency
            futures = []
            for i in range(total_requests):
                if i < concurrency:
                    target_start = wall_start + i * delay_per_worker
                    now = time.perf_counter()
                    if target_start > now:
                        time.sleep(target_start - now)
                futures.append(
                    executor.submit(
                        send_single_request, predict_url, payload, timeout, auth_token,
                    )
                )
        else:
            futures = [
                executor.submit(
                    send_single_request, predict_url, payload, timeout, auth_token,
                )
                for _ in range(total_requests)
            ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % max(1, total_requests // 10) == 0 or completed == total_requests:
                pct = completed * 100 // total_requests
                print(f"    [{pct:3d}%] {completed}/{total_requests} completed", end="\r")

    wall_duration = time.perf_counter() - wall_start
    print()  # newline after progress
    return results, wall_duration


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(sorted_data: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def compute_statistics(
    results: List[RequestResult],
    wall_duration: float,
    concurrency: int,
    audio_duration: float,
) -> TestStatistics:
    """Compute aggregate statistics from request results."""
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    latencies = sorted(r.latency_ms for r in results)

    if not latencies:
        return TestStatistics(
            concurrency=concurrency,
            audio_duration_sec=audio_duration,
            total_requests=0, successes=0, failures=0,
            error_rate_pct=0, total_duration_sec=wall_duration,
            throughput_rps=0, latency_min_ms=0, latency_max_ms=0,
            latency_mean_ms=0, latency_median_ms=0,
            latency_p95_ms=0, latency_p99_ms=0, latency_stddev_ms=0,
            avg_response_size_bytes=0,
        )

    return TestStatistics(
        concurrency=concurrency,
        audio_duration_sec=audio_duration,
        total_requests=len(results),
        successes=successes,
        failures=failures,
        error_rate_pct=(failures / len(results)) * 100 if results else 0,
        total_duration_sec=wall_duration,
        throughput_rps=len(results) / wall_duration if wall_duration > 0 else 0,
        latency_min_ms=latencies[0],
        latency_max_ms=latencies[-1],
        latency_mean_ms=statistics.mean(latencies),
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_stddev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        avg_response_size_bytes=(
            statistics.mean(r.response_size_bytes for r in results)
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_histogram(latencies_ms: List[float], num_buckets: int = 12, width: int = 40) -> str:
    """Create a text-based histogram of latency distribution."""
    if not latencies_ms:
        return "  (no data)"

    lo = min(latencies_ms)
    hi = max(latencies_ms)
    if lo == hi:
        return f"  All requests: {lo:.1f} ms"

    bucket_width = (hi - lo) / num_buckets
    buckets = [0] * num_buckets
    for v in latencies_ms:
        idx = min(int((v - lo) / bucket_width), num_buckets - 1)
        buckets[idx] += 1

    max_count = max(buckets)
    lines = []
    for i, count in enumerate(buckets):
        lower = lo + i * bucket_width
        upper = lower + bucket_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "#" * bar_len
        pct = (count / len(latencies_ms)) * 100
        lines.append(f"  {lower:8.0f} - {upper:8.0f} ms | {bar:<{width}} | {count:4d} ({pct:5.1f}%)")

    return "\n".join(lines)


def print_report(stats: TestStatistics, results: List[RequestResult]) -> None:
    """Print a formatted human-readable report."""
    latencies = [r.latency_ms for r in results]

    print()
    print("=" * 64)
    print("  LOAD TEST RESULTS")
    print("=" * 64)
    print(f"  Audio duration:   {stats.audio_duration_sec:.1f} s")
    print(f"  Concurrency:      {stats.concurrency}")
    print(f"  Total requests:   {stats.total_requests}")
    print(f"  Wall-clock time:  {stats.total_duration_sec:.2f} s")
    print()
    print(f"  Successes:        {stats.successes} ({100 - stats.error_rate_pct:.1f}%)")
    print(f"  Failures:         {stats.failures} ({stats.error_rate_pct:.1f}%)")
    print()
    print(f"  Throughput:       {stats.throughput_rps:.2f} req/s")
    print()
    print("  Latency (ms):")
    print(f"    Min:            {stats.latency_min_ms:.1f}")
    print(f"    Max:            {stats.latency_max_ms:.1f}")
    print(f"    Mean:           {stats.latency_mean_ms:.1f}")
    print(f"    Median:         {stats.latency_median_ms:.1f}")
    print(f"    Std Dev:        {stats.latency_stddev_ms:.1f}")
    print(f"    P95:            {stats.latency_p95_ms:.1f}")
    print(f"    P99:            {stats.latency_p99_ms:.1f}")
    print()
    print(f"  Avg response:     {stats.avg_response_size_bytes:.0f} bytes")
    print()
    print("  Latency Distribution:")
    print(format_histogram(latencies))

    # Show errors if any
    errors = [r.error for r in results if r.error]
    if errors:
        print()
        print("  Errors:")
        # Deduplicate
        from collections import Counter
        for err, count in Counter(errors).most_common(5):
            err_short = err[:80]
            print(f"    - {err_short} ({count}x)")

    print("=" * 64)


def print_sweep_table(
    all_results: List[Tuple[int, TestStatistics, List[RequestResult]]],
    audio_duration: float,
) -> None:
    """Print a comparison table across concurrency levels."""
    print()
    print("=" * 90)
    print("  CONCURRENCY SWEEP RESULTS")
    print("=" * 90)
    print(f"  Audio duration: {audio_duration:.1f} s")
    print()

    header = (
        f"  {'Concurrency':>11} | {'Throughput':>10} | {'Mean':>10} | "
        f"{'Median':>10} | {'P95':>10} | {'P99':>10} | {'Errors':>6}"
    )
    units = (
        f"  {'':>11} | {'(req/s)':>10} | {'(ms)':>10} | "
        f"{'(ms)':>10} | {'(ms)':>10} | {'(ms)':>10} | {'':>6}"
    )
    separator = "  " + "-" * 86

    print(header)
    print(units)
    print(separator)

    for concurrency, stats, _ in all_results:
        row = (
            f"  {concurrency:>11} | {stats.throughput_rps:>10.2f} | "
            f"{stats.latency_mean_ms:>10.1f} | {stats.latency_median_ms:>10.1f} | "
            f"{stats.latency_p95_ms:>10.1f} | {stats.latency_p99_ms:>10.1f} | "
            f"{stats.failures:>6}"
        )
        print(row)

    print("=" * 90)


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def run_sweep(
    predict_url: str,
    concurrency_levels: List[int],
    requests_per_level: int,
    payload: bytes,
    ramp_up_seconds: float = 0.0,
    cooldown_seconds: float = 5.0,
    auth_token: Optional[str] = None,
    timeout: float = 120.0,
    audio_duration: float = 1.0,
) -> List[Tuple[int, TestStatistics, List[RequestResult]]]:
    """Run load tests at multiple concurrency levels."""
    all_results = []

    for i, concurrency in enumerate(concurrency_levels):
        print(f"\n{'='*64}")
        print(f"  SWEEP LEVEL {i+1}/{len(concurrency_levels)}: concurrency={concurrency}")
        print(f"{'='*64}")

        results, wall_duration = run_load_test(
            predict_url, concurrency, requests_per_level,
            payload, ramp_up_seconds, auth_token, timeout,
        )
        stats = compute_statistics(results, wall_duration, concurrency, audio_duration)
        all_results.append((concurrency, stats, results))

        # Brief summary for this level
        print(f"  Throughput: {stats.throughput_rps:.2f} req/s | "
              f"Mean: {stats.latency_mean_ms:.0f} ms | "
              f"P95: {stats.latency_p95_ms:.0f} ms | "
              f"Errors: {stats.failures}")

        # Cooldown between levels (skip after last)
        if i < len(concurrency_levels) - 1:
            # Refresh token if using Vertex AI (tokens expire after 60 min)
            if auth_token:
                new_token = get_vertex_ai_token()
                if new_token:
                    auth_token = new_token

            print(f"  Cooling down {cooldown_seconds:.0f}s ...")
            time.sleep(cooldown_seconds)

    return all_results


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def results_to_json(
    all_results: List[Tuple[int, TestStatistics, List[RequestResult]]],
    metadata: dict,
) -> dict:
    """Convert all results to a JSON-serializable dict."""
    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "implementation": metadata.get("label", "unknown"),
            "url": metadata.get("url", ""),
            "audio_duration_sec": metadata.get("audio_duration", 0),
        },
        "results": [
            {
                "concurrency": concurrency,
                "statistics": asdict(stats),
                "raw_latencies_ms": [r.latency_ms for r in results],
            }
            for concurrency, stats, results in all_results
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load test for Parakeet ASR /predict endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 10 requests, 2 concurrent workers
  python3 load_test.py --concurrency 2 --total-requests 10

  # Sweep mode: test concurrency levels 1,2,4,8,16
  python3 load_test.py --sweep --requests-per-level 30

  # Custom concurrency levels and longer audio
  python3 load_test.py --sweep --sweep-levels 1,2,4,8 --duration 5.0

  # Save results as JSON for comparing implementations
  python3 load_test.py --sweep --output-json results_flask.json --label flask-gunicorn

  # Test against Vertex AI endpoint
  python3 load_test.py \\
      --url "https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID" \\
      --vertex-ai --concurrency 4 --total-requests 20

Third-party load testing tools:
  Locust      pip install locust          Python-native, web UI
  k6          https://k6.io              High concurrency, CI integration
  vegeta      go install                 Constant-rate testing
  wrk2        build from source          Raw throughput measurement
        """,
    )

    parser.add_argument("--url", default="http://localhost:8080",
                        help="Base URL of the endpoint (default: http://localhost:8080)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    parser.add_argument("--total-requests", type=int, default=50,
                        help="Total number of requests to send (default: 50)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Audio duration in seconds (default: 2.0)")
    parser.add_argument("--ramp-up", type=float, default=0.0,
                        help="Ramp-up time in seconds (default: 0)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds (default: 120)")

    sweep = parser.add_argument_group("sweep mode")
    sweep.add_argument("--sweep", action="store_true",
                       help="Run tests at multiple concurrency levels")
    sweep.add_argument("--sweep-levels", default="1,2,4,8,16",
                       help="Comma-separated concurrency levels (default: 1,2,4,8,16)")
    sweep.add_argument("--requests-per-level", type=int, default=30,
                       help="Requests per concurrency level (default: 30)")
    sweep.add_argument("--cooldown", type=float, default=5.0,
                       help="Seconds between sweep levels (default: 5.0)")

    output = parser.add_argument_group("output")
    output.add_argument("--output-json", type=str, default=None,
                        help="Path to write JSON results")
    output.add_argument("--label", type=str, default="flask-gunicorn",
                        help="Label for this run (default: flask-gunicorn)")

    auth = parser.add_argument_group("authentication")
    auth.add_argument("--vertex-ai", action="store_true",
                      help="Authenticate with gcloud for Vertex AI endpoints")

    parser.add_argument("--skip-health-check", action="store_true",
                        help="Skip the initial health check")

    return parser.parse_args()


def main():
    args = parse_args()

    base_url = args.url.rstrip("/")

    # Determine predict URL
    if ":predict" in base_url:
        # Vertex AI format: URL already ends with :predict
        predict_url = base_url
        is_vertex_url = True
    else:
        predict_url = f"{base_url}/predict"
        is_vertex_url = False

    # Auth
    auth_token = None
    if args.vertex_ai:
        print("Fetching gcloud access token ...")
        auth_token = get_vertex_ai_token()
        if not auth_token:
            print("ERROR: Could not get gcloud access token.", file=sys.stderr)
            print("Run: gcloud auth application-default login", file=sys.stderr)
            sys.exit(1)
        print("  Token obtained.")

    # Generate payload
    print(f"Generating {args.duration:.1f}s test audio ...")
    payload = build_payload(args.duration)
    payload_kb = len(payload) / 1024
    print(f"  Payload size: {payload_kb:.1f} KB")

    if args.vertex_ai and payload_kb > 1500:
        print("  WARNING: Payload > 1.5 MB. Vertex AI may reject large requests.",
              file=sys.stderr)

    # Health check
    if not args.skip_health_check and not is_vertex_url:
        print(f"\nHealth check: GET {base_url}/health")
        if check_health(base_url, auth_token):
            print("  Server is healthy.")
        else:
            print("  ERROR: Health check failed. Is the server running?", file=sys.stderr)
            sys.exit(1)

    print(f"\nTarget: {predict_url}")
    print(f"Label:  {args.label}")

    if args.sweep:
        levels = [int(x) for x in args.sweep_levels.split(",")]
        all_results = run_sweep(
            predict_url, levels, args.requests_per_level,
            payload, args.ramp_up, args.cooldown,
            auth_token, args.timeout, args.duration,
        )

        # Print sweep comparison table
        print_sweep_table(all_results, args.duration)

        # Print detailed report for each level
        for concurrency, stats, results in all_results:
            print_report(stats, results)
    else:
        results, wall_duration = run_load_test(
            predict_url, args.concurrency, args.total_requests,
            payload, args.ramp_up, auth_token, args.timeout,
        )
        stats = compute_statistics(
            results, wall_duration, args.concurrency, args.duration,
        )
        print_report(stats, results)
        all_results = [(args.concurrency, stats, results)]

    # JSON output
    if args.output_json:
        output = results_to_json(all_results, {
            "url": predict_url,
            "audio_duration": args.duration,
            "label": args.label,
        })
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON results written to: {args.output_json}")


if __name__ == "__main__":
    main()
