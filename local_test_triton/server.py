"""
Vertex AI compatibility proxy for Triton Inference Server.

Translates between Vertex AI's prediction format and Triton's v2 API:
  - GET  /health  → Triton /v2/health/ready
  - POST /predict → Triton /v2/models/parakeet_asr/infer

This keeps the same API contract as the Flask+gunicorn version
so existing tests and load testing scripts work unchanged.
"""

import json
import logging
import os
import sys
import urllib.error
import urllib.request

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("triton_proxy")

app = Flask(__name__)

TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
MODEL_NAME = "parakeet_asr"

AIP_HEALTH_ROUTE  = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


@app.route(AIP_HEALTH_ROUTE, methods=["GET"])
def health():
    """Proxy health check to Triton."""
    try:
        req = urllib.request.Request(f"{TRITON_URL}/v2/health/ready")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.warning("Triton health check failed: %s", e)
    return jsonify({"status": "unhealthy"}), 503


@app.route(AIP_PREDICT_ROUTE, methods=["POST"])
def predict():
    """Translate Vertex AI predict request to Triton v2 inference.

    Vertex AI format:
        {"instances": [{"audio_base64": "..."}]}

    Triton v2 format:
        {"inputs": [{"name": "AUDIO_B64", "shape": [N,1], "datatype": "BYTES", "data": [...]}]}
    """
    data = request.get_json(force=True)
    instances = data.get("instances", [])

    if not instances:
        return jsonify({"error": "No instances provided."}), 400

    # Build Triton batch request — one inference call for all instances
    audio_data = []
    for inst in instances:
        audio_b64 = inst.get("audio_base64", "")
        audio_data.append(audio_b64)

    triton_request = {
        "inputs": [
            {
                "name": "AUDIO_B64",
                "shape": [len(audio_data), 1],
                "datatype": "BYTES",
                "data": audio_data,
            }
        ],
        "outputs": [
            {"name": "TRANSCRIPTION"},
            {"name": "EOU_DETECTED"},
        ],
    }

    # Send to Triton
    triton_url = f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer"
    triton_payload = json.dumps(triton_request).encode("utf-8")
    triton_req = urllib.request.Request(
        triton_url,
        data=triton_payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(triton_req, timeout=300) as resp:
            triton_resp = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("Triton inference failed: HTTP %d: %s", e.code, body[:500])
        return jsonify({"error": f"Triton inference failed: {body[:200]}"}), 502
    except Exception as e:
        logger.error("Triton inference failed: %s", e)
        return jsonify({"error": str(e)}), 502

    # Parse Triton response into Vertex AI format
    predictions = []
    transcriptions = []
    eou_flags = []

    for output in triton_resp.get("outputs", []):
        if output["name"] == "TRANSCRIPTION":
            transcriptions = output["data"]
        elif output["name"] == "EOU_DETECTED":
            eou_flags = output["data"]

    for i in range(len(instances)):
        text = transcriptions[i] if i < len(transcriptions) else ""
        eou = eou_flags[i] if i < len(eou_flags) else False
        predictions.append({
            "transcription": text,
            "end_of_utterance_detected": bool(eou),
        })

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    app.run(host="0.0.0.0", port=port)