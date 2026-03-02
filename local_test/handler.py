"""
Vertex AI prediction handler for NVIDIA Parakeet Realtime EOU 120M.

Routes
------
GET  /health   - liveness / readiness probe
POST /predict  - transcribe audio instances
"""

import os, sys, base64, tempfile, logging

import torch
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model loading - runs once when gunicorn imports this module
# -------------------------------------------------------------------
logger.info("Loading NVIDIA Parakeet Realtime EOU 120M model ...")

import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet_realtime_eou_120m-v1"
model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    logger.info("Model loaded on GPU: %s", torch.cuda.get_device_name(0))
else:
    logger.warning("No GPU detected - inference will be slow.")

logger.info("Model ready for inference.")

# -------------------------------------------------------------------
# Flask application
# -------------------------------------------------------------------
app = Flask(__name__)

AIP_HEALTH_ROUTE  = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


@app.route(AIP_HEALTH_ROUTE, methods=["GET"])
def health():
    """Vertex AI health / readiness probe."""
    return jsonify({"status": "healthy"}), 200


@app.route(AIP_PREDICT_ROUTE, methods=["POST"])
def predict():
    """Transcribe base64-encoded 16 kHz mono WAV audio.

    Request body
    ------------
    {
        "instances": [
            {"audio_base64": "<base64-encoded WAV bytes>"}
        ]
    }

    Response body
    -------------
    {
        "predictions": [
            {
                "transcription": "hello how are you<EOU>",
                "end_of_utterance_detected": true
            }
        ]
    }
    """
    data = request.get_json(force=True)
    instances = data.get("instances", [])

    if not instances:
        return jsonify({"error": "No instances provided."}), 400

    predictions = []

    for instance in instances:
        audio_b64 = instance.get("audio_base64", "")
        if not audio_b64:
            predictions.append(
                {"transcription": "", "end_of_utterance_detected": False,
                 "error": "Missing audio_base64."}
            )
            continue

        # Write decoded audio to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(audio_b64))
            tmp = f.name

        try:
            output = model.transcribe([tmp])
            text = (
                output[0].text
                if hasattr(output[0], "text")
                else str(output[0])
            )
            predictions.append({
                "transcription": text,
                "end_of_utterance_detected": "<EOU>" in text or "<eou>" in text,
            })
        except Exception as exc:
            logger.error("Transcription failed: %s", exc, exc_info=True)
            predictions.append(
                {"transcription": "", "end_of_utterance_detected": False,
                 "error": str(exc)}
            )
        finally:
            os.unlink(tmp)

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
