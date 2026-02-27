# Deploying NVIDIA Parakeet Realtime EOU 120M on Vertex AI

Documentation for `deploy_parakeet_eou_vertex_ai.ipynb`.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
  - [Serving Container](#serving-container)
  - [Prediction Handler](#prediction-handler)
  - [Vertex AI Deployment](#vertex-ai-deployment)
- [API Reference](#api-reference)
  - [Health Endpoint](#health-endpoint)
  - [Predict Endpoint](#predict-endpoint)
- [Usage](#usage)
  - [Running the Notebook](#running-the-notebook)
  - [Sending Predictions Programmatically](#sending-predictions-programmatically)
  - [Using Custom Audio Files](#using-custom-audio-files)
- [Infrastructure and Cost](#infrastructure-and-cost)
  - [GPU Selection](#gpu-selection)
  - [Estimated Costs](#estimated-costs)
  - [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)

---

## Overview

This project deploys the [NVIDIA Parakeet Realtime EOU 120M](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) model as an online prediction endpoint on Google Cloud Vertex AI, runnable end-to-end from a Google Colab notebook.

**Parakeet Realtime EOU 120M** is a streaming automatic speech recognition (ASR) model that jointly performs:

1. **Speech-to-text transcription** — converts audio to text in real time.
2. **End-of-utterance (EOU) detection** — emits a special `<EOU>` token when it detects that the speaker has finished an utterance (e.g. `"what is your name<EOU>"`).

| Property | Value |
|---|---|
| Architecture | FastConformer-RNNT (Cache-Aware Streaming) |
| Parameters | 120 M |
| Framework | NVIDIA NeMo (`nemo_toolkit[asr]`) |
| Language | English |
| Input | 16 kHz, mono, WAV audio (minimum 160 ms) |
| Chunk size | 80 ms |
| EOU latency (p50) | 160 ms |
| EOU latency (p90) | 280 ms |
| License | NVIDIA Open Model License |

The model is designed for interactive voice-agent pipelines where low-latency turn-taking is critical. The `<EOU>` signal can trigger downstream actions (e.g. sending the transcript to an LLM) as soon as the speaker stops talking.

---

## Architecture

The deployment follows a standard Vertex AI custom-container pattern:

```
┌──────────────────────────────────────────────────────────────┐
│  Google Colab                                                │
│                                                              │
│  1. Authenticate (OAuth)                                     │
│  2. Write handler.py + Dockerfile                            │
│  3. gcloud builds submit  ──►  Cloud Build                   │
│  4. aiplatform.Model.upload()                                │
│  5. aiplatform.Endpoint.create()                             │
│  6. model.deploy()                                           │
│  7. endpoint.predict()                                       │
└──────────────────────────────────────────────────────────────┘
         │                            │
         │ (3) Build & push           │ (6) Pull & run
         ▼                            ▼
┌─────────────────┐        ┌─────────────────────────────────┐
│ Artifact Registry│        │ Vertex AI Endpoint              │
│                 │        │                                 │
│ parakeet-eou-   │◄───────│  n1-standard-4 + NVIDIA T4 GPU  │
│ serving:v1      │        │                                 │
└─────────────────┘        │  ┌───────────────────────────┐  │
                           │  │  Docker Container          │  │
                           │  │                           │  │
                           │  │  gunicorn (1 worker)      │  │
                           │  │    └─ Flask app            │  │
                           │  │        ├─ /health (GET)    │  │
                           │  │        └─ /predict (POST)  │  │
                           │  │                           │  │
                           │  │  NeMo ASR model (GPU)     │  │
                           │  └───────────────────────────┘  │
                           └─────────────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Notebook runtime | Google Colab | Orchestration, testing |
| Authentication | `google.colab.auth` (OAuth) | Colab-to-GCP credential bridge |
| Container build | Cloud Build | Builds Docker image remotely |
| Container registry | Artifact Registry | Stores the serving image |
| Model registry | Vertex AI Model Registry | Tracks the model resource |
| Serving endpoint | Vertex AI Prediction | Hosts the model for online inference |
| Base image | `nvcr.io/nvidia/pytorch:24.01-py3` | CUDA 12 + PyTorch runtime |
| ML framework | NeMo Toolkit (ASR) | Model loading and inference |
| HTTP server | Flask + gunicorn | Serves health and prediction routes |
| GPU | NVIDIA T4 (16 GB VRAM) | Accelerates inference |

---

## Prerequisites

### Google Cloud Setup

1. A **Google Cloud project** with billing enabled.
2. The following **APIs** must be enabled on the project:

   | API | Console link |
   |---|---|
   | Vertex AI API | `console.cloud.google.com/apis/api/aiplatform.googleapis.com` |
   | Artifact Registry API | `console.cloud.google.com/apis/api/artifactregistry.googleapis.com` |
   | Cloud Build API | `console.cloud.google.com/apis/api/cloudbuild.googleapis.com` |

3. Your user account (or service account) needs these **IAM roles**:

   | Role | Why |
   |---|---|
   | `roles/aiplatform.user` | Upload models, create endpoints, deploy, predict |
   | `roles/artifactregistry.writer` | Push container images |
   | `roles/cloudbuild.builds.editor` | Submit Cloud Build jobs |
   | `roles/storage.admin` | Read/write staging buckets used by Vertex AI |

### Python Dependencies

Installed automatically in the notebook:

| Package | Version | Purpose |
|---|---|---|
| `google-cloud-aiplatform` | latest | Vertex AI SDK |
| `gTTS` | latest | Generate test speech audio |
| `pydub` | latest | Audio format conversion |

### Container Dependencies (inside Docker image)

| Package | Purpose |
|---|---|
| `nemo_toolkit[asr]` | NeMo ASR model loading and inference |
| `flask` | HTTP serving |
| `gunicorn` | Production WSGI server |
| `soundfile` | Audio I/O |
| `torchvision` | Reinstalled for NeMo/PyTorch version compatibility |
| `torchaudio` | Reinstalled for NeMo/PyTorch version compatibility |
| `torch` | PyTorch (included in base image) |
| `libsndfile1`, `ffmpeg`, `sox` | System-level audio libraries |

---

## Configuration

All configurable values are in a single notebook cell with Colab form annotations:

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ID` | `"your-project-id"` | Your GCP project ID. **Must be changed.** |
| `REGION` | `"us-central1"` | GCP region for all resources. Must have T4 GPU availability. |
| `REPOSITORY` | `"ml-serving"` | Artifact Registry Docker repository name. |
| `IMAGE_NAME` | `"parakeet-eou-serving"` | Docker image name. |
| `IMAGE_TAG` | `"v1"` | Docker image tag. |

Derived values (computed automatically):

| Variable | Value |
|---|---|
| `CONTAINER_IMAGE_URI` | `{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}` |
| `ENDPOINT_DISPLAY_NAME` | `"parakeet-eou-endpoint"` |
| `MODEL_DISPLAY_NAME` | `"parakeet-realtime-eou-120m"` |

---

## Implementation Details

### Serving Container

The Docker container is built on `nvcr.io/nvidia/pytorch:24.01-py3` (CUDA 12.3, PyTorch 2.2).

**Build stages:**

1. **System libraries** — installs `libsndfile1`, `ffmpeg`, and `sox` for audio processing.
2. **Python dependencies** — installs NeMo ASR toolkit, Flask, gunicorn, soundfile, and explicitly reinstalls `torchvision` and `torchaudio` to ensure version compatibility with whatever PyTorch version NeMo resolves.
3. **Model pre-download** — runs `ASRModel.from_pretrained()` during the build to cache the model weights inside the image. This avoids a HuggingFace download on every container startup, making deployments faster and more reliable.
4. **Handler copy** — copies `handler.py` into the image.

**Runtime configuration:**

The container reads three environment variables set by Vertex AI:

| Variable | Default | Description |
|---|---|---|
| `AIP_HTTP_PORT` | `8080` | Port gunicorn listens on |
| `AIP_HEALTH_ROUTE` | `/health` | Health check path |
| `AIP_PREDICT_ROUTE` | `/predict` | Prediction path |

**Gunicorn settings:**

| Flag | Value | Rationale |
|---|---|---|
| `--workers` | `1` | One GPU = one worker to avoid CUDA context conflicts |
| `--threads` | `2` | Light concurrency for I/O-bound pre/post-processing |
| `--timeout` | `300` | Generous timeout for model loading at startup |

### Prediction Handler

`handler.py` is a Flask application with two routes. The model is loaded **once** at module import time (when gunicorn starts the worker process):

```
gunicorn starts
  └─ imports handler.py
       └─ ASRModel.from_pretrained()    # loads from cache (~10s)
       └─ model.eval()
       └─ model.cuda()                  # moves to GPU
       └─ Flask app is ready
            ├─ GET  /health   → 200 OK
            └─ POST /predict  → transcription results
```

**Prediction flow for each instance:**

1. Decode `audio_base64` from the request JSON.
2. Write decoded bytes to a temporary `.wav` file on disk.
3. Call `model.transcribe([tmp_path])` — NeMo reads the WAV and runs inference.
4. Extract `.text` from the output (e.g. `"hello how are you<EOU>"`).
5. Check for `<EOU>` / `<eou>` tokens and set the `end_of_utterance_detected` flag.
6. Delete the temp file and return the result.

### Vertex AI Deployment

The deployment uses three Vertex AI SDK calls:

1. **`Model.upload()`** — registers the container image as a model in the Vertex AI Model Registry. No `artifact_uri` is needed because the model weights are baked into the container.

2. **`Endpoint.create()`** — creates a Vertex AI Endpoint resource that will serve as the public prediction URL.

3. **`model.deploy()`** — deploys the model to the endpoint with the specified machine type and GPU. Key parameters:

   | Parameter | Value | Notes |
   |---|---|---|
   | `machine_type` | `n1-standard-4` | 4 vCPUs, 15 GB RAM |
   | `accelerator_type` | `NVIDIA_TESLA_T4` | 16 GB VRAM |
   | `accelerator_count` | `1` | Single GPU |
   | `min_replica_count` | `1` | Always-on (no scale-to-zero) |
   | `max_replica_count` | `1` | No autoscaling (change for production) |
   | `deploy_request_timeout` | `3600` | 1 hour — allows for container pull + model init |

---

## API Reference

Once deployed, the endpoint accepts requests via `endpoint.predict()` (Python SDK) or the Vertex AI REST API.

### Health Endpoint

```
GET /health
```

**Response:**

```json
{
  "status": "healthy"
}
```

Returns `200 OK` when the model is loaded and the server is ready.

### Predict Endpoint

```
POST /predict
Content-Type: application/json
```

**Request body:**

```json
{
  "instances": [
    {
      "audio_base64": "<base64-encoded 16 kHz mono WAV bytes>"
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `instances` | array | yes | List of audio instances to transcribe |
| `instances[].audio_base64` | string | yes | Base64-encoded WAV audio (16 kHz, mono, 16-bit PCM) |

**Response body:**

```json
{
  "predictions": [
    {
      "transcription": "hello what is the weather like today<EOU>",
      "end_of_utterance_detected": true
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `predictions[].transcription` | string | Raw transcription text (lowercase, no punctuation). Contains `<EOU>` if end-of-utterance was detected. |
| `predictions[].end_of_utterance_detected` | boolean | `true` if the transcription contains an `<EOU>` token. |
| `predictions[].error` | string | Present only if transcription failed for this instance. |

**Audio requirements:**

| Property | Requirement |
|---|---|
| Format | WAV (PCM) |
| Sample rate | 16,000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit |
| Minimum duration | 160 ms |
| Encoding | Base64 |

---

## Usage

### Running the Notebook

1. Open `deploy_parakeet_eou_vertex_ai.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Update `PROJECT_ID` in the configuration cell to your GCP project ID.
3. Run all cells sequentially. Expected timeline:

   | Step | Duration |
   |---|---|
   | Install + authenticate | ~1 min |
   | Cloud Build (container) | 20–40 min |
   | Model upload | ~1 min |
   | Endpoint deploy | 10–15 min |
   | Test predictions | ~30 sec |

4. Total time from start to first prediction: **~35–60 minutes**.

### Sending Predictions Programmatically

After deployment, you can call the endpoint from any Python environment:

```python
from google.cloud import aiplatform
import base64

aiplatform.init(project="your-project-id", location="us-central1")

# Get reference to existing endpoint
endpoint = aiplatform.Endpoint("projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID")

# Read and encode audio
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send prediction
response = endpoint.predict(instances=[{"audio_base64": audio_b64}])

for pred in response.predictions:
    print(pred["transcription"])
    print(pred["end_of_utterance_detected"])
```

### Using Custom Audio Files

The notebook includes a cell that uses Colab's file upload widget:

```python
from google.colab import files
uploaded = files.upload()  # pick a .wav file
```

If your audio is not in the required format, convert it first:

```python
from pydub import AudioSegment

audio = (
    AudioSegment.from_file("input.mp3")
    .set_frame_rate(16000)
    .set_channels(1)
    .set_sample_width(2)
)
audio.export("output.wav", format="wav")
```

### Calling via REST API (curl)

```bash
ENDPOINT_ID="1234567890"
PROJECT_ID="your-project-id"
REGION="us-central1"

# Base64-encode your audio
AUDIO_B64=$(base64 -w 0 audio.wav)

curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict" \
  -d "{\"instances\": [{\"audio_base64\": \"${AUDIO_B64}\"}]}"
```

---

## Infrastructure and Cost

### GPU Selection

The NVIDIA T4 was chosen for this deployment:

| Factor | Detail |
|---|---|
| Model size | 120 M parameters (~480 MB FP32, ~240 MB FP16) |
| T4 VRAM | 16 GB — over 30x the model's memory requirement |
| T4 cost | ~$0.35/hr on Vertex AI (cheapest GPU option) |
| Alternative | L4 on `g2-standard-4` ($0.70/hr) — faster but unnecessary for this model |
| Overkill | A100/H100 — significantly more expensive with no benefit for 120M params |

The T4 provides ample headroom for batch inference and concurrent requests while being the most cost-effective GPU on Vertex AI.

### Estimated Costs

| Resource | Cost | Notes |
|---|---|---|
| **Vertex AI endpoint** (T4) | ~$0.35/hr | Per node-hour while deployed |
| Cloud Build | ~$0.003/build-min | ~$0.10 for a 30-min build |
| Artifact Registry | ~$0.10/GB/month | Container image storage |
| Prediction requests | Included | No per-request charge beyond node cost |

**Example:** Running the endpoint for 8 hours costs approximately **$2.80** in compute.

### Scaling

The notebook deploys with `min_replica_count=1, max_replica_count=1` (no autoscaling). For production, consider:

```python
model.deploy(
    ...
    min_replica_count=1,
    max_replica_count=5,    # scale up under load
)
```

Vertex AI autoscaling monitors CPU/GPU utilization and adjusts replicas automatically. Each replica runs on its own `n1-standard-4` + T4 node.

---

## Troubleshooting

### Cloud Build times out

The default timeout in the notebook is 3600s (1 hour). If the build exceeds this:

- Increase `--timeout` in the `gcloud builds submit` command.
- Use a faster machine type: `--machine-type=e2-highcpu-32`.
- Check Cloud Build logs in the GCP Console for the specific failure.

### Cloud Build cannot pull the NVIDIA base image

The `nvcr.io/nvidia/pytorch:24.01-py3` image is publicly accessible. If pulls fail:

- Verify network connectivity from Cloud Build.
- Try a different tag (e.g. `24.03-py3`, `23.10-py3`).
- Check [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for available tags.

### Deployment fails (health check timeout)

If the container fails to pass health checks:

- Check container logs in the GCP Console (Vertex AI > Endpoints > Logs).
- Ensure the model was pre-downloaded during the Docker build.
- Increase `deploy_request_timeout` if the container needs more startup time.
- Verify the health route matches: the handler uses `/health` and the `Model.upload()` call specifies `serving_container_health_route="/health"`.

### Prediction returns an error

- **"Missing audio_base64"** — the request JSON must include an `audio_base64` field in each instance.
- **Transcription error** — check that the audio is valid WAV format, 16 kHz, mono. The model requires a minimum of 160 ms of audio.
- **Timeout** — for very long audio files, the default gunicorn timeout (300s) may be insufficient. Increase `--timeout` in the Dockerfile `CMD`.

### Quota errors

If you see quota errors during deployment:

- Check GPU quota for `NVIDIA_T4` in your region: GCP Console > IAM & Admin > Quotas.
- Request a quota increase if needed.
- Try a different region (e.g. `us-east1`, `europe-west4`).

### Authentication issues in Colab

- Re-run `auth.authenticate_user()` if your session has expired.
- Ensure your account has the required IAM roles listed in [Prerequisites](#prerequisites).
- For non-Colab environments, run `gcloud auth application-default login`.

---

## Cleanup

To avoid ongoing charges, delete all resources after use:

```python
# 1. Undeploy the model from the endpoint
endpoint.undeploy_all()

# 2. Delete the endpoint
endpoint.delete()

# 3. Delete the model from the registry
model.delete()
```

Optionally, delete the container image from Artifact Registry:

```bash
gcloud artifacts docker images delete \
    REGION-docker.pkg.dev/PROJECT_ID/ml-serving/parakeet-eou-serving:v1 \
    --quiet
```

And delete the Artifact Registry repository if no longer needed:

```bash
gcloud artifacts repositories delete ml-serving \
    --location=REGION \
    --quiet
```
