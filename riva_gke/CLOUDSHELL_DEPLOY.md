# Deploy Riva NIM Parakeet ASR on GKE — Cloud Shell Guide

Step-by-step commands to deploy NVIDIA Riva NIM with Parakeet 1.1B CTC on GKE,
designed to be run directly in [Google Cloud Shell](https://shell.cloud.google.com).

Cloud Shell has `gcloud`, `kubectl`, and `helm` pre-installed — no setup needed.

> **Assumes the GKE cluster is already created.** If not, see the
> "Create Cluster" section at the bottom of this document.

---

## Step 0 — Set Variables

Copy this entire block, replace the placeholder values, and paste into Cloud Shell:

```bash
# ── EDIT THESE ──
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export NGC_API_KEY="your-ngc-api-key"

# ── Leave these as-is ──
export CLUSTER_NAME="riva-asr-cluster"
export RIVA_MODEL="parakeet-1-1b-ctc-en-us"

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE
```

---

## Step 1 — Connect to the Cluster

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
kubectl get nodes
```

You should see your node(s) listed. Verify the GPU is available:

```bash
kubectl get nodes -o=custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
```

Expected output should show `1` under the GPU column.

---

## Step 2 — Install NVIDIA GPU Drivers

GKE nodes need the NVIDIA device plugin to expose GPUs to pods:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

Wait ~60 seconds, then verify:

```bash
kubectl get pods -n kube-system -l k8s-app=nvidia-driver-installer
```

All pods should show `Running` or `Completed`.

---

## Step 3 — Create NGC Secrets

These secrets allow the cluster to pull the Riva container from NGC and download model weights:

```bash
# Image pull secret
kubectl create secret docker-registry ngc-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password=$NGC_API_KEY \
    --dry-run=client -o yaml | kubectl apply -f -

# API key secret
kubectl create secret generic ngc-api \
    --from-literal=NGC_API_KEY=$NGC_API_KEY \
    --dry-run=client -o yaml | kubectl apply -f -
```

Verify:

```bash
kubectl get secrets
```

You should see `ngc-secret` and `ngc-api` listed.

---

## Step 4 — Install Helm and Add NVIDIA Repo

Cloud Shell has Helm pre-installed. Add the NVIDIA NIM chart repository:

```bash
helm repo add nim https://helm.ngc.nvidia.com/nim/nvidia \
    --username='$oauthtoken' --password=$NGC_API_KEY

helm repo update nim
```

Verify the Riva chart is available:

```bash
helm search repo nim/riva-nim
```

---

## Step 5 — Create the Helm Values File

```bash
cat > /tmp/riva-values.yaml << 'EOF'
image:
  repository: nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us
  tag: latest
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: ngc-secret

nim:
  ngcAPISecret: ngc-api

env:
  - name: NIM_TAGS_SELECTOR
    value: "name=parakeet-1-1b-ctc-en-us,mode=str,vad=default"

resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

service:
  type: LoadBalancer
  ports:
    - name: http
      port: 9000
      targetPort: 9000
      protocol: TCP
    - name: grpc
      port: 50051
      targetPort: 50051
      protocol: TCP

persistence:
  enabled: true
  size: 50Gi

startupProbe:
  httpGet:
    path: /v1/health/ready
    port: 9000
  initialDelaySeconds: 60
  periodSeconds: 30
  failureThreshold: 40
  timeoutSeconds: 10

livenessProbe:
  httpGet:
    path: /v1/health/ready
    port: 9000
  periodSeconds: 30
  failureThreshold: 3
  timeoutSeconds: 10
EOF
```

---

## Step 6 — Deploy Riva NIM

```bash
helm install riva-asr nim/riva-nim -f /tmp/riva-values.yaml
```

This starts the deployment. The pod will download the model and optimize it
with TensorRT on first run (**15-30 minutes**).

---

## Step 7 — Monitor Deployment

Watch the pod status until it shows `Running` and `1/1 READY`:

```bash
kubectl get pods -w
```

(Press `Ctrl+C` to stop watching once the pod is ready.)

To see detailed progress (model download, TensorRT optimization):

```bash
kubectl logs -f -l app.kubernetes.io/instance=riva-asr
```

If the pod is stuck in `Pending`, check events:

```bash
kubectl describe pod -l app.kubernetes.io/instance=riva-asr
```

---

## Step 8 — Get the External IP

Once the pod is running, get the LoadBalancer external IP:

```bash
kubectl get svc riva-asr-riva-nim
```

Look for the `EXTERNAL-IP` column. If it shows `<pending>`, wait a minute and
re-run. Save it to a variable:

```bash
export EXTERNAL_IP=$(kubectl get svc riva-asr-riva-nim \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "External IP: $EXTERNAL_IP"
```

---

## Step 9 — Verify Health

```bash
curl -s http://$EXTERNAL_IP:9000/v1/health/ready | python3 -m json.tool
```

Expected output:

```json
{
    "ready": true
}
```

---

## Step 10 — Test Offline Transcription (HTTP)

Generate a test WAV file and send it to the HTTP API:

```bash
# Install test dependencies
pip install -q gTTS pydub

# Generate test audio
python3 -c "
from gtts import gTTS
from pydub import AudioSegment
tts = gTTS('Hello, what is the weather like today?', lang='en')
tts.save('/tmp/test.mp3')
audio = AudioSegment.from_mp3('/tmp/test.mp3').set_frame_rate(16000).set_channels(1).set_sample_width(2)
audio.export('/tmp/test.wav', format='wav')
print('Test audio generated: /tmp/test.wav')
"

# Send to Riva HTTP API
curl -s http://$EXTERNAL_IP:9000/v1/audio/transcriptions \
    -F language=en \
    -F file=@/tmp/test.wav | python3 -m json.tool
```

---

## Step 11 — Test gRPC Streaming

```bash
# Install Riva client
pip install -q nvidia-riva-client

# Clone the official Python client scripts
git clone --depth 1 https://github.com/nvidia-riva/python-clients.git /tmp/riva-clients 2>/dev/null

# Run streaming transcription
python3 /tmp/riva-clients/scripts/asr/transcribe_file.py \
    --server $EXTERNAL_IP:50051 \
    --language-code en-US \
    --input-file /tmp/test.wav
```

You should see interim and final transcription results printed as the audio
is streamed to the server.

For a more detailed streaming test with interim results visible:

```bash
python3 << 'PYEOF'
import os
import riva.client

EXTERNAL_IP = os.environ["EXTERNAL_IP"]

auth = riva.client.Auth(uri=f"{EXTERNAL_IP}:50051", use_ssl=False)
asr_service = riva.client.ASRService(auth)

config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
    ),
    interim_results=True,
)

CHUNK_SIZE = 8000  # 250ms of 16kHz 16-bit mono

with open("/tmp/test.wav", "rb") as f:
    f.read(44)  # skip WAV header
    chunks = []
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)

print(f"Streaming {len(chunks)} chunks ...\n")

responses = asr_service.streaming_response_generator(
    audio_chunks=chunks,
    streaming_config=config,
)

final = ""
for resp in responses:
    for result in resp.results:
        text = result.alternatives[0].transcript
        if result.is_final:
            final += text
            print(f"  [FINAL]   {text}")
        else:
            print(f"  [INTERIM] {text}")

print(f"\nFull transcription: \"{final}\"")
PYEOF
```

---

## Step 12 — Test WebSocket Realtime API

```bash
pip install -q websocket-client

python3 << 'PYEOF'
import os, json, base64, threading, websocket

EXTERNAL_IP = os.environ["EXTERNAL_IP"]
ws_url = f"ws://{EXTERNAL_IP}:9000/v1/realtime?intent=transcription"
print(f"Connecting to {ws_url} ...\n")

results = []
done = threading.Event()

def on_message(ws, msg):
    event = json.loads(msg)
    t = event.get("type", "")
    if t == "conversation.item.input_audio_transcription.delta":
        text = event.get("delta", "")
        print(f"  [PARTIAL] {text}")
        results.append(text)
    elif t == "conversation.item.input_audio_transcription.completed":
        text = event.get("transcript", "")
        print(f"  [FINAL]   {text}")
        results.append(text)
        done.set()
    elif t == "error":
        print(f"  [ERROR]   {event}")
        done.set()

def on_open(ws):
    ws.send(json.dumps({
        "type": "transcription_session.update",
        "input_audio_format": "pcm16",
        "language": "en-US",
    }))
    with open("/tmp/test.wav", "rb") as f:
        f.read(44)
        while True:
            chunk = f.read(8000)
            if not chunk:
                break
            ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            }))
    ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    ws.send(json.dumps({"type": "input_audio_buffer.done"}))

def on_error(ws, err):
    print(f"  [ERROR] {err}")
    done.set()

app = websocket.WebSocketApp(ws_url, on_open=on_open,
                              on_message=on_message, on_error=on_error)
t = threading.Thread(target=app.run_forever, kwargs={"ping_interval": 30})
t.daemon = True
t.start()
done.wait(timeout=30)
app.close()

if results:
    print(f"\nFinal: \"{results[-1]}\"")
else:
    print("\nNo results (timeout)")
PYEOF
```

---

## Cleanup

When you're done, delete everything to stop charges (~$0.83/hr):

```bash
# Remove the Helm release
helm uninstall riva-asr

# Delete the GKE cluster
gcloud container clusters delete $CLUSTER_NAME --zone $ZONE --quiet
```

---

## Appendix: Create the GKE Cluster

If you haven't created the cluster yet, run this first:

```bash
gcloud container clusters create $CLUSTER_NAME \
    --zone $ZONE \
    --machine-type n1-standard-8 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes 1 \
    --scopes cloud-platform \
    --quiet
```

This takes 3-5 minutes. Then continue from Step 1 above.

---

## Troubleshooting

### Pod stuck in `Pending`
```bash
kubectl describe pod -l app.kubernetes.io/instance=riva-asr
```
Common causes: no GPU available (check quota), insufficient CPU/memory on the node.

### Pod in `CrashLoopBackOff`
```bash
kubectl logs -l app.kubernetes.io/instance=riva-asr --previous
```
Common causes: invalid NGC_API_KEY, model download failure, insufficient GPU memory.

### LoadBalancer IP stays `<pending>`
```bash
kubectl describe svc riva-asr-riva-nim
```
Check if firewall rules are blocking. You can also use port-forwarding as a workaround:
```bash
kubectl port-forward svc/riva-asr-riva-nim 9000:9000 50051:50051
# Then use localhost instead of EXTERNAL_IP
```

### Health check returns `{"ready": false}`
The model is still loading. Check logs for TensorRT optimization progress:
```bash
kubectl logs -f -l app.kubernetes.io/instance=riva-asr
```

### Helm chart not found
```bash
helm repo update nim
helm search repo nim/riva-nim --versions
```
If empty, re-add the repo with correct NGC credentials.
