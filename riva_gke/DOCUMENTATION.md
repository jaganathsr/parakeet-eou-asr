# Realtime Streaming ASR: NVIDIA Riva on GKE vs Vertex AI

## Overview

This document explains why a Riva NIM deployment on Google Kubernetes Engine (GKE) is the recommended architecture for realtime streaming audio transcription, compared to a Vertex AI endpoint serving the same Parakeet ASR model.

Both approaches use NVIDIA Parakeet models for English speech recognition. The critical difference is in **how audio reaches the model and how results are returned** — not the model itself.

## The Core Problem with Vertex AI for Realtime Audio

Vertex AI prediction endpoints are built around HTTP request-response:

1. Client records audio
2. Client encodes the **entire** audio clip as base64
3. Client sends a single `POST /predict` request
4. Server transcribes the complete clip
5. Server returns the full transcription

This works for **batch/offline transcription** — you have a finished recording and want the text. But for a live microphone, phone call, or voice assistant, this model breaks down:

- You don't have the "complete audio" — the person is still speaking
- You can't wait until they finish to start transcribing
- The user expects to see words appearing as they speak
- The application needs to know when the speaker pauses (end-of-utterance) to trigger downstream actions like sending to an LLM

**Vertex AI has no mechanism for maintaining a persistent, bidirectional audio stream.** Each request is stateless and independent.

## How Riva Solves This

Riva uses **gRPC bidirectional streaming** and **WebSocket connections** — protocols designed for continuous, two-way data flow:

1. Client opens a persistent connection (gRPC stream or WebSocket)
2. Client sends small audio chunks (~250ms) as they're captured from the mic
3. Server processes each chunk incrementally, maintaining state from previous chunks
4. Server sends back **partial transcriptions** as soon as they're computed (~200ms latency)
5. When the speaker pauses, server sends a **final transcription** for that utterance
6. Connection stays open — repeat from step 2

The client sees something like:
```
[INTERIM]  "hello"
[INTERIM]  "hello how"
[INTERIM]  "hello how are"
[FINAL]    "Hello, how are you?"    ← speaker paused here
[INTERIM]  "I'm"
[INTERIM]  "I'm doing"
[FINAL]    "I'm doing great."
```

## Detailed Comparison

### Latency

| Metric | Vertex AI Endpoint | Riva on GKE |
|--------|-------------------|-------------|
| Time to first result | 1-3 seconds (full request-response cycle) | ~200ms (first interim result) |
| Per-utterance latency | Audio duration + inference + network round-trip | Overlapped with audio capture |
| Perceived responsiveness | User waits for entire clip to process | Words appear as user speaks |

With Vertex AI, a 5-second utterance takes at least 5 seconds of recording + inference time before any text appears. With Riva, the user sees partial text within 200ms of starting to speak.

### End-of-Utterance (EOU) Detection

| Aspect | Vertex AI (Parakeet EOU 120M) | Riva (Parakeet 1.1B CTC) |
|--------|-------------------------------|--------------------------|
| Method | Model outputs `<EOU>` tokens in transcription text | Pipeline-level: silence detection + Silero VAD + Two-Pass EOU |
| When detected | After the entire audio clip is processed | In realtime, as the speaker pauses |
| Accuracy | Depends on where the audio was clipped | VAD-based detection is more reliable for natural speech |
| Configurability | None — baked into model output | Adjustable thresholds: `stop_history(ms)`, VAD sensitivity |
| Two-Pass EOU | Not available | Supported — sends early EOU for faster downstream processing |

Two-Pass EOU is particularly valuable for voice assistants: the system can start processing the LLM response as soon as the early EOU fires, while the more accurate final transcript is still being computed.

### Streaming Protocol

| Aspect | Vertex AI | Riva on GKE |
|--------|-----------|-------------|
| Protocol | HTTP/1.1 POST | gRPC (HTTP/2) + WebSocket |
| Connection model | Stateless request-response | Persistent bidirectional stream |
| Audio format | Base64-encoded WAV in JSON body | Raw PCM16 chunks (gRPC) or base64 chunks (WebSocket) |
| Partial results | Not possible | Interim transcriptions after each chunk |
| Session state | None — each request is independent | Server maintains model cache across chunks |
| Max payload | ~1.5 MB (Vertex AI limit) | Unlimited (streamed incrementally) |

The base64-encoded WAV format used by Vertex AI adds ~33% overhead to the audio data. Riva's gRPC endpoint accepts raw PCM bytes with zero encoding overhead.

### Model Optimization

| Aspect | Vertex AI (our Triton setup) | Riva NIM |
|--------|-----|----------|
| Runtime | PyTorch (NeMo) | TensorRT |
| Inference speed | Baseline | 2-5x faster |
| GPU memory efficiency | Higher (PyTorch overhead) | Lower (optimized engine) |
| Concurrent streams | Limited by PyTorch memory usage | Optimized for multi-stream batching |

Riva NIM containers include pre-optimized TensorRT engines. The TensorRT conversion happens automatically at first startup and is cached for subsequent runs. This optimization is significant — the same T4 GPU can serve 2-5x more concurrent requests under Riva compared to running the NeMo model directly in PyTorch.

### Scalability

| Aspect | Vertex AI | Riva on GKE |
|--------|-----------|-------------|
| Concurrent connections | N/A (stateless HTTP) | Up to 100 WebSocket connections per pod |
| Horizontal scaling | Vertex AI autoscaler (min/max replicas) | GKE node pool scaling + Helm replica count |
| Scale-to-zero | No (minimum 1 replica always running) | Possible with GKE node auto-provisioning |
| Multi-GPU | One GPU per container | One GPU per pod, multiple pods per cluster |

### Cost Comparison

Both approaches use similar GPU infrastructure (T4 or L4), so raw compute cost is comparable:

| Component | Vertex AI | GKE + Riva |
|-----------|-----------|------------|
| GPU (T4) | ~$0.35/hr | ~$0.35/hr |
| Compute (n1-standard-4/8) | ~$0.19/hr | ~$0.38/hr |
| Management overhead | Included in Vertex AI pricing | GKE cluster fee: $0.10/hr |
| **Total** | **~$0.54/hr** | **~$0.83/hr** |

Riva on GKE costs slightly more due to the larger machine type (n1-standard-8 vs n1-standard-4) and GKE management fee. However, the TensorRT optimization means each GPU handles more concurrent requests, so **cost per transcription is often lower** at scale.

### Operational Complexity

| Aspect | Vertex AI | Riva on GKE |
|--------|-----------|-------------|
| Setup time | ~30 min (single notebook) | ~45 min (cluster + Helm + secrets) |
| Infrastructure management | Fully managed | You manage the GKE cluster |
| Model updates | Rebuild and redeploy Docker image | Update Helm chart image tag |
| Monitoring | Vertex AI metrics dashboard | GKE monitoring + custom metrics |
| SSL/TLS | Automatic (Vertex AI endpoints) | Manual (ingress controller or managed certificate) |

Vertex AI is simpler to operate. Riva on GKE requires Kubernetes knowledge and ongoing cluster management. This is the primary trade-off for gaining streaming capabilities.

## When to Use Each Approach

### Use Vertex AI (Flask/Triton) when:

- Transcribing **pre-recorded audio files** (podcasts, meetings, voicemails)
- Audio is available as a **complete clip** before transcription starts
- **Simplicity** is more important than latency
- You need **minimal infrastructure management**
- Request volume is low to moderate (< 10 concurrent requests)

### Use Riva on GKE when:

- Building a **voice assistant** or **conversational AI** system
- Transcribing **live audio** from a microphone or phone call
- Users expect to see **words appearing as they speak**
- You need **end-of-utterance detection** to trigger downstream actions in realtime
- You need **sub-second latency** from speech to text
- You're serving **many concurrent audio streams** (call centers, live captioning)
- You want **TensorRT-optimized inference** for maximum GPU efficiency

## Architecture Reference

### Vertex AI Approach (batch/offline)

```
Client                          Vertex AI Endpoint
  │                                │
  │  POST /predict                 │
  │  {"instances": [{              │
  │    "audio_base64": "UklGR..." │
  │  }]}                          │
  │ ──────────────────────────────>│
  │                                │  Triton / Flask
  │                                │  ├── Decode base64
  │                                │  ├── Write temp WAV
  │                                │  ├── model.transcribe()
  │                                │  └── Return text
  │  {"predictions": [{            │
  │    "transcription": "Hello..." │
  │  }]}                          │
  │ <──────────────────────────────│
  │                                │
  └── Connection closed ───────────┘
```

### Riva on GKE (realtime streaming)

```
Client                          Riva on GKE
  │                                │
  │  Open gRPC stream              │
  │ ──────────────────────────────>│
  │                                │
  │  Audio chunk 1 (250ms)         │
  │ ──────────────────────────────>│
  │                                │  Process with cached state
  │          Interim: "hello"      │
  │ <──────────────────────────────│
  │                                │
  │  Audio chunk 2 (250ms)         │
  │ ──────────────────────────────>│
  │                                │  Update cached state
  │      Interim: "hello how"      │
  │ <──────────────────────────────│
  │                                │
  │  Audio chunk 3 (250ms)         │
  │ ──────────────────────────────>│
  │                                │  Detect silence → EOU
  │  Final: "Hello, how are you?"  │
  │ <──────────────────────────────│
  │                                │
  │  Audio chunk 4 (250ms)         │
  │ ──────────────────────────────>│
  │       ... continues ...        │
  │                                │
  └── Stream stays open ───────────┘
```

## Summary

The Vertex AI approach treats transcription as a function call: audio in, text out. It is simple, managed, and sufficient for offline workloads.

The Riva approach treats transcription as a continuous process: audio flows in, text flows out, state is maintained, and the system reacts to speech boundaries in realtime. This is fundamentally different and cannot be replicated with HTTP request-response endpoints.

For any application where a human is speaking and expects immediate feedback — voice assistants, live captioning, call analytics, dictation — Riva on GKE is the correct architecture.
