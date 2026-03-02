FROM nvcr.io/nvidia/pytorch:24.01-py3

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg sox && \
    rm -rf /var/lib/apt/lists/*

# Python deps — NeMo ASR toolkit + serving stack
# NeMo pins its own PyTorch version, so install it first.
RUN pip install --no-cache-dir \
    Cython packaging \
    flask gunicorn soundfile \
    "nemo_toolkit[asr]"

# Install torchvision and torchaudio AFTER NeMo so pip resolves versions
# that match the PyTorch version NeMo installed.
# Pin setuptools<78 because 78+ removed pkg_resources, which NeMo needs.
RUN pip install --no-cache-dir torchvision torchaudio "setuptools<78"

# Pre-download the model during build so startup is fast
RUN python -c "\
import nemo.collections.asr as nemo_asr; \
m = nemo_asr.models.ASRModel.from_pretrained( \
    'nvidia/parakeet_realtime_eou_120m-v1'); \
del m; print('Model cached.')"

# Copy serving handler
COPY handler.py /app/handler.py
WORKDIR /app

# Vertex AI injects these env vars at runtime
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

EXPOSE 8080

CMD exec gunicorn \
    --bind 0.0.0.0:${AIP_HTTP_PORT} \
    --timeout 300 \
    --workers 1 \
    --threads 2 \
    handler:app
