FROM nvcr.io/nvidia/tritonserver:24.01-py3

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg sox && \
    rm -rf /var/lib/apt/lists/*

# The tritonserver base image has a distutils-installed blinker that pip
# cannot uninstall. Remove it manually so Flask can install its version.
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
           /usr/lib/python3/dist-packages/blinker-*.egg-info

# Python deps — NeMo ASR toolkit + proxy server
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
RUN python3 -c "\
import nemo.collections.asr as nemo_asr; \
m = nemo_asr.models.ASRModel.from_pretrained( \
    'nvidia/parakeet_realtime_eou_120m-v1'); \
del m; print('Model cached.')"

# Copy Triton model repository
COPY models/ /models/

# Copy Vertex AI compatibility proxy
COPY server.py /app/server.py
WORKDIR /app

# Vertex AI environment variables
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV TRITON_URL=http://localhost:8000

EXPOSE 8080

# Clear the base image's ENTRYPOINT so our shell-form CMD runs correctly.
ENTRYPOINT []

# Start Triton in the background, then the compatibility proxy
CMD tritonserver --model-repository=/models --http-port=8000 --log-verbose=0 & \
    TRITON_PID=$! && \
    echo "Waiting for Triton to start ..." && \
    for i in $(seq 1 120); do \
        if curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1; then \
            echo "Triton is ready."; break; \
        fi; \
        sleep 2; \
    done && \
    exec gunicorn \
        --bind 0.0.0.0:${AIP_HTTP_PORT} \
        --timeout 300 \
        --workers 1 \
        --threads 4 \
        server:app