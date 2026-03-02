"""
Triton Python Backend for NVIDIA Parakeet Realtime EOU 120M.

Handles batched inference requests from Triton Inference Server.
Each request contains a base64-encoded WAV audio clip; the model
returns the transcription and whether an end-of-utterance was detected.
"""

import base64
import logging
import os
import sys
import tempfile

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("parakeet_triton")


class TritonPythonModel:
    """Triton Python Backend model for Parakeet ASR."""

    def initialize(self, args):
        """Load the NeMo ASR model. Called once when Triton loads the model."""
        logger.info("Loading NVIDIA Parakeet Realtime EOU 120M model ...")

        import nemo.collections.asr as nemo_asr

        model_name = "nvidia/parakeet_realtime_eou_120m-v1"
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("No GPU detected — inference will be slow.")

        logger.info("Model ready for inference.")

    def execute(self, requests):
        """Handle a batch of inference requests.

        Each request contains an AUDIO_B64 input (base64-encoded WAV).
        Returns TRANSCRIPTION (string) and EOU_DETECTED (bool) for each.
        """
        responses = []

        # Collect all audio files for batch transcription
        tmp_paths = []
        valid_indices = []
        error_results = {}

        for idx, request in enumerate(requests):
            try:
                audio_b64_tensor = pb_utils.get_input_tensor_by_name(
                    request, "AUDIO_B64"
                )
                audio_b64 = audio_b64_tensor.as_numpy()[0][0]
                if isinstance(audio_b64, bytes):
                    audio_b64 = audio_b64.decode("utf-8")

                if not audio_b64:
                    error_results[idx] = "Missing audio data."
                    continue

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as f:
                    f.write(base64.b64decode(audio_b64))
                    tmp_paths.append(f.name)
                    valid_indices.append(idx)

            except Exception as exc:
                logger.error("Failed to decode request %d: %s", idx, exc)
                error_results[idx] = str(exc)

        # Batch transcription — all valid audio files at once
        transcriptions = {}
        if tmp_paths:
            try:
                outputs = self.model.transcribe(tmp_paths)
                for i, output in enumerate(outputs):
                    text = (
                        output.text
                        if hasattr(output, "text")
                        else str(output)
                    )
                    transcriptions[valid_indices[i]] = text
            except Exception as exc:
                logger.error("Batch transcription failed: %s", exc, exc_info=True)
                for i in valid_indices:
                    error_results[i] = str(exc)
            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        # Build responses for each request
        for idx in range(len(requests)):
            if idx in transcriptions:
                text = transcriptions[idx]
                eou = "<EOU>" in text or "<eou>" in text
            elif idx in error_results:
                text = ""
                eou = False
            else:
                text = ""
                eou = False

            transcription_tensor = pb_utils.Tensor(
                "TRANSCRIPTION",
                np.array([[text]], dtype=object),
            )
            eou_tensor = pb_utils.Tensor(
                "EOU_DETECTED",
                np.array([[eou]], dtype=bool),
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[transcription_tensor, eou_tensor]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up when the model is unloaded."""
        logger.info("Parakeet ASR model unloaded.")