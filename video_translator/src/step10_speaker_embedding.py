"""
Step 10 - Speaker analysis: embedding + gender + age + emotion.

The gender detected here is what step11 (Gemma 4 transcription/
translation) uses for grammatically gender-correct translations, and what
step12 uses to guarantee
male -> male, female -> female, other -> other voice cloning:
XTTS/MMS clone the *actual* reference speaker, and this step is the
sanity-check / metadata record confirming that identity is preserved.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from config import PipelineConfig
from src.utils import get_logger, load_json, save_json

LOG = get_logger(__name__)

EMBEDDING_MODEL_NAME = "pyannote/embedding"
AGE_GENDER_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
EMOTION_MODEL_NAME = "superb/wav2vec2-base-superb-er"

GENDER_LABELS = ["female", "male", "child"]


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


def build_age_gender_model():
    from transformers import Wav2Vec2PreTrainedModel
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        _tied_weights_keys = []

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = torch.mean(outputs[0], dim=1)
            logits_age = self.age(hidden_states)
            logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
            return logits_age, logits_gender

    return AgeGenderModel


def age_to_bucket(age_years: float) -> str:
    if age_years < 13:
        return "child"
    if age_years < 20:
        return "teen"
    if age_years < 30:
        return "20-29"
    if age_years < 40:
        return "30-39"
    if age_years < 50:
        return "40-49"
    if age_years < 60:
        return "50-59"
    return "60+"


def run(cfg: PipelineConfig) -> list:
    import librosa
    from pyannote.audio import Inference, Model
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor

    if not cfg.normalized_speakers_json.exists():
        raise FileNotFoundError(cfg.normalized_speakers_json)

    cfg.dir_speaker_analysis.mkdir(parents=True, exist_ok=True)
    embedding_dir = cfg.dir_speaker_analysis / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"  # these models are small; CPU is fast enough and avoids GPU memory churn

    speakers = load_json(cfg.normalized_speakers_json)

    LOG.info("Loading speaker embedding model (pyannote/embedding)...")
    # pyannote.audio 4.x renamed use_auth_token= to token=.
    embedding_model = Model.from_pretrained(EMBEDDING_MODEL_NAME, token=cfg.hf_token)
    embedding_inference = Inference(embedding_model, window="whole")

    LOG.info("Loading age/gender model...")
    AgeGenderModel = build_age_gender_model()
    age_gender_processor = Wav2Vec2Processor.from_pretrained(AGE_GENDER_MODEL_NAME)
    age_gender_model = AgeGenderModel.from_pretrained(AGE_GENDER_MODEL_NAME, low_cpu_mem_usage=False)
    age_gender_model.to(device).eval()

    LOG.info("Loading emotion model...")
    emotion_extractor = AutoFeatureExtractor.from_pretrained(EMOTION_MODEL_NAME)
    emotion_model = AutoModelForAudioClassification.from_pretrained(EMOTION_MODEL_NAME)
    emotion_model.to(device).eval()

    results = []
    start = time.time()

    for index, speaker in enumerate(speakers, start=1):
        speaker_id = speaker["speaker_id"]
        audio_path = speaker["audio_file"]
        LOG.info(f"[{index}/{len(speakers)}] {speaker_id}")

        try:
            wav, _ = librosa.load(str(audio_path), sr=cfg.sample_rate, mono=True)

            # Preload the waveform ourselves instead of passing a file path:
            # pyannote.audio 4.x decodes paths via torchcodec, whose prebuilt
            # shared libs try to dlopen a CUDA runtime lib even on CPU-only
            # installs and fail if it's missing. A {'waveform', 'sample_rate'}
            # dict is an officially supported Inference() input that skips
            # torchcodec entirely.
            waveform_dict = {
                "waveform": torch.from_numpy(wav).unsqueeze(0).float(),
                "sample_rate": cfg.sample_rate,
            }
            embedding = np.asarray(embedding_inference(waveform_dict), dtype=np.float32)
            embedding_file = embedding_dir / f"{speaker_id}.npy"
            np.save(embedding_file, embedding)

            inputs = age_gender_processor(wav, sampling_rate=cfg.sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits_age, logits_gender = age_gender_model(inputs.input_values.to(device))

            age_years = float(logits_age.squeeze().item()) * 100.0
            gender_probs = logits_gender.squeeze().cpu().numpy()
            gender_idx = int(np.argmax(gender_probs))

            emo_inputs = emotion_extractor(wav, sampling_rate=cfg.sample_rate, return_tensors="pt")
            with torch.no_grad():
                emo_logits = emotion_model(**{k: v.to(device) for k, v in emo_inputs.items()}).logits
            emo_probs = torch.softmax(emo_logits, dim=-1).squeeze().cpu().numpy()
            emo_labels = [emotion_model.config.id2label[i] for i in range(len(emo_probs))]
            top_emo = int(np.argmax(emo_probs))

            results.append({
                "speaker_id": speaker_id,
                "audio_file": str(audio_path),
                "embedding_file": str(embedding_file),
                "embedding_dimension": int(embedding.shape[-1]),
                "gender": GENDER_LABELS[gender_idx],
                "gender_confidence": round(float(gender_probs[gender_idx]), 4),
                "emotion": emo_labels[top_emo],
                "emotion_scores": {l: round(float(p), 4) for l, p in zip(emo_labels, emo_probs)},
                "age_range": age_to_bucket(age_years),
                "age_estimated_years": round(age_years, 1),
            })

        except Exception as e:
            LOG.error(f"Failed for {speaker_id}: {e}")

    save_json(cfg.speaker_analysis_json, results)
    LOG.info(f"Speaker analysis done in {time.time() - start:.2f}s for {len(results)} speakers")
    return results


if __name__ == "__main__":
    run(PipelineConfig())
