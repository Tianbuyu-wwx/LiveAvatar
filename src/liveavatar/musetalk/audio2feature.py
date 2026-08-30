import os
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WhisperModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class Audio2Feature:
    def __init__(self, whisper_model_type="tiny", model_path="./models/whisper"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.whisper = WhisperModel.from_pretrained(model_path)
        self.whisper = self.whisper.to(device=device, dtype=weight_dtype).eval()
        self.whisper.requires_grad_(False)

    def get_sliced_feature(
        self,
        feature_array,
        vid_idx,
        audio_feat_length=[2, 2],
        fps=25,
    ):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        # Symmetric context: look ``audio_feat_length[0]`` video frames back and
        # ``audio_feat_length[1]`` video frames forward.  One video frame maps to
        # 2 Whisper frames (50 Hz / 25 fps), hence the ``* 2`` factor.
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            clipped = max(0, min(length - 1, idx))
            x = feature_array[clipped]
            selected_feature.append(x)
            selected_idx.append(clipped)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)  # 50*384
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(
        self,
        feature_array,
        vid_idx,
        audio_feat_length=[2, 2],
        fps=25,
    ):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-audio_feat_length[0], audio_feat_length[1] + 1):
            left_idx = int((vid_idx + dt) * 50 / fps)
            if left_idx < 1 or left_idx > length - 1:
                print('test-----,left_idx=', left_idx)
                left_idx = max(0, left_idx)
                left_idx = min(length - 1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx - 1:left_idx + 1]
                selected_feature.append(x)
                selected_idx.append(left_idx - 1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)  # 50*384
        return selected_feature, selected_idx

    def feature2chunks(
        self,
        feature_array,
        fps,
        batch_size,
        audio_feat_length=[2, 2],
        start=0,
    ):
        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps
        i = 0
        for _ in range(batch_size):
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i + start,
                audio_feat_length=audio_feat_length,
                fps=fps,
            )
            whisper_chunks.append(selected_feature)
            i += 1
        return whisper_chunks

    def feature2chunks_with_idx(
        self,
        feature_array,
        fps,
        batch_size,
        audio_feat_length=[2, 2],
        start=0,
    ):
        """Like feature2chunks but also return per-frame audio feature indices.

        Returns
        -------
        chunks : list[np.ndarray]
            Audio feature slices for each video frame.
        meta : list[dict]
            Each dict has ``center``, ``left``, ``right`` and ``selected`` keys
            (indices into the 50 Hz Whisper feature array).
        """
        chunks = []
        meta = []
        i = 0
        for _ in range(batch_size):
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i + start,
                audio_feat_length=audio_feat_length,
                fps=fps,
            )
            chunks.append(selected_feature)
            meta.append(
                {
                    "center": int((i + start) * 50 / fps),
                    "left": selected_idx[0],
                    "right": selected_idx[-1],
                    "selected": selected_idx,
                }
            )
            i += 1
        return chunks, meta

    def audio2feat(self, wav_data):
        input_feature = self.feature_extractor(
            wav_data,
            return_tensors="pt",
            sampling_rate=16000,
        ).input_features
        input_feature = input_feature.to(device).to(weight_dtype)
        whisper_feature = self.whisper.encoder(
            input_feature, output_hidden_states=True
        ).hidden_states
        whisper_feature = torch.stack(whisper_feature, dim=2)
        return whisper_feature.squeeze(0).cpu().numpy()
