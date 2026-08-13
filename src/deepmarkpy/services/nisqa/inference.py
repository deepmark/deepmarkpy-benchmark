"""NISQA inference engine — runs inside the container."""

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np
import soundfile as sf


_NISQA_MAX_SEC = 12.0
_NISQA_CHUNK_SEC = 10.0
_NISQA_MIN_CHUNK_SEC = 3.0

NISQA_DIMENSIONS = ["mos", "noi", "dis", "col", "loud"]


class NISQAEngine:
    def __init__(self, weights_path: str = "/app/weights/nisqa.tar"):
        from nisqa.NISQA_model import nisqaModel

        with contextlib.redirect_stdout(io.StringIO()):
            self.model = nisqaModel({
                "mode": "predict_file",
                "pretrained_model": weights_path,
                "deg": __file__,
                "tr_bs_val": 1,
                "tr_num_workers": 0,
                "output_dir": None,
                "ms_channel": None,
            })

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        """Score a signal, chunking if longer than the model's window."""
        duration = len(audio) / sr
        if duration <= _NISQA_MAX_SEC:
            return self._predict_chunk(audio, sr)

        chunk_len = int(_NISQA_CHUNK_SEC * sr)
        min_len = int(_NISQA_MIN_CHUNK_SEC * sr)
        accum = {k: [] for k in NISQA_DIMENSIONS}

        for start in range(0, len(audio), chunk_len):
            chunk = audio[start:start + chunk_len]
            if len(chunk) < min_len:
                continue
            res = self._predict_chunk(chunk, sr)
            if res is None:
                continue
            for k in NISQA_DIMENSIONS:
                accum[k].append(res[k])

        if not accum["mos"]:
            return None

        return {k: float(np.mean(accum[k])) for k in NISQA_DIMENSIONS}

    def _predict_chunk(self, audio: np.ndarray, sr: int):
        """Run NISQA on a single segment."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)
            self.model.args["deg"] = tmp_path

            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Empty filters detected in mel frequency basis",
                    category=UserWarning,
                )
                self.model._loadDatasets()
                df = self.model.predict()

            row = df.iloc[0]
            return {
                "mos": float(row["mos_pred"]),
                "noi": float(row["noi_pred"]),
                "dis": float(row["dis_pred"]),
                "col": float(row["col_pred"]),
                "loud": float(row["loud_pred"]),
            }
        except (RuntimeError, ValueError, KeyError):
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
