"""The parity tooling must run what it claims to run, on what it claims to run it.

No containers and no GPU: these tests cover the parts that silently decide a
verdict's meaning.
- The flags each container gets: a GPU run on our slice with TF32 off; a CPU
  run that cannot see a GPU.
- The corpus and the per-case seeds and watermarks, which must not depend on
  order or on how many times they are built.
- Which image a verdict vouches for.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deepmarkpy.parity import cli
from deepmarkpy.parity.corpus import EDGE_LENGTHS, build_corpus, load_manifest, verify_inputs
from deepmarkpy.parity.runner import case_bits, case_seed
from deepmarkpy.parity.verdicts import passing_verdict, verdict_path, write_verdict

UUID = "MIG-50bf93fa-b44c-530a-acf3-c79763c56eec"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPMARK_PARITY_ROOT", str(tmp_path / "parity"))
    return tmp_path / "parity"


@pytest.fixture
def sources(tmp_path):
    d = tmp_path / "wavs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name in ("b_clip.wav", "a_clip.wav"):
        sf.write(d / name, 0.1 * rng.standard_normal(40000), 16000)
    return d


class TestContainerFlags:
    def _flags(self, cmd):
        return " ".join(cmd)

    def test_a_gpu_run_gets_exactly_our_slice(self, root):
        cmd = cli._container("img:gpu", ["--op", "apply"], gpu_uuid=UUID,
                             env={"NVIDIA_TF32_OVERRIDE": "0"})
        flat = self._flags(cmd)
        assert f"--device nvidia.com/gpu={UUID}" in flat
        assert "NVIDIA_TF32_OVERRIDE=0" in flat
        assert "gpu=all" not in flat and "CUDA_VISIBLE_DEVICES=" not in flat

    def test_a_cpu_run_cannot_see_a_gpu(self, root):
        flat = self._flags(cli._container("img:cpu", ["--op", "apply"]))
        assert "CUDA_VISIBLE_DEVICES=" in flat and "--device" not in flat

    def test_threads_are_pinned_and_recorded_in_the_command(self, root):
        flat = self._flags(cli._container("img:cpu", [], cpu_threads=8))
        assert "--cpus 8" in flat
        for var in ("OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8", "OPENBLAS_NUM_THREADS=8"):
            assert var in flat

    def test_the_corpus_is_mounted_read_only(self, root):
        flat = self._flags(cli._container("img:cpu", []))
        assert f"{root}:/corpus:ro" in flat
        assert "/parity/runner.py:ro" in flat


class TestCorpus:
    def test_cases_cover_speech_and_every_edge(self, root, sources):
        manifest = build_corpus(sources, root)
        names = set(manifest["cases"])
        assert {"a_clip", "b_clip"} <= names
        assert {f"len_{n}" for n in EDGE_LENGTHS} <= names
        assert {"silence", "quiet", "clipped", "noise", "sweep"} <= names
        assert manifest["cases"]["len_63999"]["samples"] == 63999

    def test_building_twice_gives_the_same_hash(self, root, sources):
        first = build_corpus(sources, root)["corpus_sha256"]
        assert build_corpus(sources, root)["corpus_sha256"] == first

    def test_edge_cases_come_from_the_first_clip_in_sorted_order(self, root, sources):
        manifest = build_corpus(sources, root)
        assert manifest["cases"]["len_16000"]["source"] == "a_clip.wav"

    def test_a_tampered_input_is_refused(self, root, sources):
        build_corpus(sources, root)
        clean = root / "16000" / "clean"
        x = np.load(clean / "noise.npy")
        x[0] += 1e-12
        np.save(clean / "noise.npy", x)
        with pytest.raises(ValueError, match="noise"):
            verify_inputs(clean, load_manifest(root)["cases"])


class TestPerCaseRandomness:
    def test_a_case_seed_depends_only_on_its_name(self):
        assert case_seed(1, "clip") == case_seed(1, "clip")
        assert case_seed(1, "clip") != case_seed(2, "clip")
        assert case_seed(1, "clip") != case_seed(1, "clip2")

    def test_watermark_bits_are_fixed_per_case_and_size(self):
        a, b = case_bits("clip", 16), case_bits("clip", 16)
        assert np.array_equal(a, b) and a.dtype == np.int32 and set(np.unique(a)) <= {0, 1}
        assert case_bits("clip", 40).size == 40


class TestVerdicts:
    def _record(self, image_id, passed=True):
        return {"service": "encodec", "op": "apply", "tier": "QUANTIZED", "passed": passed,
                "candidate": {"image_id": image_id}, "reference": {"image_id": "sha256:cpu"},
                "corpus": {"sha256": "x"}}

    def test_a_pass_vouches_for_exactly_its_image(self, tmp_path):
        write_verdict(self._record("sha256:aaaaaaaaaaaa1111"), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:aaaaaaaaaaaa1111", tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:bbbbbbbbbbbb1111", tmp_path) is None

    def test_a_prefix_collision_is_not_a_pass(self, tmp_path):
        write_verdict(self._record("sha256:aaaaaaaaaaaa1111"), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:aaaaaaaaaaaa2222", tmp_path) is None

    def test_a_fail_is_not_a_pass(self, tmp_path):
        write_verdict(self._record("sha256:cccccccccccc", passed=False), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:cccccccccccc", tmp_path) is None

    def test_an_incomplete_record_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="corpus"):
            write_verdict({k: v for k, v in self._record("sha256:d").items() if k != "corpus"}, tmp_path)

    def test_the_file_name_carries_service_op_and_image(self, tmp_path):
        p = verdict_path("vae", "apply", "sha256:0123456789abcdef", tmp_path)
        assert p.name == "vae.apply.0123456789ab.json"


class TestGates:
    def test_every_gate_names_a_known_tier_and_its_evidence(self):
        from deepmarkpy.parity.tiers import TIERS

        gates = cli.load_gates()
        for service, gate in gates.items():
            if service.startswith("_"):
                continue
            for op, spec in gate["ops"].items():
                assert spec["tier"] in set(TIERS) | {"STOCHASTIC"}, (service, op)
                assert spec["inputs"], (service, op)
                if "bits" in spec["evidence"] and op != "detect":
                    assert spec.get("detector") in gates, (service, op)

    def test_gate_params_are_what_the_benchmark_client_sends(self):
        # The parity run must exercise the request the benchmark actually makes.
        gates = cli.load_gates()
        assert gates["descript_audio_codec"]["ops"]["apply"]["params"] == {"n_codebooks_dac": 9}
        assert gates["diffusion"]["ops"]["apply"]["params"] == {"diffusion_steps": 5}
        assert gates["speech_enhancement1"]["ops"]["apply"]["params"] == {"noise_strength": 0.01}


def test_baked_weights_are_not_hidden_by_the_cache_mount(root):
    flat = " ".join(cli._container("img:cpu", [], mount_cache=False))
    assert "/root/.cache" not in flat
    assert cli.load_gates()["speech_tokenization"]["mount_cache"] is False
