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
from deepmarkpy.parity.verdicts import engine_config, passing_verdict, verdict_path, write_verdict

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
    F32 = {"engine": "plugin_inference.NeuralVocoderEngine", "numerics": {"strict": True}}
    F64 = {"engine": "engine_override.NeuralVocoderEngineF64", "numerics": {"strict": True}}

    def _record(self, image_id, passed=True, env=None):
        return {"service": "encodec", "op": "apply", "tier": "QUANTIZED", "passed": passed,
                "candidate": {"image_id": image_id, "env": env or self.F32},
                "reference": {"image_id": "sha256:cpu"}, "corpus": {"sha256": "x"}}

    def test_a_pass_vouches_for_exactly_its_image(self, tmp_path):
        write_verdict(self._record("sha256:aaaaaaaaaaaa1111"), tmp_path)
        cfg = engine_config(self.F32)
        assert passing_verdict("encodec", "apply", "sha256:aaaaaaaaaaaa1111", cfg, tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:bbbbbbbbbbbb1111", cfg, tmp_path) is None

    def test_a_prefix_collision_is_not_a_pass(self, tmp_path):
        write_verdict(self._record("sha256:aaaaaaaaaaaa1111"), tmp_path)
        cfg = engine_config(self.F32)
        assert passing_verdict("encodec", "apply", "sha256:aaaaaaaaaaaa2222", cfg, tmp_path) is None

    def test_a_pass_for_one_engine_does_not_vouch_for_another_in_the_same_image(self, tmp_path):
        write_verdict(self._record("sha256:cccccccccccc", passed=False, env=self.F32), tmp_path)
        write_verdict(self._record("sha256:cccccccccccc", passed=True, env=self.F64), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:cccccccccccc", engine_config(self.F64), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:cccccccccccc", engine_config(self.F32), tmp_path) is None

    def test_a_fail_is_not_a_pass(self, tmp_path):
        write_verdict(self._record("sha256:dddddddddddd", passed=False), tmp_path)
        assert passing_verdict("encodec", "apply", "sha256:dddddddddddd", engine_config(self.F32), tmp_path) is None

    def test_an_incomplete_record_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="corpus"):
            write_verdict({k: v for k, v in self._record("sha256:e").items() if k != "corpus"}, tmp_path)

    def test_the_file_name_carries_service_op_image_and_engine(self, tmp_path):
        p = verdict_path("vae", "apply", "sha256:0123456789abcdef", "a1b2c3d4", tmp_path)
        assert p.name == "vae.apply.0123456789ab.a1b2c3d4.json"


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


class TestTieEvidence:
    """The codec probe's evidence is what lets a QUANTIZED flip pass; it must be read strictly."""

    def _probe(self, root, margin, reproduce=True):
        stem, sr = "clip", 16000
        ref = 0.1 * np.random.default_rng(0).standard_normal(16000)
        inject = ref.copy()
        inject[3200:3300] += 0.01                      # what the GPU's codes do on the CPU: frame 10
        cand = inject + 1e-7                           # the GPU output: same codes, tiny continuous difference
        for step, codes in (("cpu_record", [[1, 2, 3, 4], [5, 6, 7, 8]]),
                            ("gpu_record", [[1, 2, 9, 4], [5, 6, 0, 8]]),   # codec frame 2 differs from layer 0
                            ("cpu_inject", [[1, 2, 9, 4], [5, 6, 0, 8]])):
            d = cli._probe_dir("dac", "gpu", step, "wm")
            d.mkdir(parents=True)
            out = {"cpu_record": ref, "gpu_record": cand if reproduce else cand + 1, "cpu_inject": inject}[step]
            np.save(d / f"{stem}.npy", out)
            np.savez(d / f"{stem}.codes.npz", **{f"layer{i}": np.array(c) for i, c in enumerate(codes)})
            np.savez(d / f"{stem}.margins.npz", layer0=np.array([1, 1, margin, 1.0]), layer1=np.ones(4))
        return ref, cand

    def test_a_near_tie_explains_the_frames_its_code_changes(self, root):
        ref, cand = self._probe(root, margin=1.3e-5)
        explained, failures, details = cli._tie_evidence("dac", "gpu", "wm", "clip", ref, cand, 16000, 1e-4)
        assert failures == [] and 10 in explained
        assert details["code_flips"] == [{"codec_frame": 2, "layer": 0, "margin": 1.3e-5}]

    def test_a_decided_choice_that_flipped_is_refused(self, root):
        ref, cand = self._probe(root, margin=5e-3)
        explained, failures, _ = cli._tie_evidence("dac", "gpu", "wm", "clip", ref, cand, 16000, 1e-4)
        assert explained == () and "not a near-tie" in failures[0]

    def test_a_probe_that_does_not_reproduce_the_runs_is_refused(self, root):
        ref, cand = self._probe(root, margin=1.3e-5, reproduce=False)
        explained, failures, _ = cli._tie_evidence("dac", "gpu", "wm", "clip", ref, cand, 16000, 1e-4)
        assert explained is None and "does not reproduce" in failures[0]
