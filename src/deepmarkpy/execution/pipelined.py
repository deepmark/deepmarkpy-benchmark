"""The pipelined executor: the sequential run loop, with independent work overlapped.

``Benchmark.run(executor="pipelined")`` returns what the sequential loop
returns, entry for entry and in the same order.

- **Ordered lane** (one thread): loading, watermark draws, embedding, every
  in-process attack and detection by an in-process model, in the sequential
  order. Host RNG draws and process-global state (libcodec2's PRNG) therefore
  follow the sequential run.
- **Service lanes**: attacks whose plugin ships a Dockerfile. They share one
  lane unless ``Parallelism.service_lanes`` gives a group its own; with an
  in-process model they stay on the ordered lane, whose detection may depend
  on what it embedded last.
- **Detection** by a dockerized model, ``detect_concurrency`` at once.
- **Metrics** in ``workers`` processes with this process's BLAS thread counts,
  which MCD and SI-SDR depend on.

At most ``files_in_flight`` files are under way. ``on_file_complete`` sees
them in order, and an error is raised from the file and attack the
sequential loop would raise it from.

A run stops if an attack changes its input in place, or if anything off the
ordered lane draws from ``random``/``np.random``: their module functions
refuse other threads while the run lasts, and the generators' state is
checked between ordered jobs and at the end.
"""

import collections
import inspect
import logging
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from deepmarkpy.benchmark import (
    _NEEDS_CLEAN_ORIGINAL,
    _NO_CROSS_MODEL,
    _TUPLE_RETURNING_ATTACKS,
)
from deepmarkpy.utils import efficiency
from deepmarkpy.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

ORDERED_LANE = "ordered"
DETECT_LANE = "detect"
METRICS_LANE = "metrics"
# Every dockerized attack that service_lanes does not name.
SHARED_SERVICE_LANE = "services"


def runs_in_service(plugin_cls):
    """Whether a plugin does its work in a container: its directory ships a Dockerfile."""
    try:
        path = inspect.getfile(plugin_cls)
    except TypeError:
        return False
    return os.path.isfile(os.path.join(os.path.dirname(path), "Dockerfile"))


def _blas_limits():
    """This process's BLAS/OpenMP thread counts, by threadpoolctl API, or None."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    limits = {}
    for pool in threadpool_info():
        limits.setdefault(pool["user_api"], pool["num_threads"])
    return limits


# A metrics worker's BLAS limits, held for the life of the worker.
_worker_blas_limits = None


def _init_worker(blas_limits):
    """Give a metrics worker the BLAS thread counts of the process that started it."""
    global _worker_blas_limits
    if blas_limits:
        from threadpoolctl import threadpool_limits

        _worker_blas_limits = threadpool_limits(limits=blas_limits)


def _timed_metrics(reference, degraded, sr, metrics):
    """``compute_metrics`` in a worker, with the seconds it took."""
    start = time.perf_counter()
    return compute_metrics(reference, degraded, sr, metrics=metrics), time.perf_counter() - start


# Module-level functions of the host RNGs that read state without drawing.
_RNG_READS = {"get_state", "getstate"}


class _OrderedRngOnly:
    """While installed, a module-level ``random``/``np.random`` call raises off one thread."""

    def __init__(self):
        self.thread = None            # ident of the one thread allowed to draw
        self._saved = []

    def install(self):
        for module, owner in ((np.random, np.random.mtrand._rand), (random, random._inst)):
            for name in dir(module):
                fn = getattr(module, name)
                if getattr(fn, "__self__", None) is owner and name not in _RNG_READS:
                    self._saved.append((module, name, fn))
                    setattr(module, name, self._guarded(fn, f"{module.__name__}.{name}"))

    def uninstall(self):
        for module, name, fn in self._saved:
            setattr(module, name, fn)
        self._saved = []

    def _guarded(self, fn, qualname):
        def guarded(*args, **kwargs):
            if threading.get_ident() != self.thread:
                raise RuntimeError(
                    f"{qualname}() was called on thread "
                    f"'{threading.current_thread().name}', off the pipelined "
                    "executor's ordered lane, so this run's host RNG draws "
                    "would no longer follow the sequential executor's. Run "
                    "with --executor sequential."
                )
            return fn(*args, **kwargs)
        guarded.__wrapped__ = fn
        return guarded


def _rng_state():
    state = np.random.get_state()
    return state[1].copy(), state[2:], random.getstate()


def _same_rng(a, b):
    return np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]


class _File:
    """One file's work in flight."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.base_filename = os.path.splitext(os.path.basename(filepath))[0]
        self.prepared = None          # Future of the ordered lane's _prepare
        self.wm_quality = None        # Future of the watermark-only metrics
        self.tasks = []
        # Set by _prepare.
        self.audio = self.watermarked_audio = self.sampling_rate = None
        self.sr_scalar = self.watermark = self.file_timings = None
        self.attack_kwargs = None
        self.pristine = None


class _Task:
    """One attack entry applied to one file."""

    def __init__(self, entry, ordered):
        self.entry = entry
        self.class_name, self.display_name = entry[0], entry[1]
        self.ordered = ordered
        self.applied = None           # Future of the attack stage
        # Set by the attack stage.
        self.attacked_audio = None
        self.attack_timings = None
        self.cross_model_accuracy = _NO_CROSS_MODEL
        self.detection = None         # (message, confidence, timings) or its Future
        self.quality = None           # Future of the metrics, or None


class PipelinedRun:
    """One ``Benchmark.run`` under the pipelined executor; see the module docstring."""

    def __init__(self, bench, *, filepaths, model_instance, model_config,
                 expanded_attacks, attack_kwargs, watermark_data, sampling_rate,
                 resolver, crop_before_attack, save_audio, output_dir, verbose,
                 parallelism):
        self.bench = bench
        self.filepaths = list(filepaths)
        self.model = model_instance
        self.returns_confidence = model_config.get("returns_confidence", False)
        self.is_zero_bit = model_config.get("is_zero_bit", False)
        self.attack_kwargs = attack_kwargs
        self.watermark_data = watermark_data
        self.sampling_rate = sampling_rate
        self.resolver = resolver
        self.crop_before_attack = crop_before_attack
        self.save_audio = save_audio
        self.output_dir = output_dir
        self.verbose = verbose
        self.parallelism = parallelism
        if parallelism.workers < 1 or parallelism.files_in_flight < 1 \
                or parallelism.detect_concurrency < 1:
            raise ValueError(f"Parallelism limits must be at least 1: {parallelism}")

        self.model_in_service = runs_in_service(type(model_instance))
        self.entries = []
        for entry in expanded_attacks:
            if entry[0] not in bench.attacks:
                logger.warning(f"Attack '{entry[0]}' not found. Skipping.")
                continue
            self.entries.append((entry, self._is_ordered(entry[0])))

        self._busy = collections.defaultdict(float)
        self._jobs = collections.Counter()
        self._busy_lock = threading.Lock()
        self._rng_after_last_job = None

    def _is_ordered(self, class_name):
        """Whether an attack must run on the ordered lane."""
        if class_name in _TUPLE_RETURNING_ATTACKS or class_name in _NEEDS_CLEAN_ORIGINAL:
            return True
        if not self.model_in_service:
            return True
        return not runs_in_service(self.bench.attacks[class_name]["class"])

    # -- running ---------------------------------------------------------

    def _plan_service_lanes(self):
        """``{attack: lane}`` for the attacks on service lanes, and each lane's slots."""
        p = self.parallelism
        service_attacks = sorted({e[0] for e, ordered in self.entries if not ordered})
        lane_of, named = {}, set()
        for group in p.service_lanes:
            if named & set(group):
                raise ValueError(
                    f"service_lanes names {sorted(named & set(group))} in more than one group"
                )
            named |= set(group)
            members = [name for name in group if name in service_attacks]
            for name in members:
                lane_of[name] = "+".join(members)
        in_process = named & {e[0] for e, ordered in self.entries if ordered}
        if in_process:
            logger.warning(
                "service_lanes names attacks that run in this process, on the "
                "ordered lane: %s", sorted(in_process),
            )
        for name in service_attacks:
            lane_of.setdefault(name, SHARED_SERVICE_LANE)

        slots = {lane: 1 for lane in lane_of.values()}
        for name, count in p.service_concurrency.items():
            if lane_of.get(name, SHARED_SERVICE_LANE) == name:
                slots[name] = max(1, int(count))
            else:
                logger.warning(
                    "service_concurrency for %s does not apply: it has no lane "
                    "to itself (name it alone in service_lanes)", name,
                )
        return lane_of, slots

    def run(self, on_file_complete=None):
        p = self.parallelism
        self._lane_of, self._service_slots = self._plan_service_lanes()
        self._ordered = ThreadPoolExecutor(1, thread_name_prefix="ordered")
        self._services = {
            lane: ThreadPoolExecutor(slots, thread_name_prefix=f"lane-{lane}")
            for lane, slots in self._service_slots.items()
        }
        self._detect = ThreadPoolExecutor(p.detect_concurrency, thread_name_prefix="detect")
        self._metrics = ProcessPoolExecutor(
            p.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(_blas_limits(),),
        )
        logger.info(
            "Pipelined executor: %d file(s); %d attack(s) on the ordered lane, "
            "%d on service lanes (%s); %d metrics worker(s), %d file(s) in flight.",
            len(self.filepaths),
            sum(1 for _, ordered in self.entries if ordered),
            sum(1 for _, ordered in self.entries if not ordered),
            "; ".join(f"{lane} x{slots}" for lane, slots in self._service_slots.items()) or "none",
            p.workers, p.files_in_flight,
        )

        self._rng_after_last_job = _rng_state()
        self._rng_guard = _OrderedRngOnly()
        self._rng_guard.install()
        results = {}
        in_flight = collections.deque()
        start = time.perf_counter()
        try:
            for filepath in self.filepaths:
                while len(in_flight) >= p.files_in_flight:
                    self._finish(in_flight.popleft(), results, on_file_complete)
                in_flight.append(self._start(filepath, in_flight, results, on_file_complete))
            while in_flight:
                self._finish(in_flight.popleft(), results, on_file_complete)
            # Every lane is idle now, so a draw after the ordered lane's last
            # job shows here.
            self._check_rng()
        except BaseException:
            self._shutdown(wait=False)
            raise
        finally:
            self._rng_guard.uninstall()
        self._shutdown(wait=True)
        self._log_lanes(time.perf_counter() - start, len(results))
        return results

    def _shutdown(self, wait):
        for executor in (self._ordered, *self._services.values(), self._detect, self._metrics):
            executor.shutdown(wait=wait, cancel_futures=True)

    def _start(self, filepath, in_flight, results, on_file_complete):
        """Queue one file's work; returns once the attacks that need only its watermarked audio are queued."""
        job = _File(filepath)
        job.prepared = self._ordered.submit(self._lane, ORDERED_LANE, self._prepare, job)
        for entry, ordered in self.entries:
            task = _Task(entry, ordered)
            job.tasks.append(task)
            if ordered:
                task.applied = self._ordered.submit(
                    self._lane, ORDERED_LANE, self._ordered_attack, job, task,
                )
        try:
            job.prepared.result()
        except Exception:
            # The files before this one finish first, as they would have
            # sequentially, and an error among them is the one raised.
            while in_flight:
                self._finish(in_flight.popleft(), results, on_file_complete)
            raise
        for task in job.tasks:
            if not task.ordered:
                lane = self._lane_of[task.class_name]
                task.applied = self._services[lane].submit(
                    self._lane, lane, self._service_attack, job, task,
                )
        return job

    def _finish(self, job, results, on_file_complete):
        """Wait for one file, assemble its results as the sequential loop does, and hand them on."""
        job.prepared.result()
        file_results = {"attacks": {}}
        file_results["watermarked_audio_quality"] = self._quality(job.wm_quality)
        for task in job.tasks:
            try:
                task.applied.result()
                detection = task.detection
                if isinstance(detection, Future):
                    detection = detection.result()
                detected_message, confidence, detect_timings = detection
                quality = self._quality(task.quality)
                entry = self.bench._attack_entry(
                    self.is_zero_bit, job.watermark, job.watermarked_audio,
                    task.attacked_audio, detected_message, confidence, quality,
                    {**task.attack_timings, **detect_timings},
                    job.file_timings, task.cross_model_accuracy,
                )
            except Exception:
                # Name the attack and file; keep the exception's type, which
                # callers use to tell infrastructure failures from bugs.
                logger.error("Attack %s failed on %s", task.display_name, job.filepath)
                raise
            file_results["attacks"][task.display_name] = entry
        self._check_inputs(job, "an attack on this file")
        results[job.filepath] = file_results
        if on_file_complete is not None:
            on_file_complete(job.filepath, file_results)

    def _quality(self, future):
        if future is None:
            return None
        quality, seconds = future.result()
        with self._busy_lock:
            self._busy[METRICS_LANE] += seconds
            self._jobs[METRICS_LANE] += 1
        return quality

    # -- lane jobs --------------------------------------------------------

    def _lane(self, lane, fn, *args):
        """Run one job on ``lane``, counting its busy time."""
        start = time.perf_counter()
        try:
            if lane == ORDERED_LANE:
                self._rng_guard.thread = threading.get_ident()
                self._check_rng()
                try:
                    return fn(*args)
                finally:
                    self._rng_after_last_job = _rng_state()
            return fn(*args)
        finally:
            with self._busy_lock:
                self._busy[lane] += time.perf_counter() - start
                self._jobs[lane] += 1

    def _prepare(self, job):
        if self.verbose:
            logger.info(f"\nProcessing file: {job.filepath}")
        (job.audio, job.watermarked_audio, job.sampling_rate, job.watermark,
         job.file_timings) = self.bench._prepare_file(
            job.filepath, self.model, self.watermark_data, self.sampling_rate,
            self.crop_before_attack, self.resolver, self.save_audio,
            self.output_dir,
        )
        job.pristine = (np.array(job.audio, copy=True),
                        np.array(job.watermarked_audio, copy=True))
        job.attack_kwargs = {
            **self.attack_kwargs,
            "watermark_data": job.watermark,
            "orig_audio": job.audio,
        }
        sr = job.sampling_rate
        job.sr_scalar = int(sr) if isinstance(sr, (np.ndarray, list)) else sr
        job.wm_quality = self._metrics.submit(
            _timed_metrics, job.audio, job.watermarked_audio, job.sr_scalar,
            set(self.resolver.all_signal_metrics()),
        )

    def _apply(self, job, task):
        attacked_audio, extra, task.attack_timings, cross_model = self.bench._apply_entry(
            task.entry, job.attack_kwargs, job.watermarked_audio, job.audio,
            job.sampling_rate, self.resolver, job.base_filename,
            self.save_audio, self.output_dir, self.verbose,
        )
        task.attacked_audio = attacked_audio
        return extra, cross_model

    def _ordered_attack(self, job, task):
        extra, cross_model = self._apply(job, task)
        if not self.model_in_service:
            task.detection = self._run_detect(task.attacked_audio, job.sampling_rate)
        if cross_model is not None:
            name, instance = cross_model
            task.cross_model_accuracy = self.bench._cross_model_accuracy(
                name, extra, instance.detect(task.attacked_audio, job.sampling_rate),
            )
        self._check_inputs(job, task.display_name)
        self._after_attack(job, task)

    def _service_attack(self, job, task):
        self._apply(job, task)
        self._after_attack(job, task)

    def _after_attack(self, job, task):
        """Queue the attacked audio's detection and metrics."""
        if task.detection is None:
            task.detection = self._detect.submit(
                self._lane, DETECT_LANE, self._run_detect,
                task.attacked_audio, job.sampling_rate,
            )
        relevant = set(self.resolver.metrics_for_attack(task.display_name))
        if relevant:
            task.quality = self._metrics.submit(
                _timed_metrics, job.audio, task.attacked_audio, job.sr_scalar, relevant,
            )

    def _run_detect(self, attacked_audio, sampling_rate):
        timings = {}
        confidence = None
        with efficiency.measure(timings, "detect_latency",
                                self.resolver.is_enabled(None, "detect_latency")):
            if self.returns_confidence:
                detected_message, confidence = self.model.detect(attacked_audio, sampling_rate)
            else:
                detected_message = self.model.detect(attacked_audio, sampling_rate)
        return detected_message, confidence, timings

    # -- guards ----------------------------------------------------------

    def _check_rng(self):
        if not _same_rng(self._rng_after_last_job, _rng_state()):
            raise RuntimeError(
                "The host RNG (random or np.random) was drawn from outside the "
                "pipelined executor's ordered lane, so this run's random draws "
                "no longer follow the sequential executor's. An attack on a "
                "service lane draws from it; run with --executor sequential."
            )

    def _check_inputs(self, job, culprit):
        audio, watermarked_audio = job.pristine
        if not (np.array_equal(job.audio, audio)
                and np.array_equal(job.watermarked_audio, watermarked_audio)):
            raise RuntimeError(
                f"{culprit} changed its input audio in place on "
                f"{job.filepath}. Every attack after it would see different "
                "audio than under the sequential executor; run with "
                "--executor sequential."
            )

    # -- reporting -------------------------------------------------------

    def _log_lanes(self, wall, n_files):
        per_file = wall / n_files if n_files else 0.0
        lines = [f"Pipelined executor: {n_files} file(s) in {wall:.1f} s ({per_file:.1f} s per file). Busy time per lane:"]
        capacity = {ORDERED_LANE: 1, DETECT_LANE: self.parallelism.detect_concurrency,
                    METRICS_LANE: self.parallelism.workers}
        capacity.update(self._service_slots)
        for lane, busy in sorted(self._busy.items(), key=lambda item: -item[1]):
            share = busy / (wall * capacity.get(lane, 1)) if wall else 0.0
            lines.append(
                f"  {lane:<32} {busy:8.1f} s over {self._jobs[lane]:5d} job(s), "
                f"{share:6.1%} of {capacity.get(lane, 1)} slot(s)"
            )
        logger.info("\n".join(lines))
