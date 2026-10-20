# Spike S1: CUDA IPC between service containers on MIG 2

Run 2026-10-19 on the development host. The slice was MIG 2, `2g.35gb`, `<MIG UUID>`, vetted idle by `deepmarkpy.utils.gpu_slice` before each run. Containers ran under rootless Podman 4.9.3 with the CDI device for that UUID only.

The spike scripts are development-host tooling, not part of this repository. Raw results are kept on the development host.

## Method

- The producer allocates 4 MiB with the CUDA **driver** API (`cuMemAlloc`) in the primary context and fills it with random bytes. It exports the allocation with `cuIpcGetMemHandle` and writes the 64-byte handle to a shared directory.
- The consumer calls `cuIpcOpenMemHandle`, copies the memory to the host and checks its SHA-256. It then wraps the pointer as a torch tensor via `__cuda_array_interface__`, with zero copies, and checks a reduction against the host bytes.
- Using the driver API through ctypes keeps the handle independent of either side's torch or CUDA-runtime version.

## Result

| Case | Works | open | Bytes | torch zero-copy view |
|---|---|---|---|---|
| Two processes, one container (torch 2.7.1+cu126) | **yes** | 0.25 ms | identical | sum matches |
| Two containers, default namespaces | **no**: `cuIpcOpenMemHandle` returns 201 `CUDA_ERROR_INVALID_CONTEXT` | — | — | — |
| Two containers in one pod, `--share ipc,pid` | **yes** | 0.14 ms | identical | sum matches |
| Same pod: producer 2.7.1+cu126, consumer 2.0.0+cu118 | **yes** | 0.14 ms | identical | sum matches (torch 2.0) |

A 4 MiB copy to the host took ~1 ms (pageable memory).

The torch wrap took ~50–60 ms. That is torch's one-time CUDA initialisation in a process that had not used CUDA yet, not a per-handoff cost.

## Conclusion: GO, under one condition

Legacy CUDA IPC works on a MIG instance across rootless containers and across torch/CUDA versions, **if the containers share IPC and PID namespaces** (one Podman pod). Without that, opening the handle fails.

What this means for the plan (§4.3, T3):
- The GPU services that exchange tensors must run in one pod (or with `--ipc=host --pid=host`).
- A handle is opened once per slot, at ~0.14 ms, and then reused.
- The zero-copy view needs no torch-version-specific API.

Whether T3 is worth building is still decided by measurement (Phase 5/6). Per hop it saves only a host copy of ~0.3–0.6 MB, about 0.1 ms. Its value is in chains that never need the host copy.
