"""Version-scoped serving commands and deployment manifests. Nothing is deployed."""

from __future__ import annotations

import json
import math
import re
import shlex
from datetime import datetime, timezone

import yaml
from pydantic import Field, model_validator

from planner.models import Assumptions, Candidate, Hardware, ModelSpec, Record, Workload
from planner.security import scrub
from planner.sizing import GIB, compatibility, quantization_flag

VLLM_VERSION = "0.10.2"
CLI_SOURCE = "https://docs.vllm.ai/en/v0.10.2/cli/serve.html"
CPU_SOURCE = "https://docs.vllm.ai/en/v0.10.2/getting_started/installation/cpu.html"


class CommandExport(Record):
    status: str
    runnable: bool
    missing: list[str]
    command: str
    argv: list[str]
    environment: dict[str, str]
    notes: list[str]
    sources: list[str]


def vllm_command(model: ModelSpec, h: Hardware, c: Candidate, w: Workload, a: Assumptions) -> CommandExport:
    missing = compatibility(model, h, a, c.tp, w)
    if not model.revision or not re.fullmatch(r"[0-9a-f]{40}", model.revision):
        missing.append("Resolved model commit SHA required")
    if not c.feasible:
        missing.append("Select a feasible configuration")
    if a.kv_dtype != "fp8_e4m3" and a.kv_dtype != a.activation_dtype:
        missing.append(
            "vLLM auto KV dtype follows serving dtype; requested mixed non-FP8 KV precision is not representable"
        )
    if not model.quantization and a.weight_precision == "bf16":
        if a.weight_bytes != (4 if a.activation_dtype == "float32" else 2):
            missing.append("Dense weight bytes must match vLLM serving dtype")
    if not a.chunked_prefill and a.batched_tokens < w.input_max + w.output_max:
        missing.append("Without chunked prefill, batched-token budget must cover max model length")
    argv = [
        "vllm",
        "serve",
        model.model_id,
        "--revision",
        model.revision or "REQUIRED_COMMIT_SHA",
        "--dtype",
        a.activation_dtype,
        "--tensor-parallel-size",
        str(c.tp),
        "--max-model-len",
        str(w.input_max + w.output_max),
        "--max-num-seqs",
        str(c.concurrency),
        "--max-num-batched-tokens",
        str(a.batched_tokens),
        "--block-size",
        str(a.block_size),
        "--kv-cache-dtype",
        "fp8_e4m3" if a.kv_dtype == "fp8_e4m3" else "auto",
        "--enable-prefix-caching" if a.prefix_caching else "--no-enable-prefix-caching",
        "--enable-chunked-prefill" if a.chunked_prefill else "--no-enable-chunked-prefill",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    quant = quantization_flag(model, a)
    if quant:
        argv[argv.index("--dtype") : argv.index("--dtype")] = ["--quantization", quant]
    env = {}
    notes = [
        f"Requires a preinstalled vLLM {VLLM_VERSION} build for the selected hardware.",
        "Command starts one replica; deployment must provision the reported replica count.",
        "HF_TOKEN is inherited from the process environment; it is never included in the command.",
        "Runtime compatibility verified against the linked versioned CLI docs; execution on target hardware remains untested.",
    ]
    if h.kind == "GPU":
        argv += ["--gpu-memory-utilization", str(a.memory_utilization), "--swap-space", "0"]
    else:
        env = {
            "VLLM_CPU_KVCACHE_SPACE": str(math.ceil(c.memory.get("kv", 0) / GIB)),
            "VLLM_CPU_OMP_THREADS_BIND": "auto",
            "VLLM_CPU_NUM_OF_RESERVED_CPU": "1",
        }
        notes.append(
            "CPU cache allocation is rounded up to GiB. Verify NUMA mapping, AMX BF16 and runtime RSS on the actual host."
        )
        rounded = math.ceil(c.memory.get("kv", 0) / GIB) * GIB
        if c.memory.get("total", 0) - c.memory.get("kv", 0) + rounded > c.memory.get("budget", 0):
            missing.append("Rounded CPU KV-cache allocation exceeds memory budget")
    if quant == "fp8" and not model.quantization:
        notes.append("--quantization fp8 quantizes BF16 weights at load time; validate output quality.")
    if a.weight_precision == "int4" and not model.quantization:
        missing.append(
            "INT4 needs a pre-quantized AWQ checkpoint: enter an AWQ model ID and fetch its metadata "
            "(vLLM cannot quantize BF16 weights to INT4 at load time)"
        )
    if a.kv_dtype == "fp8_e4m3":
        argv += ["--calculate-kv-scales"]
        notes.append(
            "FP8 KV can affect accuracy; validate model quality and CUDA/kernel support on the target installation."
        )
    prefix = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    command = (prefix + "\n" if prefix else "") + shlex.join(argv)
    if missing:
        command = "# TEMPLATE — resolve listed requirements before running\n" + command
    return CommandExport(
        status="Runnable command · target validation required"
        if not missing
        else "Template · inputs/compatibility required",
        runnable=not missing,
        missing=list(dict.fromkeys(missing)),
        command=command,
        argv=argv,
        environment=env,
        notes=notes,
        sources=[CLI_SOURCE, CPU_SOURCE],
    )


class DeploymentDetails(Record):
    namespace: str
    name: str = "inference"
    image: str
    node_selector_key: str
    node_selector_value: str
    hf_secret_name: str
    cpu_request: int = Field(ge=1)
    memory_gib: int = Field(ge=1)
    storage: str = "ephemeral"
    pvc_name: str | None = None
    shm_gib: int = Field(default=2, ge=1)
    deployment_confirmed: bool = False

    @model_validator(mode="after")
    def names(self):
        for value in (self.namespace, self.name, self.hf_secret_name):
            if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
                raise ValueError("Use Kubernetes DNS labels for namespace, app name and secret name")
        if self.storage not in {"ephemeral", "existing-rwx-pvc"}:
            raise ValueError("Storage must be ephemeral or an existing RWX PVC")
        if self.storage == "existing-rwx-pvc" and not self.pvc_name:
            raise ValueError("Existing RWX PVC name is required")
        if not self.node_selector_key.strip() or not self.node_selector_value.strip():
            raise ValueError("A verified hardware node selector is required")
        return self


def kubernetes_yaml(
    model: ModelSpec, h: Hardware, c: Candidate, w: Workload, a: Assumptions, details: DeploymentDetails
) -> str:
    command = vllm_command(model, h, c, w, a)
    if not command.runnable:
        raise ValueError("Runnable serving command required before Kubernetes export")
    if not details.deployment_confirmed:
        raise ValueError("Confirm the cluster device plugin, node topology, image, secret and storage first")
    if not details.image.endswith((f":v{a.engine_version}", f":{a.engine_version}")):
        raise ValueError(
            "Image must be tagged with the verified vLLM version; use your target-hardware build"
        )
    if details.memory_gib < math.ceil((c.memory["total"] - c.memory["safety"]) / GIB) + details.shm_gib:
        raise ValueError("Container memory must cover estimated allocations plus shared-memory space")
    labels = {"app": details.name}
    resources = {"cpu": str(details.cpu_request), "memory": f"{details.memory_gib}Gi"}
    if h.kind == "GPU":
        resources["nvidia.com/gpu"] = c.tp
    elif details.cpu_request < 16:
        raise ValueError(
            "CPU profile assumes a full 16-vCPU / 8-physical-core c7i.4xlarge; request at least 16 CPU units and verify SMT allocation"
        )
    env = [
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": details.hf_secret_name, "key": "HF_TOKEN"}},
        },
        {"name": "HF_HOME", "value": "/model-cache/huggingface"},
    ]
    env += [{"name": key, "value": value} for key, value in command.environment.items()]
    volumes = [
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{details.shm_gib}Gi"}},
        {"name": "model-cache", "emptyDir": {}},
    ]
    if details.storage == "existing-rwx-pvc":
        volumes[1] = {"name": "model-cache", "persistentVolumeClaim": {"claimName": details.pvc_name}}
    container = {
        "name": "server",
        "image": details.image,
        "command": command.argv[:2],
        "args": command.argv[2:],
        "env": env,
        "ports": [{"containerPort": 8000}],
        "resources": {"requests": resources, "limits": resources.copy()},
        "startupProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "periodSeconds": 10,
            "failureThreshold": 180,
        },
        "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 10},
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "periodSeconds": 30,
            "failureThreshold": 5,
        },
        "volumeMounts": [
            {"name": "shm", "mountPath": "/dev/shm"},
            {"name": "model-cache", "mountPath": "/model-cache"},
        ],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": details.name, "namespace": details.namespace},
        "spec": {
            "replicas": c.replicas,
            "selector": {"matchLabels": labels},
            "strategy": {"type": "Recreate"},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "terminationGracePeriodSeconds": 60,
                    "nodeSelector": {details.node_selector_key: details.node_selector_value},
                    "containers": [container],
                    "volumes": volumes,
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": details.name, "namespace": details.namespace},
        "spec": {"selector": labels, "ports": [{"port": 8000, "targetPort": 8000}], "type": "ClusterIP"},
    }
    namespace = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": details.namespace}}
    header = (
        "# Complete manifest for the supplied cluster prerequisites; runtime performance remains unvalidated.\n"
        "# Existing Secret must contain HF_TOKEN. No credentials are embedded.\n"
        "# Each TP replica must fit a single node with the cataloged fabric.\n"
        "# No HPA: queue-depth metric/adapter not supplied. kind requires explicitly configured accelerator access.\n"
        "# Storage is ephemeral or a supplied existing RWX PVC; size it for model weights.\n"
    )
    return header + yaml.safe_dump_all([namespace, deployment, service], sort_keys=False)


def sizing_report(
    model: ModelSpec,
    h: Hardware,
    c: Candidate,
    w: Workload,
    a: Assumptions,
    runs: list,
    command: CommandExport,
) -> str:
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model.model_dump(mode="json"),
        "workload": w.model_dump(mode="json"),
        "assumptions": a.model_dump(mode="json"),
        "hardware": h.model_dump(mode="json"),
        "recommendation": c.model_dump(mode="json"),
        "serving_export": command.model_dump(mode="json"),
        "benchmark_provenance": [
            r.model_dump(mode="json", exclude={"raw_result"}) for r in runs if r.run_id in c.run_ids
        ],
        "units": {
            "memory": "bytes; GiB=2^30, GB=10^9",
            "latency": "seconds",
            "bandwidth": "decimal GB/s",
            "cost": "USD",
        },
        "cost_denominator": "Requested peak RPS × average output tokens × 3600 per hour; includes idle headroom",
        "limitations": [
            "Analytical service time excludes queueing/network and is not a p95 guarantee",
            "Calibration residuals are descriptive in-sample errors, not statistical confidence intervals",
            "Pricing is illustrative unless explicitly supplied; verify allocated host/network/storage costs",
            "Disaggregation requires real router/KV-transfer compatibility and separate validation",
        ],
    }
    return json.dumps(scrub(report), indent=2)
