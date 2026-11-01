import json
import shlex

import pytest
import yaml

from planner.catalog import load_catalog
from planner.exports import DeploymentDetails, kubernetes_yaml, sizing_report, vllm_command
from planner.sizing import estimate


def test_command_is_version_scoped_and_no_secret(model, hardware, workload, assumptions, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-never-export")
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    ex = vllm_command(model, hardware, c, workload, assumptions)
    assert ex.runnable
    assert "--kv-cache-dtype auto" in ex.command
    assert "--revision " + model.revision in ex.command
    assert "--trust-remote-code" not in ex.command
    assert "synthetic-secret" not in ex.command
    assert shlex.split(ex.command) == ex.argv
    report = sizing_report(model, hardware, c, workload, assumptions, [], ex)
    assert "synthetic-secret" not in report
    data = json.loads(report)
    assert data["recommendation"]["memory"]["resident_tokens_per_sequence"] == 4608
    assert "requested" in data["cost_denominator"].lower()


def test_unresolved_model_and_unknown_version_produce_templates(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    model.revision = None
    assert not vllm_command(model, hardware, c, workload, assumptions).runnable
    model.revision = "a" * 40
    assumptions.engine_version = "9.9.9"
    ex = vllm_command(model, hardware, c, workload, assumptions)
    assert not ex.runnable
    assert "TEMPLATE" in ex.command


def test_fp8_only_with_supported_hardware(model, hardware, workload, assumptions):
    assumptions = assumptions.model_copy(update={"kv_dtype": "fp8_e4m3", "kv_bytes": 1})
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    ex = vllm_command(model, hardware, c, workload, assumptions)
    assert ex.runnable
    assert "--calculate-kv-scales" in ex.command
    assert any("accuracy" in n for n in ex.notes)
    a100 = load_catalog()[1]
    bad = estimate(model, a100, workload, assumptions, 1, 32)
    assert not vllm_command(model, a100, bad, workload, assumptions).runnable


def details(**updates):
    data = dict(
        namespace="inference",
        image="vllm/vllm-openai:v0.10.2",
        node_selector_key="accelerator",
        node_selector_value="h100",
        hf_secret_name="hf-read",
        cpu_request=8,
        memory_gib=64,
        deployment_confirmed=True,
    )
    data.update(updates)
    return DeploymentDetails(**data)


def test_kubernetes_resources_secret_probes_and_storage(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    result = kubernetes_yaml(
        model, hardware, c, workload, assumptions, details(storage="existing-rwx-pvc", pvc_name="models")
    )
    docs = list(yaml.safe_load_all(result))
    assert [d["kind"] for d in docs] == ["Namespace", "Deployment", "Service"]
    pod = docs[1]["spec"]["template"]["spec"]
    server = pod["containers"][0]
    assert server["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert server["env"][0]["valueFrom"]["secretKeyRef"]["name"] == "hf-read"
    assert server["readinessProbe"]["httpGet"]["path"] == "/health"
    assert pod["volumes"][1]["persistentVolumeClaim"]["claimName"] == "models"
    assert "HorizontalPodAutoscaler" not in result


def test_kubernetes_blocked_without_prerequisites(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 32)
    for invalid in [dict(deployment_confirmed=False), dict(memory_gib=1), dict(image="vllm:latest")]:
        with pytest.raises(ValueError):
            kubernetes_yaml(model, hardware, c, workload, assumptions, details(**invalid))


def test_fp8_weights_and_fp8_kv_add_quantization_and_kv_flags(model, hardware, workload):
    from planner.models import Assumptions

    a = Assumptions(weight_precision="fp8", kv_dtype="fp8_e4m3", kv_bytes=1)
    assert a.weight_bytes == 1
    c = estimate(model, hardware, workload, a, 1, 8)
    assert c.feasible
    ex = vllm_command(model, hardware, c, workload, a)
    assert ex.runnable
    assert ex.argv[ex.argv.index("--quantization") + 1] == "fp8"
    assert ex.argv[ex.argv.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    # Halving weight bytes frees memory compared with the BF16 default.
    bf16 = estimate(model, hardware, workload, Assumptions(), 1, 8)
    assert c.memory["weights"] == pytest.approx(bf16.memory["weights"] / 2)


def test_bf16_command_has_no_quantization_flag_and_auto_kv(model, hardware, workload, assumptions):
    c = estimate(model, hardware, workload, assumptions, 1, 8)
    ex = vllm_command(model, hardware, c, workload, assumptions)
    assert "--quantization" not in ex.argv
    assert ex.argv[ex.argv.index("--kv-cache-dtype") + 1] == "auto"


def test_int4_is_sized_but_template_until_checkpoint_is_prequantized(model, hardware, workload):
    from planner.models import Assumptions

    a = Assumptions(weight_precision="int4")
    c = estimate(model, hardware, workload, a, 1, 8)
    assert c.feasible and a.weight_bytes == 0.5
    ex = vllm_command(model, hardware, c, workload, a)
    assert not ex.runnable and "--quantization awq" in ex.command
    assert any("pre-quantized AWQ" in m for m in ex.missing)
    # A real AWQ checkpoint (known storage bytes) exports as runnable with the matching flag.
    awq = model.model_copy(
        update={
            "quantization": {"quant_method": "awq", "bits": 4},
            "effective_weight_bytes": 0.55,
            "weight_dtype": "int4",
        }
    )
    c = estimate(awq, hardware, workload, Assumptions(), 1, 8)
    ex = vllm_command(awq, hardware, c, workload, Assumptions())
    assert ex.runnable and ex.argv[ex.argv.index("--quantization") + 1] == "awq"


def test_weight_precision_must_match_bytes_and_cpu_is_rejected(model, workload):
    from pydantic import ValidationError

    from planner.models import Assumptions

    with pytest.raises(ValidationError):
        Assumptions(weight_precision="fp8", weight_bytes=2)
    cpu = next(h for h in load_catalog() if h.kind == "CPU")
    assert not estimate(model, cpu, workload, Assumptions(weight_precision="fp8"), 1, 1).feasible
