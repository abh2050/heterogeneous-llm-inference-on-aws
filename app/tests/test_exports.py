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
