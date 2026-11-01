import json
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from planner.hf_metadata import MetadataError, apply_overrides, fetch_model, normalize, parse_model_id
from planner.storage import Store


def test_url_normalization_and_validation():
    assert (
        parse_model_id("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main")
        == "Qwen/Qwen2.5-1.5B-Instruct"
    )
    for invalid in ["../secrets", "https://evil.test/a/b", "https://huggingface.co/spaces/a/b"]:
        with pytest.raises(MetadataError):
            parse_model_id(invalid)


def test_no_parameter_inference_from_name():
    m = normalize("owner/model-70B", None, {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    assert m.parameters is None
    assert "parameters" in m.missing


def test_tensor_metadata_preferred_and_override_provenance(model):
    m = normalize("org/model", "a" * 40, model.raw_config, parameters=123)
    assert m.parameters == 123
    assert m.provenance["parameters"].kind == "Repository"
    changed = apply_overrides(m, {"parameters": 456})
    assert changed.provenance["parameters"].original == 123
    assert changed.active_parameters == 456
    assert m.parameters == 123


@pytest.mark.parametrize(
    "extra",
    [
        {"sliding_window": 4096},
        {"kv_lora_rank": 512},
        {"layer_types": ["full_attention", "mamba"]},
    ],
)
def test_special_architectures_unsupported(model, extra):
    config = {**model.raw_config, **extra}
    assert normalize("org/model", None, config).unsupported


def test_qwen3_30b_a3b_moe_total_and_active_parameters():
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "vocab_size": 151936,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "shared_expert_intermediate_size": 768,
    }
    model = normalize("Qwen/Qwen3-30B-A3B", "a" * 40, config, parameters=30_500_000_000)
    assert model.unsupported == []
    assert model.parameters == 30_500_000_000
    assert 3_200_000_000 < model.active_parameters < 3_500_000_000
    assert model.is_moe
    assert model.num_experts == 128
    assert model.num_experts_per_tok == 8


def test_disabled_sliding_window_is_not_assumed_active(model):
    config = {**model.raw_config, "use_sliding_window": False, "sliding_window": 32768}
    assert not normalize("org/model", None, config).unsupported


def test_nested_multimodal_text_normalized_but_unsupported(model):
    m = normalize(
        "org/model", None, {"text_config": model.raw_config, "vision_config": {"hidden_size": 1024}}
    )
    assert m.layers == 32
    assert m.unsupported


def test_remote_code_never_called_and_sha_pinned(monkeypatch, tmp_path, model):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({**model.raw_config, "auto_map": {"AutoModel": "evil.Code"}}))
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, model_id, **kwargs):
            calls.append(("info", model_id, kwargs))
            return SimpleNamespace(sha="b" * 40, safetensors=SimpleNamespace(total=8000000000))

    def download(model_id, filename, **kwargs):
        calls.append(("download", filename, kwargs))
        return config_path

    monkeypatch.setattr("planner.hf_metadata.HfApi", FakeApi)
    monkeypatch.setattr("planner.hf_metadata.hf_hub_download", download)
    store = Store(tmp_path / "cache.sqlite3")
    m = fetch_model("org/model", "tag", store=store)
    assert m.revision == "b" * 40
    assert calls[1][1] == "config.json"
    assert calls[1][2]["revision"] == "b" * 40
    assert all("trust_remote_code" not in c[2] for c in calls)
    assert fetch_model("org/model", "tag", store=store).revision == "b" * 40
    assert len(calls) == 2
    assert store.cached_model("org/model", "b" * 40)


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_actionable_http_errors(monkeypatch, status):
    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            raise HfHubHTTPError(
                "DO NOT LEAK PROVIDER CONTENT",
                response=httpx.Response(
                    status, request=httpx.Request("GET", "https://huggingface.co/org/model")
                ),
            )

    monkeypatch.setattr("planner.hf_metadata.HfApi", FakeApi)
    with pytest.raises(MetadataError) as exc:
        fetch_model("org/model")
    assert "DO NOT LEAK" not in str(exc.value)
    assert len(str(exc.value)) > 20


def test_gated_token_does_not_bypass_approval(monkeypatch):
    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            raise GatedRepoError(
                "gated",
                response=httpx.Response(
                    403, request=httpx.Request("GET", "https://huggingface.co/org/model")
                ),
            )

    monkeypatch.setattr("planner.hf_metadata.HfApi", FakeApi)
    with pytest.raises(MetadataError, match="does not grant approval"):
        fetch_model("org/model")


def test_header_metadata_fallback_is_pinned_and_does_not_fetch_weights(monkeypatch, tmp_path, model):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model.raw_config))
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(
                sha="c" * 40, safetensors=None, siblings=[SimpleNamespace(rfilename="model.safetensors")]
            )

        def get_safetensors_metadata(self, repo, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(parameter_count={"BF16": 123456})

    monkeypatch.setattr("planner.hf_metadata.HfApi", FakeApi)
    monkeypatch.setattr("planner.hf_metadata.hf_hub_download", lambda *args, **kwargs: config_path)
    m = fetch_model("org/model")
    assert m.parameters == 123456
    assert calls[0]["revision"] == "c" * 40
    assert m.tensor_metadata["parameter_count_by_dtype"] == {"BF16": 123456}


def test_non_object_config_is_actionable():
    with pytest.raises(MetadataError, match="JSON object"):
        normalize("org/model", None, [])
