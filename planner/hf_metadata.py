"""Metadata-only Hugging Face access. Never imports Transformers or executes repo code."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils import HFValidationError, validate_repo_id

from planner.models import ModelSpec, Provenance
from planner.security import scrub
from planner.storage import Store


class MetadataError(ValueError):
    pass


def parse_model_id(value: str) -> str:
    value = value.strip()
    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.hostname != "huggingface.co" or parsed.username or parsed.password:
            raise MetadataError("Use a huggingface.co model URL or an owner/model ID")
        segments = parsed.path.strip("/").split("/")
        if not segments or segments[0] in {"datasets", "spaces"}:
            raise MetadataError("Select a model repository, not a dataset or Space")
        value = "/".join(segments[:2])
    try:
        validate_repo_id(value)
    except HFValidationError:
        raise MetadataError(
            "Invalid model ID. Use owner/model, for example Qwen/Qwen2.5-1.5B-Instruct"
        ) from None
    return value


def normalize(
    model_id: str,
    sha: str | None,
    config: dict,
    parameters: int | None = None,
    requested_revision: str = "main",
) -> ModelSpec:
    if not isinstance(config, dict):
        raise MetadataError("config.json must contain a JSON object")
    nested_key = next(
        (key for key in ("text_config", "language_config", "llm_config") if config.get(key)), None
    )
    text = config[nested_key] if nested_key else config
    if not isinstance(text, dict):
        raise MetadataError("The nested text configuration is not a JSON object")
    root = f"config.json.{nested_key}" if nested_key else "config.json"
    prov = {}

    def get(field, names, default=None):
        for name in names:
            if text.get(name) is not None:
                prov[field] = Provenance(source=f"{root}.{name}", kind="Repository")
                return text[name]
        if default is not None:
            prov[field] = Provenance(source="Dense architecture convention", kind="Derived estimate")
        return default

    model_type = get("model_type", ["model_type"], config.get("model_type", "unknown"))
    arch = (config.get("architectures") or text.get("architectures") or [model_type])[0]
    prov["architecture"] = Provenance(source="config.json.architectures / model_type", kind="Repository")
    layers = get("layers", ["num_hidden_layers", "n_layer", "num_layers"])
    heads = get("attention_heads", ["num_attention_heads", "n_head", "n_heads"])
    hidden = get("hidden_size", ["hidden_size", "n_embd", "d_model"])
    dense_types = {"llama", "qwen2", "qwen3", "mistral"}
    moe_types = {"qwen3_moe", "qwen2_moe", "mixtral"}
    supported_types = dense_types | moe_types
    kv = get(
        "kv_heads",
        ["num_key_value_heads", "num_kv_heads", "n_head_kv"],
        heads if model_type in supported_types else None,
    )
    dim = get("head_dim", ["head_dim", "attention_head_dim"])
    if dim is None and hidden and heads and hidden % heads == 0:
        dim = hidden // heads
        prov["head_dim"] = Provenance(
            source=f"{root}.hidden_size / num_attention_heads", kind="Derived estimate"
        )
    context = get("context_limit", ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"])
    dtype = get("weight_dtype", ["torch_dtype", "dtype"], "unknown")
    quant = config.get("quantization_config") or text.get("quantization_config") or {}
    prov["quantization"] = Provenance(
        source="config.json.quantization_config (empty when absent)", kind="Repository"
    )
    num_experts = get("num_experts", ["num_experts", "num_local_experts", "n_routed_experts"])
    experts_per_tok = get(
        "num_experts_per_tok", ["num_experts_per_tok", "num_experts_per_token", "num_selected_experts", "top_k"]
    )
    expert_size = get(
        "expert_intermediate_size",
        ["moe_intermediate_size", "expert_intermediate_size", "ffn_dim_per_expert"],
    )
    is_moe = bool(num_experts and experts_per_tok)
    notes, unsupported = [], []
    if model_type not in supported_types:
        unsupported.append(f"No validated estimator for model_type={model_type}")
    known_architectures = {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "MistralForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen2MoeForCausalLM",
        "MixtralForCausalLM",
    }
    if arch not in known_architectures:
        unsupported.append(f"Runtime compatibility is unverified for architecture={arch}")
    if (
        any(config.get(k) is not None for k in ("vision_config", "audio_config", "image_token_index"))
        or text is not config
    ):
        unsupported.append("Multimodal/nested text repository: vision/audio/embedding costs are not modeled")
    if any(text.get(k) for k in ("num_local_experts", "num_experts", "n_routed_experts")) and not is_moe:
        unsupported.append("Mixture-of-experts config is missing experts-per-token metadata")
    sliding = bool(text.get("sliding_window")) and text.get("use_sliding_window") is not False
    layer_types = text.get("layer_types") or []
    hybrid = any(layer not in {"full_attention", "global_attention"} for layer in layer_types)
    if sliding or text.get("use_sliding_window") or hybrid:
        unsupported.append(
            "Sliding-window or hybrid layer attention requires an architecture-specific cache estimator"
        )
    if any(text.get(k) is not None for k in ("kv_lora_rank", "qk_rope_head_dim", "mamba_d_state", "ssm_cfg")):
        unsupported.append("Alternative KV/state-space representation is unsupported")
    active_parameters = None
    if parameters:
        prov["parameters"] = Provenance(
            source="Hugging Face safetensors tensor metadata.total", kind="Repository"
        )
    elif model_type in supported_types and not unsupported:
        intermediate, vocab = text.get("intermediate_size"), text.get("vocab_size")
        if is_moe and all((layers, heads, hidden, kv, dim, expert_size, vocab)):
            tied = text.get("tie_word_embeddings", False)
            shared_size = text.get("shared_expert_intermediate_size", 0) or 0
            attention = hidden * heads * dim * 2 + hidden * kv * dim * 2
            routed = num_experts * 3 * hidden * expert_size
            shared = 3 * hidden * shared_size
            router = hidden * num_experts
            parameters = int(
                layers * (attention + routed + shared + router + 2 * hidden)
                + hidden * vocab * (1 if tied else 2)
                + hidden
            )
            prov["parameters"] = Provenance(
                source="Derived MoE tensor shapes; biases excluded", kind="Derived estimate"
            )
            notes.append("Total parameter count is a shape-based MoE estimate, not repository tensor metadata")
        elif not is_moe and all((layers, heads, hidden, kv, dim, intermediate, vocab)):
            # Llama-style gated MLP + attention + RMSNorm, excludes unknown biases.
            tied = text.get("tie_word_embeddings", False)
            parameters = int(
                layers
                * (hidden * heads * dim * 2 + hidden * kv * dim * 2 + 3 * hidden * intermediate + 2 * hidden)
                + hidden * vocab * (1 if tied else 2)
                + hidden
            )
            prov["parameters"] = Provenance(
                source="Derived dense gated-MLP tensor shapes; biases excluded", kind="Derived estimate"
            )
            notes.append("Parameter count is a shape-based estimate, not repository tensor metadata")
    if is_moe and parameters and all((layers, hidden, expert_size)):
        routed_total = layers * num_experts * 3 * hidden * expert_size
        routed_active = layers * experts_per_tok * 3 * hidden * expert_size
        active_parameters = int(parameters - routed_total + routed_active)
        if active_parameters <= 0 or active_parameters > parameters:
            active_parameters = None
            unsupported.append("MoE active-parameter estimate is inconsistent with the total parameter count")
        else:
            prov["active_parameters"] = Provenance(
                source="Total parameters - inactive routed experts (top-k active experts per layer)",
                kind="Derived estimate",
            )
            notes.append(
                "MoE memory uses all parameters; decode compute and weight bandwidth use active parameters per token"
            )
    if config.get("auto_map"):
        notes.append(
            "Repository declares custom code; it is never executed. Runtime support needs verification"
        )
    if quant:
        notes.append(
            "Quantized repository: supply effective storage bytes/parameter plus format overhead; kernel support must be verified"
        )
    if dtype == "unknown":
        notes.append("Weight dtype is absent; choose an explicit serving precision")
    if text.get("rope_scaling"):
        notes.append(
            "RoPE scaling is recorded; context limit is not expanded beyond the explicit config value"
        )
    spec = ModelSpec(
        model_id=model_id,
        revision=sha,
        requested_revision=requested_revision,
        model_type=model_type,
        architecture=arch,
        parameters=parameters,
        active_parameters=active_parameters if is_moe else parameters,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        expert_intermediate_size=expert_size,
        layers=layers,
        attention_heads=heads,
        kv_heads=kv,
        head_dim=dim,
        hidden_size=hidden,
        context_limit=context,
        weight_dtype=dtype,
        quantization=scrub(quant),
        provenance=prov,
        unsupported=unsupported,
        notes=notes,
        raw_config=scrub(config),
    )
    if spec.active_parameters and not spec.is_moe:
        spec.provenance["active_parameters"] = Provenance(
            source="Dense model: active = total parameters", kind="Derived estimate"
        )
    return spec


def fetch_model(
    value: str,
    revision: str = "main",
    token: str | None = None,
    store: Store | None = None,
    refresh: bool = False,
) -> ModelSpec:
    model_id = parse_model_id(value)
    revision = revision.strip() or "main"
    if store and not refresh and (cached := store.cached_model(model_id, revision)):
        return cached
    try:
        api = HfApi(token=token)
        info = api.model_info(model_id, revision=revision, files_metadata=False, timeout=20)
        if not info.sha:
            raise MetadataError("Hub did not resolve a commit SHA; lookup cannot be pinned safely")
        path = hf_hub_download(model_id, "config.json", revision=info.sha, token=token, etag_timeout=20)
        with open(path) as f:
            config = json.load(f)
        tensors = getattr(info, "safetensors", None)
        total = getattr(tensors, "total", None)
        counts = getattr(tensors, "parameters", None) or {}
        tensor_source = "Hub model_info.safetensors"
        tensor_note = None
        siblings = {getattr(item, "rfilename", "") for item in getattr(info, "siblings", []) or []}
        if not total and siblings.intersection({"model.safetensors", "model.safetensors.index.json"}):
            try:
                headers = api.get_safetensors_metadata(model_id, revision=info.sha, token=token, timeout=15)
                counts = headers.parameter_count
                total = sum(counts.values()) or None
                tensor_source = "Pinned safetensors headers via official Hub metadata client"
            except Exception:
                tensor_note = "Tensor headers were unavailable; parameter count is derived or missing, never inferred from the name"
        spec = normalize(model_id, info.sha, config, total, revision)
        spec.tensor_metadata = {"source": tensor_source, "parameter_count_by_dtype": counts}
        if total:
            spec.provenance["parameters"] = Provenance(source=tensor_source, kind="Repository")
        if tensor_note:
            spec.notes.append(tensor_note)
        if store:
            store.cache_model(spec)
        return spec
    except GatedRepoError:
        raise MetadataError(
            "Gated repository: request access on the model page, then set HF_TOKEN with approved read access. A token alone does not grant approval."
        ) from None
    except RevisionNotFoundError:
        raise MetadataError("Revision not found. Check the branch, tag, or commit SHA.") from None
    except RepositoryNotFoundError:
        raise MetadataError(
            "Repository not found or access denied. Check the ID, HF_TOKEN read scope, and private/gated access approval."
        ) from None
    except EntryNotFoundError:
        raise MetadataError(
            "config.json is missing. Use a supported model repository or inspect a local configuration with explicit overrides."
        ) from None
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            message = "Hub rate limit reached. Retry later or configure HF_TOKEN; cached successful lookups remain available."
        elif status in (401, 403):
            message = "Hub access denied. Check HF_TOKEN permissions and model access approval."
        else:
            message = "Hub request failed. Check connectivity and retry; cached metadata remains available."
        raise MetadataError(message) from None
    except (httpx.HTTPError, OSError):
        raise MetadataError(
            "Network or local-cache failure. Check connectivity and HF cache permissions, then retry."
        ) from None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, MetadataError):
            raise
        raise MetadataError(
            "Model configuration could not be normalized. Inspect its config.json and supply missing fields explicitly."
        ) from None


def apply_overrides(model: ModelSpec, overrides: dict) -> ModelSpec:
    allowed = {
        "parameters",
        "active_parameters",
        "layers",
        "attention_heads",
        "kv_heads",
        "head_dim",
        "hidden_size",
        "context_limit",
        "weight_dtype",
        "effective_weight_bytes",
    }
    data = model.model_dump()
    for field, value in overrides.items():
        if field not in allowed:
            raise ValueError(f"Unsupported override field: {field}")
        data["provenance"][field] = Provenance(
            source="Explicit user override", kind="Override", original=getattr(model, field)
        ).model_dump()
        data[field] = value
        if field == "parameters" and not model.unsupported and "active_parameters" not in overrides:
            data["active_parameters"] = value
            data["provenance"]["active_parameters"] = Provenance(
                source="Dense active = overridden total", kind="Override"
            ).model_dump()
    return ModelSpec.model_validate(data)
