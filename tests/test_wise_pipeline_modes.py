"""Regression tests for isolated WISE baseline and LAP+PMRL modes."""

from pathlib import Path
from types import SimpleNamespace

import torch

from easyeditor.models.wise.WISE import WISEMultimodal
from easyeditor.models.wise.wise_multimodal_hparams import WISEMultimodalHyperParams
from easyeditor.models.wise.utils import multimodal_tokenize
from easyeditor.evaluate.multimodal_evaluate import prepare_multimodal_hf_edit
from easyeditor.editors.multimodal_editor import MultimodalEditor


class _Output:
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels


class _Adapter:
    def __init__(self):
        self.original_layer_output = None
        self.new_weight_layer_output = None
        self.wise_edit_activation_count = None


class _LegacyBlipWrapper:
    config = None

    def __init__(self):
        self.adapter = _Adapter()
        self.calls = []

    def __call__(self, batch):
        self.calls.append(batch["kind"])
        if batch["kind"] == "edit":
            self.adapter.original_layer_output = torch.ones(1, 3, 4)
            self.adapter.new_weight_layer_output = torch.ones(1, 3, 4) * 2
            return _Output(
                torch.zeros(1, 4, 5, requires_grad=True),
                torch.tensor([[-100, -100, 1, 2]]),
            )
        self.adapter.original_layer_output = torch.ones(1, 5, 4) * 3
        self.adapter.new_weight_layer_output = torch.ones(1, 5, 4) * 4
        return _Output(torch.zeros(1, 6, 5), torch.full((1, 6), -100))


def _wise_shell(using_extra=False):
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    wise.config = SimpleNamespace(
        model_name="blip2",
        batch_size=1,
        using_extra=using_extra,
    )
    wise.model = _LegacyBlipWrapper()
    wise.edit_module = SimpleNamespace(adapter=wise.model.adapter)
    wise.layer_name = "adapter"
    wise.get_adapter_layer = lambda: wise.model.adapter
    return wise


def test_baseline_mode_only_runs_canonical_edit_and_locality_forwards():
    wise = _wise_shell(using_extra=False)
    wise._compute_lap_pmrl_loss = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("enhancement path must not run in baseline mode")
    )

    loss = wise._cal_ft_loss(
        [{"kind": "edit"}, {"kind": "locality"}],
        {"labels": torch.tensor([[-100, 1, 2]])},
        last_prompt_token_loc=torch.tensor([0]),
        ans_token_len=2,
    )

    assert torch.isfinite(loss)
    assert wise.model.calls == ["edit", "locality"]
    assert wise.model.adapter.original_layer_output.shape == (8, 4)
    assert wise.model.adapter.wise_edit_activation_count == 3


def test_hparams_define_explicit_baseline_and_lap_pmrl_modes():
    root = Path(__file__).resolve().parents[1]
    baseline = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/blip2_ic_baseline.yaml")
    )
    enhanced = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/blip2_ic_lap_pmrl.yaml")
    )

    assert baseline.using_extra is False
    assert baseline.using_lap is False
    assert baseline.using_pmrl is False
    assert enhanced.using_extra is True
    assert enhanced.using_lap is True
    assert enhanced.using_pmrl is True
    assert enhanced.num_rephrase > 1


def test_caption_dataset_has_no_hard_coded_record_39_filter():
    source = (
        Path(__file__).resolve().parents[1]
        / "easyeditor/dataset/coco_caption.py"
    ).read_text(encoding="utf-8")
    assert "if i < 39" not in source
    assert "if i == 40" not in source


def test_qwen_configs_define_explicit_baseline_and_lap_pmrl_modes():
    root = Path(__file__).resolve().parents[1]
    baseline = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/qwen2vl_ic_baseline.yaml")
    )
    enhanced = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/qwen2vl_ic_lap_pmrl.yaml")
    )

    assert baseline.model_name.endswith("Qwen2-VL-7B-Instruct")
    assert baseline.using_extra is False
    assert baseline.using_lap is False
    assert baseline.using_pmrl is False
    assert enhanced.using_extra is True
    assert enhanced.using_lap is True
    assert enhanced.using_pmrl is True
    assert enhanced.num_rephrase > 1


def test_llava_configs_define_explicit_baseline_and_lap_pmrl_modes():
    root = Path(__file__).resolve().parents[1]
    baseline = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/llavaov_ic_baseline.yaml")
    )
    enhanced = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/llavaov_ic_lap_pmrl.yaml")
    )

    assert baseline.model_name.endswith("llava-onevision-qwen2-7b-ov-hf")
    assert baseline.using_extra is False
    assert baseline.using_lap is False
    assert baseline.using_pmrl is False
    assert enhanced.using_extra is True
    assert enhanced.using_lap is True
    assert enhanced.using_pmrl is True
    assert enhanced.num_rephrase > 1


def test_llava_vqa_configs_define_explicit_baseline_and_lap_pmrl_modes():
    root = Path(__file__).resolve().parents[1]
    baseline = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/llavaov_vqa_baseline.yaml")
    )
    enhanced = WISEMultimodalHyperParams.from_hparams(
        str(root / "hparams/WISE/llavaov_vqa_lap_pmrl.yaml")
    )

    assert baseline.model_name.endswith("llava-onevision-qwen2-7b-ov-hf")
    assert (baseline.using_extra, baseline.using_lap, baseline.using_pmrl) == (False, False, False)
    assert (enhanced.using_extra, enhanced.using_lap, enhanced.using_pmrl) == (True, True, True)
    assert enhanced.lar_target_loss_weight == 0.25


def test_hf_multimodal_ft_loss_uses_masked_labels_and_gates_pmrl():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    wise.config = SimpleNamespace(
        model_name="/models/qwen2-vl-2b-instruct",
        batch_size=1,
        using_extra=False,
        using_lap=False,
        using_pmrl=False,
    )

    class _HFModel:
        config = SimpleNamespace()

        def __init__(self):
            self.adapter = _Adapter()
            self.calls = 0

        def __call__(self, **batch):
            self.calls += 1
            seq = batch["input_ids"].size(1)
            self.adapter.original_layer_output = torch.ones(1, seq, 4) * self.calls
            self.adapter.new_weight_layer_output = torch.ones(1, seq, 4) * (self.calls + 1)
            logits = torch.zeros(1, seq, 7, requires_grad=True)
            return SimpleNamespace(logits=logits)

    wise.model = _HFModel()
    wise.get_adapter_layer = lambda: wise.model.adapter
    wise._compute_hf_lap_pmrl_loss = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("enhancement path must not run in baseline mode")
    )
    edit_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }
    locality_inputs = {
        "input_ids": torch.tensor([[1, 5, 6]]),
        "labels": torch.tensor([[-100, -100, 6]]),
    }
    loss = wise._cal_ft_loss(
        [edit_inputs, locality_inputs],
        {"input_ids": edit_inputs["input_ids"], "labels": edit_inputs["labels"]},
        last_prompt_token_loc=torch.tensor([2, 4]),
        ans_token_len=2,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_pmrl_alignment_is_non_degenerate_and_regularization_is_weighted():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    base = torch.randn(6, 8, requires_grad=True)
    views = [base, base + 0.05, base - 0.03]
    total, components = wise.pmrl_loss(
        views,
        tau_alignment=0.05,
        tau_regularization=0.1,
        alignment_weight=1.0,
        regularization_weight=0.1,
        return_components=True,
    )
    assert components["alignment"].item() > 0
    assert components["regularization"].item() >= 0
    expected = components["alignment"] + 0.1 * components["alignment"].detach()
    assert torch.allclose(total, expected, rtol=1e-5, atol=1e-6)
    total.backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_hf_lap_target_loss_keeps_logits_in_model_dtype():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    logits = torch.randn(1, 4, 7, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4]])

    loss = wise._hf_target_loss(logits, labels, reduction="sum")

    assert torch.isfinite(loss)
    assert loss.dtype == torch.bfloat16
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_lar_randomized_pgd_stays_in_budget_and_generates_diverse_views():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    torch.manual_seed(7)
    base = torch.zeros(1, 6, 4)
    gradient = torch.ones_like(base)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    perturbations = wise._build_lar_perturbations(
        base,
        gradient,
        mask,
        num_views=4,
        epsilon=0.1,
        random_start=True,
        pgd_steps=2,
        step_size=0.05,
    )
    assert len(perturbations) == 4
    for delta in perturbations:
        assert torch.all(delta[:, ~mask[0], :] == 0)
        assert delta.float().flatten(1).norm(dim=1).max().item() <= 0.10001
    assert any(not torch.equal(perturbations[0], item) for item in perturbations[1:])


def test_llava_hf_lap_inputs_use_image_sizes_without_qwen_rope():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    wise.config = SimpleNamespace(model_name="/models/llava-onevision-qwen2-7b")

    class _Core:
        def __init__(self):
            self.received_image_sizes = None

        def get_input_embeddings(self):
            return lambda input_ids: torch.zeros(input_ids.size(0), input_ids.size(1), 4)

        def get_image_features(
            self, pixel_values, image_sizes, vision_feature_layer=None,
            vision_feature_select_strategy=None, batch_num_images=None,
        ):
            self.received_image_sizes = image_sizes
            return [torch.ones(2, 4)]

        def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
            mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
            mask[:, :2, :] = True
            return mask, None

        def get_rope_index(self, *args, **kwargs):
            raise AssertionError("LLaVA must not use the Qwen RoPE API")

    core = _Core()
    wise.model = SimpleNamespace(model=core)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_sizes": torch.tensor([[2, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3]]),
    }
    fused, visual_mask, forward_kwargs = wise._prepare_hf_lap_inputs(batch)
    assert core.received_image_sizes is batch["image_sizes"]
    assert fused.shape == (1, 3, 4)
    assert visual_mask.tolist() == [[True, True, False]]
    assert "position_ids" not in forward_kwargs
    assert "labels" not in forward_kwargs


def test_random_start_lar_accepts_saturated_zero_base_gradient():
    wise = object.__new__(WISEMultimodal)
    torch.nn.Module.__init__(wise)
    torch.manual_seed(9)
    base = torch.zeros(1, 5, 3)
    gradient = torch.zeros_like(base)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    perturbations = wise._build_lar_perturbations(
        base, gradient, mask, num_views=3, epsilon=0.1,
        random_start=True, pgd_steps=1, step_size=0.1,
    )
    assert len(perturbations) == 3
    assert all(torch.isfinite(delta).all() for delta in perturbations)
    assert all(delta.float().flatten(1).norm(dim=1).max().item() <= 0.10001 for delta in perturbations)
    assert any(torch.count_nonzero(delta).item() > 0 for delta in perturbations)


def test_qwen_wise_edit_prompt_matches_evaluation_chat_template():
    class _Batch(dict):
        def to(self, *args, **kwargs):
            return self

    class _Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            rows = []
            for text in texts:
                rows.append([ord(char) for char in text])
            width = max(len(row) for row in rows)
            rows = [row + [0] * (width - len(row)) for row in rows]
            return _Batch(input_ids=torch.tensor(rows))

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            content = messages[0]["content"]
            text = next(item["text"] for item in content if item["type"] == "text")
            has_image = any(item["type"] == "image" for item in content)
            image = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
            return f"<system>helper</system><user>{image}{text}</user><assistant>"

    class _Processor:
        tokenizer = _Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return self.tokenizer.apply_chat_template(*args, **kwargs)

        def __call__(self, text, images=None, **kwargs):
            return self.tokenizer(text, **kwargs)

    processor = _Processor()
    hparams = SimpleNamespace(
        model_name="qwen2-vl", dtype=torch.float32, device="cpu",
        use_chat_template=True, objective_optimization="only_label",
    )
    request = {
        "prompt": "describe",
        "target": "answer",
        "image": object(),
        "file_type": "image",
        "locality_prompt": "local question",
        "locality_ground_truth": "local answer",
    }
    wise_inputs, *_ = multimodal_tokenize(
        [request], processor, "cpu", hparams=hparams
    )
    evaluation = prepare_multimodal_hf_edit(
        hparams, processor, "answer", "describe", request["image"], "image"
    )["multimodal_inputs"]
    assert torch.equal(wise_inputs[0]["input_ids"], evaluation["input_ids"])


def test_qwen_singleton_reset_restores_original_layer():
    class _Layer(torch.nn.Linear):
        pass

    model = torch.nn.Module()
    model.config = SimpleNamespace(hidden_act="silu")
    model.device = torch.device("cpu")
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.language_model.layers[0].mlp = torch.nn.Module()
    original = _Layer(4, 4, bias=False)
    model.model.language_model.layers[0].mlp.down_proj = original
    config = SimpleNamespace(
        inner_params=["model.language_model.layers.0.mlp.down_proj.weight"],
        sequential_edit=False, model_name="qwen2-vl", retrieve=True,
        hidden_act="silu", device="cpu",
    )
    wise = WISEMultimodal(config=config, model=model, device="cpu")
    assert wise.get_adapter_layer().original_layer is not original
    wise.reset_layer()
    restored = model.model.language_model.layers[0].mlp.down_proj
    assert not hasattr(restored, "new_weight")
    assert torch.equal(restored.weight, original.weight)
