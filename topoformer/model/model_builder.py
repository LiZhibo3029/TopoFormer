import types

import torch

from .disentanglement import OrthogonalDisentangler
from .lora import inject_lora_vit
from .convnext_fusion import build_convnextv2, GatedFusionFD
from .four_stage_cti import FourStageCTI


def build_topoformer_model(
    device,
    fd_label_names,
    dino_local_ckpt: str,
    topo_c: int,
    convnext_ckpt: str,
    cti_grid: int,
):
    vit = OrthogonalDisentangler(
        encoder_source="dinov3",
        encoder_name="dinov3_vit7b16",
        pretrained_path=dino_local_ckpt,
        out_dim=len(fd_label_names),
        frozen_blocks=999,
        c_mode="cls",
        head_style="mlp",
        debug_shapes=False,
    )

    for p in vit.encoder.parameters():
        p.requires_grad = False

    if str(device).startswith("cuda"):
        vit.encoder.to(dtype=torch.float16)

    count = inject_lora_vit(vit.encoder, r=16, alpha=16.0, dropout=0.05, targets=("qkv", "proj"))
    if count != 80:
        raise ValueError(f"Expected 80 LoRA attention projections across 40 blocks, got {count}")
    for name, parameter in vit.encoder.named_parameters():
        if "lora_" in name.lower():
            parameter.requires_grad = True
    print(f"[lora] injected into encoder: replaced_linears={count}")

    vit.adapter_disc.float()
    vit.adapter_indisc.float()
    vit.decoderA.float()
    vit.decoderB.float()
    vit.decoderA_fused.float()
    vit.proj_disc_simple.float()
    vit.proj_indisc_simple.float()
    vit.pool_attn.float()

    _orig_encode = vit._encode

    def _encode_fp32(self, x, topo_levels=None):
        feat = _orig_encode(x, topo_levels=topo_levels).float()
        return feat

    vit._encode = types.MethodType(_encode_fp32, vit)

    cnx = build_convnextv2(convnext_ckpt, "convnextv2_base", device)

    model = GatedFusionFD(vit, cnx, dnfd=True)
    model.proj_cnx.float()
    model.gate_ln.float()
    model.gate_fc.float()

    model = model.to(device)
    if len(vit.encoder.blocks) != 40 or vit.encoder.n_storage_tokens != 4:
        raise ValueError("TopoCTI requires the ViT-7B/16 token layout")
    if topo_c != 400 or cti_grid not in (8, 14):
        raise ValueError("Expected four 400-channel DeepHisto or CRC TopoFPN levels")
    with torch.random.fork_rng(devices=[]):
        cti = FourStageCTI(vit.encoder.embed_dim, topo_c, cti_grid)
    vit.encoder._topo_hook = cti.to(device=device, dtype=torch.float32)
    # Surface any async CUDA illegal-access / ECC error that was queued
    # by an earlier op (e.g. previous `model.to()` from preflight), so
    # the user gets a clean stack instead of a misleading one.
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            # Re-raise with a clearer message.
            raise RuntimeError(
                f"CUDA error during `model.to({device})` — this is almost "
                f"always a *previous* async illegal memory access (ECC / "
                f"OOB) that bubbled up here. Check earlier GPU work and "
                f"consider `CUDA_LAUNCH_BLOCKING=1`."
            ) from None
    return model