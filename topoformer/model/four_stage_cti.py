import torch
from torch import nn
from torch.nn import functional as F


def clip_update(update, reference):
    limit = 0.3 * reference.norm(dim=-1, keepdim=True)
    scale = (limit / update.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
    return scale * update


class CTIStage(nn.Module):
    def __init__(self, index, token_dim, topo_channels, grid):
        super().__init__()
        native = grid // 2 ** index
        indices = [(j * grid + native - 1) // native for j in range(native)]
        self.register_buffer("native_indices", torch.tensor(indices), persistent=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.visual_norm = nn.LayerNorm(token_dim)
        self.topo_norm = nn.LayerNorm(topo_channels)
        self.out = nn.Linear(512, token_dim)
        self.last_amax = None
        self.visual_gate = nn.Linear(token_dim, 512)
        self.topo_projection = nn.Linear(topo_channels, 512)
        if index == 3:
            self.pool_score = nn.Linear(512, 1, bias=False)
            self.cls_out = nn.Linear(512, token_dim)
            self.cls_gate = nn.Parameter(torch.zeros(()))

    def native(self, cached):
        return cached.index_select(-2, self.native_indices).index_select(-1, self.native_indices)

    def forward(self, tokens, topo_map):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            cls = tokens[:, :1].float()
            patch = tokens[:, 5:].float()
            visual = self.visual_norm(patch)
            topo = topo_map.float()
            cells = self.native(topo)
            aligned = F.interpolate(cells, size=(14, 14), mode="bilinear", align_corners=False)
            ph = self.topo_norm(aligned.flatten(2).transpose(1, 2))
            context = self.visual_gate(visual).sigmoid() * self.topo_projection(ph)
            delta = self.out(context)

            updated_patch = (patch + 0.1 * self.gate.sigmoid() * clip_update(delta, patch)).to(tokens.dtype)
            if hasattr(self, "cls_out"):
                weights = self.pool_score(context).softmax(dim=1)
                pooled = (weights * context).sum(dim=1, keepdim=True)
                cls_delta = clip_update(self.cls_out(pooled), cls)
                updated_cls = (cls + 0.1 * self.cls_gate.sigmoid() * cls_delta).to(tokens.dtype)
                updated = torch.cat((updated_cls, tokens[:, 1:5], updated_patch), dim=1)
            else:
                updated = torch.cat((tokens[:, :5], updated_patch), dim=1)

        self.last_amax = updated.detach().abs().amax()
        return updated


class FourStageCTI(nn.Module):
    stage_starts = (0, 10, 20, 30)

    levels = (0, 1, 2, 3)

    def __init__(self, token_dim, topo_channels, grid):
        super().__init__()
        self.grid = grid
        self.stages = nn.ModuleList(CTIStage(i, token_dim, topo_channels, grid) for i in self.levels)

    def forward(self, block_idx, tokens, topo_levels):
        if block_idx not in self.stage_starts:
            return tokens
        index = block_idx // 10
        topo = topo_levels[f"L{index}"]
        assert len(tokens) == 1 and tokens[0].shape[1] == 201 and topo.shape[-2:] == (self.grid, self.grid)
        return [self.stages[index](tokens[0], topo)]
