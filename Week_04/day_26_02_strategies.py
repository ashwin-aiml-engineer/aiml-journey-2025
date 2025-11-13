"""Day 26.02 — Fine-tuning strategies: freezing, discriminative LR, gradual unfreeze
Run time: ~12 minutes

- Demonstrates control flow for freezing/unfreezing and shows discriminative LR pseudocode
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping PyTorch fine-tuning strategies demo")
    raise SystemExit(0)

try:
    import torch
    has_torch = True
except Exception:
    has_torch = False


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_layers(model, n=1):
    # naive: unfreeze last n parameter groups
    params = list(model.named_parameters())
    for name, p in params[-n:]:
        p.requires_grad = True


def discriminative_lr(param_groups, base_lr=1e-5, factor=2.0):
    # param_groups: list of parameter groups from bottom->top
    lrs = [base_lr * (factor ** i) for i in range(len(param_groups))]
    for pg, lr in zip(param_groups, lrs):
        pg['lr'] = lr
    return param_groups

if __name__ == '__main__':
    if not has_torch:
        print('PyTorch not available — pseudocode demo:')
        print('- freeze_backbone(model)')
        print('- train head for few epochs with lr=1e-3')
        print('- unfreeze last block and continue with smaller lr (e.g., 1e-5)')
    else:
        print('PyTorch available — show sample parameter counts:')
        m = torch.nn.Linear(10, 2)
        print('params:', sum(p.numel() for p in m.parameters()))

    # Exercises:
    # - Sketch a plan to gradually unfreeze 3 blocks with decreasing LR per block.
    # - Implement discriminative LR groups for a small model.