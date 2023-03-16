import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * group["lr"])

                # Decay the first and second moment running average coefficient
                p.data.add_(grad, alpha=-group["lr"])
                p.data.addcdiv_(
                    p.data, p.data.abs().add(group["eps"]), value=group["lr"] * beta1
                )
                p.data.addcdiv_(
                    grad, grad.abs().add(group["eps"]), value=group["lr"] * beta2
                )

        return loss
