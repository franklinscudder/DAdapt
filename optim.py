from torch.optim import Optimizer
from torch import Tensor, zeros_like
from torch.linalg import norm


class SGDD(Optimizer):
    """Stochastic Gradient Descent with D-Adaptation as defined in https://arxiv.org/abs/2301.07733
    (Algorithm 3, Page 8).

    params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, d0=1e-6, yk=1.) -> None:
        defaults = dict(d0=1e-6, yk=1.)
        super().__init__(params, defaults)

        self.d0 = d0
        self.yk = yk

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad = p.grad.data
                
                if len(state) == 0:
                    state["step"] = 0
                    state["d"] = Tensor([group["d0"]])
                    state["s"] = zeros_like(grad)
                    state["lk2_grad2_sum"] = Tensor([0.])

                state["step"] += 1

                d = state["d"]
                yk = group["yk"]

                if state["step"] == 1:
                    group["grad0_norm"] = norm(grad)

                lk = d * yk / group["grad0_norm"]
                lk_grad = lk * grad

                print(grad.shape)

                state["s"].add_(lk_grad)
                p.data.sub_(lk_grad)

                state["lk2_grad2_sum"].add_(lk ** 2 + norm(grad) ** 2)

                d_hat = (norm(state["s"]) ** 2 - state["lk2_grad2_sum"]) / norm(state["s"])
                state["d"] = max(state["d"], d_hat)
        
        return loss
