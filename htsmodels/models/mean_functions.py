import gpytorch
import torch


class ZeroMean(gpytorch.means.Mean):
    def forward(self, input):
        return torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)


class LinearMean(gpytorch.means.Mean):
    def __init__(self, bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.tensor([[0.1]])))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.tensor([[0.1]])))
        else:
            self.bias = None

    def forward(self, x):
        x = x.float()
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res


class PiecewiseLinearMean(gpytorch.means.Mean):
    def __init__(self, changepoints):
        super().__init__()
        self.changepoints = changepoints
        self.register_parameter(name="k",
                                parameter=torch.nn.Parameter(torch.tensor([[0.1]])))
        self.register_parameter(name="m",
                                parameter=torch.nn.Parameter(torch.tensor([[0.1]])))
        self.register_parameter(name="b",
                                parameter=torch.nn.Parameter(torch.tile(torch.tensor([0.1]), (len(changepoints),))))

    def forward(self, x):
        x = x.float()
        A = (0.5 * (1.0 + torch.sgn(torch.tile(x.reshape((-1, 1)), (1, 4)) - self.changepoints))).float()

        res = ((self.k + torch.matmul(A, self.b.reshape((-1, 1)))) * x
               + (self.m + torch.matmul(A, (-torch.from_numpy(self.changepoints).float() * self.b))).reshape(-1, 1))

        return res.reshape((-1,))
