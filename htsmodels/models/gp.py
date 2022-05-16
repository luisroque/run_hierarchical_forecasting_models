import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, cov, changepoints, mean_func):
        super().__init__(train_x, train_y, likelihood)
        self.changepoints = changepoints
        self.mean_module = mean_func(self.changepoints)
        self.covar_module = cov

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
