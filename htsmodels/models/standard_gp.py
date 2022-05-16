import torch
import numpy as np
import gpytorch
from .gp import ExactGPModel
from .mean_functions import PiecewiseLinearMean, LinearMean, ZeroMean
from gpytorch.mlls import SumMarginalLogLikelihood
from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
import pickle
import tsaugmentation as tsag
from pathlib import Path
import time


class SGP:

    def __init__(self, dataset, groups, input_dir='./', n_samples=500,
                 store_prediction_samples=False,
                 store_prediction_points=False):
        self.dataset = dataset
        self.groups = groups
        self.input_dir = input_dir
        self.timer_start = time.time()
        self.wall_time_preprocess = None
        self.wall_time_build_model = None
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        self.groups, self.dt = self._preprocess()
        self._create_directories()
        self.n_samples = n_samples
        self.store_prediction_samples = store_prediction_samples
        self.store_prediction_points = store_prediction_points

        self.train_x = torch.arange(groups['train']['n'])
        self.train_x = self.train_x.type(torch.DoubleTensor)
        self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = torch.from_numpy(groups['train']['data'])

    def _create_directories(self):
        # Create directory to store results if does not exist
        Path(f'{self.input_dir}results').mkdir(parents=True, exist_ok=True)

    def _preprocess(self):
        dt = tsag.preprocessing.utils.DataTransform(self.groups)
        self.wall_time_preprocess = time.time() - self.timer_start
        return dt.std_transf_train(), dt

    def _build_cov_matrices(self):
        covs = []
        for i in range(self.groups['train']['s']):
            # RBF kernel
            rbf_kernel = gpytorch.kernels.RBFKernel()
            rbf_kernel.lengthscale = torch.tensor([1.])
            scale_rbf_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
            scale_rbf_kernel.outputscale = torch.tensor([0.5])

            # Periodic Kernel
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            periodic_kernel.period_length = torch.tensor([self.groups['seasonality']])
            periodic_kernel.lengthscale = torch.tensor([0.5])
            scale_periodic_kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)
            scale_periodic_kernel.outputscale = torch.tensor([1.5])

            # Cov Matrix
            cov = scale_rbf_kernel + scale_periodic_kernel
            covs.append(cov)

        return covs

    def _build_model(self, mean_function):
        covs = self._build_cov_matrices()
        n_changepoints = 4
        changepoints = np.linspace(0, self.groups['train']['n'], n_changepoints + 2)[1:-1]

        model_list = []
        likelihood_list = []
        for i in range(self.groups['train']['s']):
            likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
            if mean_function == 'piecewise_linear':
                model_list.append(ExactGPModel(self.train_x,
                                               self.train_y[:, i],
                                               likelihood_list[i],
                                               covs[i],
                                               PiecewiseLinearMean(changepoints)))
            elif mean_function == 'zero':
                model_list.append(ExactGPModel(self.train_x,
                                               self.train_y[:, i],
                                               likelihood_list[i],
                                               covs[i],
                                               gpytorch.means.ZeroMean()))
            elif mean_function == 'linear':
                model_list.append(ExactGPModel(self.train_x,
                                               self.train_y[:, i],
                                               likelihood_list[i],
                                               covs[i],
                                               LinearMean(1)))

        self.wall_time_build_model = time.time() - self.timer_start - self.wall_time_preprocess
        return likelihood_list, model_list

    def train(self, n_iterations=500, lr=1e-3, mean_function='piecewise_linear'):
        likelihood_list, model_list = self._build_model(mean_function)

        model = gpytorch.models.IndependentModelList(*model_list)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        mll = SumMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        for i in range(n_iterations):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss.item()))
            optimizer.step()

        self.wall_time_train = time.time() - self.timer_start - self.wall_time_build_model
        return model, likelihood

    def predict(self, model, likelihood):
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.arange(self.groups['predict']['n']).type(torch.DoubleTensor)
            predictions = likelihood(*model(*[test_x for i in range(self.groups['predict']['s'])]))

        i = 0
        samples = np.zeros((self.n_samples, self.groups['predict']['n'], self.groups['predict']['s']))
        for pred in predictions:
            samples[:, :, i] = np.random.normal(pred.mean.detach().numpy(), np.sqrt(pred.variance.detach().numpy()),
                                                size=(self.n_samples, self.groups['predict']['n']))
            i += 1

        samples = np.transpose(samples, (1, 2, 0))

        # transform back the data
        samples = ((samples * self.dt.std_data[np.newaxis, :, np.newaxis]) + self.dt.mu_data[np.newaxis, :, np.newaxis])
        self.groups = self.dt.inv_transf_train()

        # Clip predictions to 0 if there are negative numbers
        samples[samples < 0] = 0

        self.wall_time_predict = time.time() - self.timer_start - self.wall_time_train
        return samples

    def store_metrics(self, res):
        with open(f'{self.input_dir}results/results_gp_cov_{self.dataset}.pickle', 'wb') as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, samples):
        calc_results = CalculateResultsBottomUp(samples,
                                                self.groups,
                                                self.store_prediction_samples,
                                                self.store_prediction_points)
        res = calc_results.calculate_metrics()
        self.wall_time_total = time.time() - self.timer_start

        res['wall_time'] = {}
        res['wall_time']['wall_time_preprocess'] = self.wall_time_preprocess
        res['wall_time']['wall_time_build_model'] = self.wall_time_build_model
        res['wall_time']['wall_time_train'] = self.wall_time_train
        res['wall_time']['wall_time_predict'] = self.wall_time_predict
        res['wall_time']['wall_time_total'] = self.wall_time_total

        return res

