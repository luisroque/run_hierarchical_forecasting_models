import numpy as np
from sklearn.metrics import mean_squared_error
import properscoring as ps


class CalculateResultsBase:
    """Calculate the results and store them using pickle files

    Currently we have implemented MASE and RMSE.

    Attributes:
        pred_samples (array): predictions of shape [number of samples, h, number of series]
            - we transform it immediately to the shape [h, number of series] by averaging over the samples
        groups (obj): dict containing the data and several other attributes of the dataset

    """

    def __init__(self, groups, samples=500):
        self.groups = groups
        self.seas = self.groups['seasonality']
        self.h = self.groups['h']
        self.n = self.groups['predict']['n']
        self.s = self.groups['predict']['s']
        self.y_f = self.groups['predict']['data'].reshape(self.s, self.n).T
        self.errs = ['mase', 'rmse', 'CRPS']
        self.levels = list(self.groups['train']['groups_names'].keys())
        self.levels.extend(('bottom', 'total'))
        self.samples = samples

    def mase(self, y, f):
        """Calculates the Mean Absolute Scaled Error

        Args:
            param y (array): original series values shape=[n,s]
            param f (array): predictions shape=[h,s]

        Returns:
            MASE (array): shape=[s,]

        """
        return ((self.n - self.seas) / self.h * (np.sum(np.abs(y[self.n - self.h:self.n, :] - f), axis=0)
                                                 / np.sum(np.abs(y[self.seas:self.n, :] - y[:self.n - self.seas, :]),
                                                          axis=0)))

    def calculate_metrics_for_individual_group(self,
                                               group_name,
                                               y,
                                               predictions_mean,
                                               error_metrics,
                                               predictions_sample=np.zeros((1,)),
                                               predictions_std=None):
        """Calculates the main metrics for each group

        Args:
            param group_name: group that we want to calculate the error metrics
            param y: original series values with the granularity of the group to calculate
            param predictions_mean: predictions mean with the granularity of the group to calculate
            param error_metrics: dict to add new results
            param predictions_sample: samples of the predictions
            param predictions_variance: variance of the predictions

        Returns:
            error (obj): contains both the error metric for each individual series of each group and the average

        """
        y_p = y[self.n - self.h:self.n, :]
        if predictions_sample.any():
            predictions_sample = predictions_sample[self.n - self.h:self.n, :, :]
        n_s = y_p.shape[1]
        error_metrics['mase'][f'{group_name}_ind'] = np.round(self.mase(y=y,
                                                                        f=predictions_mean), 3)
        error_metrics['mase'][f'{group_name}'] = np.round(np.mean(error_metrics['mase'][f'{group_name}_ind']), 3)
        error_metrics['rmse'][f'{group_name}_ind'] = np.round(mean_squared_error(y_p,
                                                              predictions_mean,
                                                              squared=False,
                                                              multioutput='raw_values'),
                                                              3)
        error_metrics['rmse'][f'{group_name}'] = np.round(np.mean(error_metrics['rmse'][f'{group_name}_ind']), 3)

        if predictions_std is not None:
            error_metrics['CRPS'][f'{group_name}'] = ps.crps_ensemble(y_p,
                                                                      np.random.normal(loc=predictions_mean,
                                                                                       scale=predictions_std,
                                                                                       size=(self.samples,
                                                                                             self.h,
                                                                                             n_s))
                                                                      .reshape((self.h,
                                                                                n_s,
                                                                                -1))).mean()
            error_metrics['CRPS'][f'{group_name}_ind'] = ps.crps_ensemble(y_p,
                                                                          np.random.normal(loc=predictions_mean,
                                                                                           scale=predictions_std,
                                                                                           size=(self.samples,
                                                                                                 self.h,
                                                                                                 n_s))
                                                                          .reshape((self.h,
                                                                                    n_s,
                                                                                    -1))).mean(axis=0)
        else:
            error_metrics['CRPS'][f'{group_name}'] = ps.crps_ensemble(y_p, predictions_sample).mean()
            error_metrics['CRPS'][f'{group_name}_ind'] = ps.crps_ensemble(y_p, predictions_sample).mean(axis=0)

        return error_metrics


class CalculateResultsBottomUp(CalculateResultsBase):
    r"""
    Calculate results for the bottom-up strategy.

    From the prediction of the bottom level series, aggregate the results for the upper levels
    considering the hierarchical structure and compute the error metrics accordingly.

    Parameters
    ----------
    pred_samples : numpy array
        results for the bottom series
    groups : dict
        all the information regarding the different groups
    """

    def __init__(self, pred_samples, groups, store_prediction_samples, store_prediction_points):
        super().__init__(groups=groups)
        self.pred_mean = np.mean(pred_samples, axis=2).reshape(self.n, self.s)[self.n - self.h:self.n, :]
        self.pred_mean_complete = np.mean(pred_samples, axis=2).reshape(self.n, self.s)
        self.pred_samples = pred_samples
        self.store_prediction_samples = store_prediction_samples
        self.store_prediction_points = store_prediction_points

    def compute_error_for_every_group(self, error_metrics):
        """Computes the error metrics for all the groups

        Returns:
            error (obj): - contains all the error metric for each group in the dataset
                         - contains all the predictions for all the groups

        """
        idx_dict_new = dict()
        for group in list(self.groups['predict']['groups_names'].keys()):
            y_g = np.zeros((self.groups['predict']['n'], self.groups['predict']['groups_names'][group].shape[0]))
            f_g = np.zeros((self.h, self.groups['predict']['groups_names'][group].shape[0]))
            s_g = np.zeros((self.n, self.groups['predict']['groups_names'][group].shape[0], self.pred_samples.shape[2]))

            for idx, name in enumerate(self.groups['predict']['groups_names'][group]):
                idx_dict_new[name] = np.where(self.groups['predict']['groups_idx'][group] == idx, 1, 0).reshape((1, -1))

                y_g[:, idx] = np.sum(idx_dict_new[name] * self.y_f, axis=1)
                f_g[:, idx] = np.sum(idx_dict_new[name] * self.pred_mean, axis=1)
                s_g[:, idx, :] = np.sum(idx_dict_new[name][:, :, np.newaxis] * self.pred_samples, axis=1)
                if self.store_prediction_points:
                    error_metrics['predictions']['points'][name] = f_g[:, idx]
                if self.store_prediction_samples:
                    error_metrics['predictions']['samples'][name] = s_g[:, idx, :]

            error_metrics = self.calculate_metrics_for_individual_group(group,
                                                                        y_g,
                                                                        f_g,
                                                                        error_metrics,
                                                                        predictions_sample=s_g)

        return error_metrics

    def bottom_up(self, level, error_metrics):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for the specific level

        """
        if level == 'bottom':
            error_metrics = self.calculate_metrics_for_individual_group(level,
                                                                        self.y_f,
                                                                        self.pred_mean,
                                                                        error_metrics,
                                                                        predictions_sample=self.pred_samples)
            if self.store_prediction_points:
                error_metrics['predictions']['points']['bottom'] = self.pred_mean
            if self.store_prediction_samples:
                error_metrics['predictions']['samples']['bottom'] = self.pred_samples
        elif level == 'total':
            error_metrics = self.calculate_metrics_for_individual_group(level,
                                                                        np.sum(self.y_f, axis=1).reshape(-1, 1),
                                                                        np.sum(self.pred_mean, axis=1).reshape(-1, 1),
                                                                        error_metrics,
                                                                        predictions_sample=np.sum(self.pred_samples,
                                                                                                  axis=1)
                                                                        .reshape((self.n, 1, -1)))

            if self.store_prediction_points:
                error_metrics['predictions']['points']['total'] = np.sum(self.pred_mean, axis=1)
            if self.store_prediction_samples:
                error_metrics['predictions']['samples']['total'] = np.sum(self.pred_samples, axis=1)
        elif level == 'groups':
            self.compute_error_for_every_group(error_metrics)

        return error_metrics

    def calculate_metrics(self):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for each individual series of each group and the average
                         - contains all the predictions for all the series and groups

        """
        error_metrics = dict()
        error_metrics['mase'] = {}
        error_metrics['rmse'] = {}
        error_metrics['CRPS'] = {}
        if self.store_prediction_points or self.store_prediction_samples:
            error_metrics['predictions'] = {}
        if self.store_prediction_points:
            error_metrics['predictions']['points'] = {}
        if self.store_prediction_samples:
            error_metrics['predictions']['samples'] = {}

        error_metrics = self.bottom_up('bottom', error_metrics)
        error_metrics = self.bottom_up('total', error_metrics)
        error_metrics = self.bottom_up('groups', error_metrics)

        # Aggregate all errors and create the 'all' category
        for err in self.errs:
            error_metrics[err]['all_ind'] = np.squeeze(np.concatenate([error_metrics[err][f'{x}_ind'].reshape((-1, 1))
                                                                       for x in self.levels], 0))
            error_metrics[err]['all'] = np.mean(error_metrics[err]['all_ind'])

        return error_metrics


class CalculateResultsMint(CalculateResultsBase):
    r"""
    Calculate results for MinT reconciliation.

    The usage is nearly identical to the bottom-up reconciliation strategy, but the implementation
    is a bit different, since the MinT algorithm yields results for all levels in the hierarchy.
    We just need to filter those to get the mean, lower and upper values for each level.

    Parameters
    ----------
    df_results_mint : pandas
        df with the results
    groups : dict
        dictionary with all the information regarding the different groups
    """
    def __init__(self, df_results_mint, groups, store_prediction_samples, store_prediction_points):
        super().__init__(groups=groups)
        self.df_results_mint = df_results_mint
        self.store_prediction_samples = store_prediction_samples
        self.store_prediction_points = store_prediction_points

    def compute_error_for_every_group(self, error_metrics):
        """Computes the error metrics for all the groups

        Returns:
            error (obj): - contains all the error metric for each group in the dataset
                         - contains all the predictions for all the groups

        """
        idx_dict_new = dict()
        for group in list(self.groups['train']['groups_names'].keys()):
            y_g = np.zeros((self.groups['predict']['n'], self.groups['train']['groups_names'][group].shape[0]))
            f_g = np.zeros((self.h, self.groups['train']['groups_names'][group].shape[0]))
            std_g = np.zeros((self.h, self.groups['train']['groups_names'][group].shape[0]))

            for idx, name in enumerate(self.groups['train']['groups_names'][group]):
                idx_dict_new[name] = np.where(self.groups['train']['groups_idx'][group] == idx, 1, 0)

                y_g[:, idx] = np.sum(idx_dict_new[name] * self.y_f, axis=1)
                pred_samples_group = self.df_results_mint.loc[self.df_results_mint[group].isin([name])]\
                    .drop([group], axis=1)
                f_g[:, idx] = np.asarray(pred_samples_group.loc[pred_samples_group
                                         .iloc[:, :self.groups['train']['g_number']-1]
                                         .isin(['<aggregated>']).all(axis=1)]['mean']).reshape((self.h,))
                std_g[:, idx] = np.asarray(pred_samples_group.loc[pred_samples_group
                                           .iloc[:, :self.groups['train']['g_number']-1]
                                           .isin(['<aggregated>']).all(axis=1)]['std']).reshape((self.h,))
                if self.store_prediction_points:
                    error_metrics['predictions']['points'][name] = f_g
                if self.store_prediction_samples:
                    error_metrics['predictions']['samples'][name] = np.random.normal(loc=f_g,
                                                                                     scale=std_g,
                                                                                     size=(self.samples,
                                                                                           self.h,
                                                                                           self.s)).reshape((self.h,
                                                                                                             self.s,
                                                                                                             -1))

            error_metrics = self.calculate_metrics_for_individual_group(group,
                                                                        y_g,
                                                                        f_g,
                                                                        error_metrics,
                                                                        predictions_std=std_g)
            if self.store_prediction_points:
                error_metrics['predictions']['points'][group] = f_g
            if self.store_prediction_samples:
                error_metrics['predictions']['samples'][group] = np.random.normal(loc=f_g,
                                                                                  scale=std_g,
                                                                                  size=(self.samples,
                                                                                        self.h,
                                                                                        self.s)).reshape((self.h,
                                                                                                          self.s,
                                                                                                          -1))
        return error_metrics

    def mint_reconciliation(self, level, error_metrics):
        """Get the results for each of the level of aggregation using the MinT strategy (bottom-up not needed)

        Returns:
            error (obj): - contains all the error metric for the specific level

        """
        if level == 'bottom':
            n_s = self.s
            pred_samples_mean = np.asarray(self.df_results_mint.loc[~self.df_results_mint.isin(['<aggregated>'])
                                                                    .any(axis=1)]['mean']).reshape((self.h, n_s))
            pred_samples_std = np.asarray(self.df_results_mint.loc[~self.df_results_mint.isin(['<aggregated>'])
                                                                   .any(axis=1)]['std']).reshape((self.h, n_s))
            error_metrics = self.calculate_metrics_for_individual_group(level,
                                                                        self.y_f,
                                                                        pred_samples_mean,
                                                                        error_metrics,
                                                                        predictions_std=pred_samples_std)
            if self.store_prediction_points:
                error_metrics['predictions']['points']['bottom'] = pred_samples_mean
            if self.store_prediction_samples:
                error_metrics['predictions']['samples']['bottom'] = np.random.normal(loc=pred_samples_mean,
                                                                                     scale=pred_samples_std,
                                                                                     size=(self.samples,
                                                                                           self.h,
                                                                                           self.s)).reshape((self.h,
                                                                                                             n_s,
                                                                                                             -1))
        elif level == 'total':
            n_s = 1
            pred_samples_mean = np.asarray(self.df_results_mint.loc[self.df_results_mint
                                           .iloc[:, :self.groups['train']['g_number']]
                                           .isin(['<aggregated>']).all(axis=1)]['mean']).reshape((self.h, n_s))
            pred_samples_std = np.asarray(self.df_results_mint.loc[self.df_results_mint
                                          .iloc[:, :self.groups['train']['g_number']]
                                          .isin(['<aggregated>']).all(axis=1)]['std']).reshape((self.h, n_s))

            error_metrics = self.calculate_metrics_for_individual_group(level,
                                                                        np.sum(self.y_f, axis=1).reshape(-1, 1),
                                                                        pred_samples_mean,
                                                                        error_metrics,
                                                                        predictions_std=pred_samples_std)
            if self.store_prediction_points:
                error_metrics['predictions']['points']['total'] = np.squeeze(pred_samples_mean)
            if self.store_prediction_samples:
                error_metrics['predictions']['samples']['total'] = np.random.normal(loc=pred_samples_mean,
                                                                                    scale=pred_samples_std,
                                                                                    size=(self.samples,
                                                                                          self.h,
                                                                                          n_s)).reshape((self.h,
                                                                                                         -1))
        elif level == 'groups':
            self.compute_error_for_every_group(error_metrics)

        return error_metrics

    def calculate_metrics(self):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for each individual series of each group and the average
                         - contains all the predictions for all the series and groups

        """
        error_metrics = dict()
        error_metrics['mase'] = {}
        error_metrics['rmse'] = {}
        error_metrics['CRPS'] = {}
        if self.store_prediction_points or self.store_prediction_samples:
            error_metrics['predictions'] = {}
        if self.store_prediction_points:
            error_metrics['predictions']['points'] = {}
        if self.store_prediction_samples:
            error_metrics['predictions']['samples'] = {}

        error_metrics = self.mint_reconciliation('bottom', error_metrics)
        error_metrics = self.mint_reconciliation('total', error_metrics)
        error_metrics = self.mint_reconciliation('groups', error_metrics)

        # Aggregate all errors and create the 'all' category
        for err in self.errs:
            error_metrics[err]['all_ind'] = np.squeeze(np.concatenate([error_metrics[err][f'{x}_ind'].reshape((-1, 1))
                                                                       for x in self.levels], 0))
            error_metrics[err]['all'] = np.mean(error_metrics[err]['all_ind'])

        return error_metrics
