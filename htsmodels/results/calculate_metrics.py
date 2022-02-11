import numpy as np
from sklearn.metrics import mean_squared_error


class CalculateStoreResults:
    """Calculate the results and store them using pickle files

    Currently we have implemented MASE and RMSE.

    Attributes:
        pred_samples (array): predictions of shape [number of samples, h, number of series]
            - we transform it immediately to the shape [h, number of series] by averaging over the samples
        groups (obj): dict containing the data and several other attributes of the dataset

    """

    def __init__(self, pred_samples, groups):
        self.groups = groups
        self.seas = self.groups['seasonality']
        self.h = self.groups['h']
        self.n = self.groups['predict']['n']
        self.s = self.groups['predict']['s']
        self.pred_samples = np.mean(pred_samples, axis=0).reshape(self.n, self.s)[self.n - self.h:self.n, :]
        self.y_f = self.groups['predict']['data'].reshape(self.s, self.n).T
        self.errs = ['mase', 'rmse']
        self.levels = list(self.groups['predict']['groups_names'].keys())
        self.levels.extend(('bottom', 'total'))

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

    def calculate_metrics_for_individual_group(self, group_name, y, predictions, error_metrics):
        """Calculates the main metrics for each group

        Args:
            param group_name: group that we want to calculate the error metrics
            param y: original series values with the granularity of the group to calculate
            param predictions: predictions with the granularity of the group to calculate
            param error_metrics: dict to add new results

        Returns:
            error (obj): contains both the error metric for each individual series of each group and the average

        """
        error_metrics['mase'][f'{group_name}_ind'] = np.round(self.mase(y=y,
                                                                        f=predictions), 3)
        error_metrics['mase'][f'{group_name}'] = np.round(np.mean(error_metrics['mase'][f'{group_name}_ind']), 3)
        error_metrics['rmse'][f'{group_name}_ind'] = np.round(mean_squared_error(y[self.n - self.h:self.n, :],
                                                              predictions,
                                                              squared=False,
                                                              multioutput='raw_values'),
                                                              3)
        error_metrics['rmse'][f'{group_name}'] = np.round(np.mean(error_metrics['rmse'][f'{group_name}_ind']), 3)

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
        error_metrics['predictions'] = {}

        error_metrics = self.calculate_metrics_for_individual_group('bottom',
                                                                    self.y_f,
                                                                    self.pred_samples,
                                                                    error_metrics)
        error_metrics['predictions']['bottom'] = self.pred_samples
        error_metrics = self.calculate_metrics_for_individual_group('total',
                                                                    np.sum(self.y_f, axis=1).reshape(-1, 1),
                                                                    np.sum(self.pred_samples, axis=1).reshape(-1, 1),
                                                                    error_metrics)
        error_metrics['predictions']['total'] = np.sum(self.pred_samples, axis=1).reshape(-1, 1)
        # All the groups present in the data
        idx_dict_new = dict()
        for group in list(self.groups['predict']['groups_names'].keys()):
            y_g = np.zeros((self.groups['predict']['n'], self.groups['predict']['groups_names'][group].shape[0]))
            f_g = np.zeros((self.h, self.groups['predict']['groups_names'][group].shape[0]))

            for idx, name in enumerate(self.groups['predict']['groups_names'][group]):
                idx_dict_new[name] = np.where(self.groups['predict']['groups_idx'][group] == idx, 1, 0)

                y_g[:, idx] = np.sum(idx_dict_new[name] * self.y_f, axis=1)
                f_g[:, idx] = np.sum(idx_dict_new[name] * self.pred_samples, axis=1)
                error_metrics['predictions'][name] = np.sum(idx_dict_new[name] * self.pred_samples, axis=1)

            error_metrics = self.calculate_metrics_for_individual_group(group,
                                                                        y_g,
                                                                        f_g,
                                                                        error_metrics)
            error_metrics['predictions'][group] = np.sum(f_g, axis=1)
            list(self.groups['predict']['groups_names'].keys())
        for err in self.errs:
            error_metrics[err]['all_ind'] = np.concatenate([error_metrics[err][f'{x}_ind'] for x in
                                                            self.levels],
                                                           0)
            error_metrics[err]['all'] = np.mean(error_metrics[err]['all_ind'])

        return error_metrics


