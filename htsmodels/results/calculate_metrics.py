import numpy as np
from sklearn.metrics import mean_squared_error
import properscoring as ps
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


class CalculateResultsBase:
    """Calculate the results and store them using pickle files

    Currently we have implemented MASE and RMSE.

    Attributes:
        pred_samples (array): predictions of shape [number of samples, h, number of series]
            - we transform it immediately to the shape [h, number of series] by averaging over the samples
        groups (obj): dict containing the data and several other attributes of the dataset

    """

    def __init__(self, groups, dataset):
        self.groups = groups
        self.seas = self.groups["seasonality"]
        self.h = self.groups["h"]
        self.n = self.groups["predict"]["n"]
        self.s = self.groups["predict"]["s"]
        self.y_f = self.groups["predict"]["data"].reshape(self.s, self.n).T
        self.errs = ["mase", "rmse", "CRPS"]
        self.levels = list(self.groups["train"]["groups_names"].keys())
        self.levels.extend(("bottom", "total"))
        self.dataset = dataset
        self.mase = MeanAbsoluteScaledError(multioutput="raw_values")

    def calculate_metrics_for_individual_group(
        self, group_name, y, predictions_mean, predictions_std, error_metrics
    ):
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

        y_true = y[-self.h :, :]
        y_train = y[: -self.h, :]
        f = predictions_mean[-self.h :]
        f_std = predictions_std[-self.h :]
        error_metrics["mase"][f"{group_name}_ind"] = np.round(
            self.mase(y_true=y_true, y_pred=f, y_train=y_train, sp=self.seas), 3
        )
        error_metrics["mase"][f"{group_name}"] = np.round(
            np.mean(error_metrics["mase"][f"{group_name}_ind"]), 3
        )
        error_metrics["rmse"][f"{group_name}_ind"] = np.round(
            mean_squared_error(y_true, f, squared=False, multioutput="raw_values"),
            3,
        )
        error_metrics["rmse"][f"{group_name}"] = np.round(
            np.mean(error_metrics["rmse"][f"{group_name}_ind"]), 3
        )

        error_metrics["CRPS"][f"{group_name}"] = ps.crps_gaussian(
            y_true, f, f_std
        ).mean()
        error_metrics["CRPS"][f"{group_name}_ind"] = ps.crps_gaussian(
            y_true, f, f_std
        ).mean(axis=0)

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

    def __init__(self, groups, dataset, predictions_mean, predictions_std):
        super().__init__(groups=groups, dataset=dataset)
        self.groups_names = list(self.groups["predict"]["groups_names"].keys())
        self.n_samples = groups["predict"]["n"]
        self.predictions_mean = predictions_mean
        self.predictions_std = predictions_std

    def compute_error_for_every_group(self, error_metrics):
        """Computes the error metrics for all the groups

        Returns:
            error (obj): - contains all the error metric for each group in the dataset
                         - contains all the predictions for all the groups

        """
        group_element_active = dict()
        for group in list(self.groups["predict"]["groups_names"].keys()):
            n_elements_group = self.groups["predict"]["groups_names"][group].shape[0]
            group_elements = self.groups["predict"]["groups_names"][group]
            groups_idx = self.groups["predict"]["groups_idx"][group]

            y_g = np.zeros((self.n_samples, n_elements_group))
            mean_g = np.zeros((self.n_samples, n_elements_group))
            std_g = np.zeros((self.n_samples, n_elements_group))

            for group_idx, element_name in enumerate(group_elements):
                group_element_active[element_name] = np.where(
                    groups_idx == group_idx, 1, 0
                ).reshape((1, -1))

                y_g[:, group_idx] = np.sum(
                    group_element_active[element_name] * self.y_f, axis=1
                )
                mean_g[:, group_idx] = np.sum(
                    group_element_active[element_name] * self.predictions_mean, axis=1
                )
                std_g[:, group_idx] = np.sum(
                    group_element_active[element_name] * self.predictions_mean, axis=1
                )

            error_metrics = self.calculate_metrics_for_individual_group(
                group_name=group,
                y=y_g,
                predictions_mean=mean_g,
                predictions_std=std_g,
                error_metrics=error_metrics,
            )
        return error_metrics

    def bottom_up(self, level, error_metrics):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for the specific level

        """
        if level == "bottom":
            error_metrics = self.calculate_metrics_for_individual_group(
                group_name=level,
                y=self.y_f,
                predictions_mean=self.predictions_mean,
                predictions_std=self.predictions_std,
                error_metrics=error_metrics,
            )
        elif level == "total":
            np.sqrt(np.sum(self.predictions_std**2, axis=1)).reshape(-1, 1)
            error_metrics = self.calculate_metrics_for_individual_group(
                group_name=level,
                y=np.sum(self.y_f, axis=1).reshape(-1, 1),
                predictions_mean=np.sum(self.predictions_mean, axis=1).reshape(-1, 1),
                # The variance of the resulting distribution will be the sum
                # of the variances of the original Gaussian distributions
                predictions_std=np.sqrt(
                    np.sum(self.predictions_std**2, axis=1)
                ).reshape(-1, 1),
                error_metrics=error_metrics,
            )
        elif level == "groups":
            self.compute_error_for_every_group(error_metrics)

        return error_metrics

    def calculate_metrics(self):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for each individual series of each group and the average
                         - contains all the predictions for all the series and groups

        """
        error_metrics = dict()
        error_metrics["mase"] = {}
        error_metrics["rmse"] = {}
        error_metrics["CRPS"] = {}

        error_metrics = self.bottom_up("bottom", error_metrics)
        error_metrics = self.bottom_up("total", error_metrics)
        error_metrics = self.bottom_up("groups", error_metrics)

        # Aggregate all errors and create the 'all' category
        for err in self.errs:
            error_metrics[err]["all_ind"] = np.squeeze(
                np.concatenate(
                    [
                        error_metrics[err][f"{x}_ind"].reshape((-1, 1))
                        for x in self.levels
                    ],
                    0,
                )
            )
            error_metrics[err]["all"] = np.mean(error_metrics[err]["all_ind"])

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

    def __init__(self, df_results_mint, groups, dataset):
        super().__init__(groups=groups, dataset=dataset)
        self.df_results_mint = df_results_mint

    def compute_error_for_every_group(self, error_metrics):
        """Computes the error metrics for all the groups

        Returns:
            error (obj): - contains all the error metric for each group in the dataset
                         - contains all the predictions for all the groups

        """
        idx_dict_new = dict()
        for group in list(self.groups["train"]["groups_names"].keys()):
            y_g = np.zeros(
                (
                    self.groups["predict"]["n"],
                    self.groups["train"]["groups_names"][group].shape[0],
                )
            )
            f_g = np.zeros(
                (self.h, self.groups["train"]["groups_names"][group].shape[0])
            )
            std_g = np.zeros(
                (self.h, self.groups["train"]["groups_names"][group].shape[0])
            )

            for idx, name in enumerate(self.groups["train"]["groups_names"][group]):
                idx_dict_new[name] = np.where(
                    self.groups["train"]["groups_idx"][group] == idx, 1, 0
                )

                y_g[:, idx] = np.sum(idx_dict_new[name] * self.y_f, axis=1)
                pred_samples_group = self.df_results_mint.loc[
                    self.df_results_mint[group].isin([name])
                ].drop([group], axis=1)
                f_g[:, idx] = np.asarray(
                    pred_samples_group.loc[
                        pred_samples_group.iloc[
                            :, : self.groups["train"]["g_number"] - 1
                        ]
                        .isin(["<aggregated>"])
                        .all(axis=1)
                    ]["mean"]
                ).reshape((self.h,))
                std_g[:, idx] = np.asarray(
                    pred_samples_group.loc[
                        pred_samples_group.iloc[
                            :, : self.groups["train"]["g_number"] - 1
                        ]
                        .isin(["<aggregated>"])
                        .all(axis=1)
                    ]["std"]
                ).reshape((self.h,))

            error_metrics = self.calculate_metrics_for_individual_group(
                group, y_g, f_g, std_g, error_metrics
            )

        return error_metrics

    def mint_reconciliation(self, level, error_metrics):
        """Get the results for each of the level of aggregation using the MinT strategy (bottom-up not needed)

        Returns:
            error (obj): - contains all the error metric for the specific level

        """
        if level == "bottom":
            n_s = self.s
            pred_samples_mean = (
                np.asarray(
                    self.df_results_mint.loc[
                        ~self.df_results_mint.isin(["<aggregated>"]).any(axis=1)
                    ]["mean"]
                )
                .reshape((n_s, self.h))
                .T
            )
            pred_samples_std = (
                np.asarray(
                    self.df_results_mint.loc[
                        ~self.df_results_mint.isin(["<aggregated>"]).any(axis=1)
                    ]["std"]
                )
                .reshape((n_s, self.h))
                .T
            )
            error_metrics = self.calculate_metrics_for_individual_group(
                level,
                self.y_f,
                pred_samples_mean,
                pred_samples_std,
                error_metrics,
            )

        elif level == "total":
            n_s = 1
            pred_samples_mean = (
                np.asarray(
                    self.df_results_mint.loc[
                        self.df_results_mint.iloc[:, : self.groups["train"]["g_number"]]
                        .isin(["<aggregated>"])
                        .all(axis=1)
                    ]["mean"]
                )
                .reshape((n_s, self.h))
                .T
            )
            pred_samples_std = (
                np.asarray(
                    self.df_results_mint.loc[
                        self.df_results_mint.iloc[:, : self.groups["train"]["g_number"]]
                        .isin(["<aggregated>"])
                        .all(axis=1)
                    ]["std"]
                )
                .reshape((n_s, self.h))
                .T
            )

            error_metrics = self.calculate_metrics_for_individual_group(
                level,
                np.sum(self.y_f, axis=1).reshape(-1, 1),
                pred_samples_mean,
                pred_samples_std,
                error_metrics,
            )
        elif level == "groups":
            self.compute_error_for_every_group(error_metrics)

        return error_metrics

    def calculate_metrics(self):
        """Aggregates the results for all the groups

        Returns:
            error (obj): - contains all the error metric for each individual series of each group and the average
                         - contains all the predictions for all the series and groups

        """
        error_metrics = dict()
        error_metrics["mase"] = {}
        error_metrics["rmse"] = {}
        error_metrics["CRPS"] = {}

        error_metrics = self.mint_reconciliation("bottom", error_metrics)
        error_metrics = self.mint_reconciliation("total", error_metrics)
        error_metrics = self.mint_reconciliation("groups", error_metrics)

        # Aggregate all errors and create the 'all' category
        for err in self.errs:
            error_metrics[err]["all_ind"] = np.squeeze(
                np.concatenate(
                    [
                        error_metrics[err][f"{x}_ind"].reshape((-1, 1))
                        for x in self.levels
                    ],
                    0,
                )
            )
            error_metrics[err]["all"] = np.mean(error_metrics[err]["all_ind"])

        return error_metrics
