import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import numpy as np
from htsmodels.results.calculate_metrics import calculate_metrics
import pickle


class MinT:

    def __init__(self, dataset, groups):
        self.dataset = dataset
        self.groups = groups
        dict_groups = {k.capitalize(): np.tile(groups['train']['groups_names'][k][groups['train']['groups_idx'][k]],
                                               (groups['predict']['n'], 1)).T.reshape(-1, ) for k in
                       [k for k, v in groups['train']['groups_n'].items()]}
        dict_groups['Count'] = groups['predict']['data']
        dict_groups['Date'] = np.tile(np.array(groups['dates']), (groups['train']['s'],))
        self.df = pd.DataFrame(dict_groups)

    def train(self):
        r = robjects.r
        script_dir = os.path.dirname(__file__)
        rel_path = "prison_hts_function.R"
        abs_file_path = os.path.join(script_dir, rel_path)
        r['source'](abs_file_path)
        function_r = robjects.globalenv['prison_results']
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(self.df)
        df_result_r = function_r(df_r)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_result = ro.conversion.rpy2py(df_result_r)
        df_result[['.mean']] = df_result[['.mean']].astype('float')
        return df_result

    def results(self, pred_mint):
        groups_names = pred_mint.columns[:-3]
        for group in groups_names:
            pred_mint = pred_mint.loc[(pred_mint[group] != '<aggregated>')]
            sort_group = pd.unique(
                self.groups['train']['groups_names'][group.lower()][self.groups['train']['groups_idx'][group.lower()]])
            pred_mint[group] = pred_mint[group].astype("category")
            pred_mint[group].cat.set_categories(sort_group, inplace=True)

        pred_mint = pred_mint.sort_values([k.title() for k in self.groups['train']['groups_names']])

        pred_mint = pred_mint.reset_index().drop('index', axis=1)

        for group in groups_names:
            # Assert order is correct between original dataset and predictions
            np.testing.assert_array_equal(pd.unique(pred_mint[group]), pd.unique(
                self.groups['train']['groups_names'][group.lower()][self.groups['train']['groups_idx'][group.lower()]]))

        h = self.groups['h']
        s = self.groups['train']['s']
        n = self.groups['train']['n']

        pred = pred_mint['.mean'].to_numpy().reshape(s, h).T
        pred_complete = np.concatenate((np.zeros((n, s)), pred), axis=0)[np.newaxis, :, :]
        return pred_complete

    def store_metrics(self, res):
        with open(f'results_gp_cov_{self.dataset}.pickle', 'wb') as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, mean):
        res = calculate_metrics(mean, self.groups)
        return res
