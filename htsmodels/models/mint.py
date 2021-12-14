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

    def __init__(self, dataset, groups, aggregate_key=None):
        self.dataset = dataset
        self.groups = groups
        dict_groups = {k.capitalize(): np.tile(groups['train']['groups_names'][k][groups['train']['groups_idx'][k]],
                                               (groups['predict']['n'], 1)).T.reshape(-1, ) for k in
                       [k for k, v in groups['train']['groups_n'].items()]}
        groups['predict']['data_matrix'][:groups['train']['n'],:] = groups['train']['data']
        dict_groups['Count'] = groups['predict']['data_matrix'].T.reshape(-1,)
        dict_groups['Date'] = np.tile(np.array(groups['dates']), (groups['train']['s'],))
        self.df = pd.DataFrame(dict_groups)

        time_interval = (self.groups['dates'][1] - self.groups['dates'][0]).days
        if time_interval < 8:
            self.time_int = 'week'
        elif time_interval < 32:
            self.time_int = 'month'
        elif time_interval < 93:
            self.time_int = 'quarter'
        elif time_interval < 367:
            self.time_int = 'year'

        self.last_train_date = (self.groups['dates'][self.groups['train']['n']-1]).strftime("%Y-%m-%d")
        if aggregate_key:
            self.aggregate_key = aggregate_key
        else:
            # Default is to assume that there is a group structure (e.g. State * Gender)
            # and no direct hierarchy (e.g. State / Region)
            self.aggregate_key = ' * '.join([k.capitalize() for k in self.groups['train']['groups_names']])

    def train(self):
        robjects.r('''
            library('fpp3')
            results_fn <- function(df, time, h, string_aggregate, start_predict_date) {
              if (time == 'quarter') {
                fn = yearquarter
              } else if (time == 'month') {
                fn = yearmonth
              } else if (time == 'week') {
                fn = yearweek
              } else if (time == 'year') {
                fn = year
              } 
              data <- df %>%
                mutate(Time = fn(Date)) %>%
                select(-Date) %>%
                as_tsibble(key = colnames(df)[2:length(colnames(df))-2], index = Time) %>%
                relocate(Time)
              
              data_gts <- data %>%
                aggregate_key(.spec = !!rlang::parse_expr(string_aggregate), Count = sum(Count))
              
              fit <- data_gts %>%
                filter(Time <= fn(as.Date(start_predict_date))) %>%
                model(base = ETS(Count)) %>%
                reconcile(
                  bottom_up = bottom_up(base),
                  MinT = min_trace(base, method = "mint_shrink")
                )
              fc <- fit %>% forecast(h = h)
              
              fc_csv = fc %>% 
                as_tibble %>% 
                filter(.model=='MinT') %>% 
                select(-Count) %>% 
                mutate(.mean=.mean) %>%
                mutate(.mean=(sprintf("%0.2f", .mean))) %>%
                rename(time=Time) %>%
                lapply(as.character) %>% 
                data.frame(stringsAsFactors=FALSE)
              
              return (fc_csv)
            }
        ''')
        function_r = robjects.globalenv['results_fn']
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(self.df)

        df_result_r = function_r(df_r,
                                 self.time_int,
                                 self.groups['h'],
                                 self.aggregate_key,
                                 self.last_train_date)
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
