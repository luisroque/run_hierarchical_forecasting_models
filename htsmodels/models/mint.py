import pickle
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
import psutil

import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

from htsmodels.results.calculate_metrics import CalculateResultsMint
from htsmodels.utils.logger import Logger
from htsmodels import __version__


class MinT:

    def __init__(self, dataset, groups, aggregate_key=None, input_dir='./', log_dir: str = "."):

        self.dataset = dataset
        self.groups = groups
        self.timer_start = time.time()
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        # Ensure that keys are capitalized for every dataset
        self.groups['train']['groups_names'] = self._get_capitalized_keys(self.groups['train']['groups_names'])
        self.groups['train']['groups_idx'] = self._get_capitalized_keys(self.groups['train']['groups_idx'])
        self.groups['train']['groups_n'] = self._get_capitalized_keys(self.groups['train']['groups_n'])
        self.input_dir = input_dir
        dict_groups = {k.capitalize(): np.tile(groups['train']['groups_names'][k][groups['train']['groups_idx'][k]],
                                               (groups['predict']['n'], 1)).T.reshape(-1, ) for k in
                       [k for k, v in groups['train']['groups_n'].items()]}
        groups['predict']['data_matrix'][:groups['train']['n'], :] = groups['train']['data']
        dict_groups['Count'] = groups['predict']['data_matrix'].T.reshape(-1,)
        dict_groups['Date'] = np.tile(np.array(groups['dates']), (groups['train']['s'],))
        self.df = pd.DataFrame(dict_groups)
        self.algorithm = 'mint'

        time_interval = (self.groups['dates'][1] - self.groups['dates'][0]).days
        if time_interval < 2:
            self.time_int = 'day'
        elif time_interval < 8:
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

        self.logger_train = Logger(
            "train", algorithm='mint', dataset=self.dataset, to_file=True, log_dir=log_dir
        )
        self.logger_predict = Logger(
            "predict", algorithm='mint', dataset=self.dataset, to_file=True, log_dir=log_dir
        )
        self.logger_metrics = Logger(
            "metrics", algorithm='mint', dataset=self.dataset, to_file=True, log_dir=log_dir
        )

        self.model_version = __version__

    def train(self, algorithm='ets', rec_method='mint', track_mem=True):
        """Train ETS or ARIMA with conventional bottom-up or MinT reconciliation strategies

        :param algorithm: ets or arima
        :param rec_method: reconciliation method (bottom_up, mint or base)
            base means that we just forecast all time series without any reconciliation
        :return: df with results
        """
        timer_start = time.time()
        robjects.r('''
            library('fpp3')
            results_fn <- function(df, time, h, string_aggregate, start_predict_date, algorithm, rec_method) {
              if (time == 'quarter') {
                fn = yearquarter
              } else if (time == 'month') {
                fn = yearmonth
              } else if (time == 'week') {
                fn = yearweek
              } else if (time == 'year') {
                fn = year
              } else if (time == 'day') {
                fn = ymd
              } 
              
              # Get the correct algorithm and assign the respective function
              if (algorithm == 'ets') {
                algo_fn = ETS
              } else if (algorithm == 'arima') {
                algo_fn = ARIMA
              }
              
              # Get the correct reconciliation method and assign the respective function
              if (rec_method == 'bottom_up') {
                rec_method_fn = bottom_up
              } else if (rec_method == 'mint') {
                rec_method_fn = min_trace
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
                  model(base = algo_fn(Count)) %>%
                  reconcile(
                    if (rec_method == 'mint') {
                        rec = rec_method_fn(base, method = 'mint_shrink')
                    } else if (rec_method == 'bottom_up') {
                        rec = rec_method_fn(base)
                    } else {
                    }
                  )
              fc <- fit %>% forecast(h = h)
              
              fc_csv = fc %>% 
                        # getting the 95% interval
                        hilo(level = 95) %>%
                        unpack_hilo('95%') %>% 
                        as_tibble %>% 
                        mutate(mean=mean(Count)) %>%
                        mutate(std=sqrt(distributional::variance(Count))) %>%
                        select(-Count) %>% 
                        rename(lower='95%_lower') %>%
                        rename(upper='95%_upper') %>%
                        select(-.mean) %>%
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
                                 self.last_train_date,
                                 algorithm,
                                 rec_method)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_result = ro.conversion.rpy2py(df_result_r)

        # When it runs the reconciliation it also runs the base algorithm, so
        # it has two different outputs stored in the df that we need to filter
        if not rec_method == 'base':
            df_result = df_result.loc[df_result['.model'] != 'base']

        # name properly the models in the dataframe
        df_result.loc[:, '.model'] = algorithm + '_' + rec_method

        df_result[['mean']] = df_result[['mean']].astype('float')
        df_result[['lower']] = df_result[['lower']].astype('float')
        df_result[['upper']] = df_result[['upper']].astype('float')
        df_result[['std']] = df_result[['std']].astype('float')

        if track_mem:
            # Track RAM usage
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024 ** 3)
            self.logger_train.info(f"train used {mem:.3f} GB of RAM")

        self.wall_time_train = time.time() - self.timer_start
        td = timedelta(seconds=int(time.time() - self.timer_start))
        self.logger_train.info(f"wall time train {str(td)}")
        self.wall_time_predict = 0

        return df_result

    @staticmethod
    def _get_capitalized_keys(dict_to_capitalize):
        indexes_dict = {}
        for k, v in dict_to_capitalize.items():
            # ensure that groups names are capitalized
            indexes_dict[k.capitalize()] = v
        return indexes_dict

    def results(self, pred_mint):
        cols = list(self.groups['train']['groups_names'].keys())

        if self.time_int == 'day':
            pred_mint['Date'] = pred_mint['time']
        elif self.time_int == 'week':
            pred_mint['Date'] = pred_mint['time'].apply(lambda x: datetime.strptime(x + str(0), '%Y W%W%w')
                                                        .strftime('%Y-%m-%d')).apply(pd.to_datetime)
        elif self.time_int == 'month':
            pred_mint['Date'] = pred_mint['time'].apply(lambda x: datetime.strptime(x, '%Y %b')
                                                        .strftime('%Y-%m-%d')).apply(pd.to_datetime)
        elif self.time_int == 'quarter':
            pred_mint['Date'] = pd.PeriodIndex(pred_mint['time'].str.replace(' ', '-'), freq='Q').to_timestamp()
        elif self.time_int == 'year':
            pred_mint['Date'] = pd.PeriodIndex(pred_mint['time'], freq='Y').to_timestamp()

        cols.append('Date')
        # Zip can sometimes have the dtype int and breaks
        pred_mint = pred_mint.astype({k: 'string' for k in cols})
        pred_mint['Date'] = pd.to_datetime(pred_mint['Date'])

        # Filter only the predictions
        pred_mint = pred_mint[pred_mint['Date'] > self.last_train_date]

        return pred_mint

    def store_metrics(self, res, track_mem=True):
        with open(f"{self.input_dir}results_gp_cov_{self.dataset}_{self.model_version}.pickle", 'wb') as handle:
            if track_mem:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger_metrics.info(
                    f"Storing error metrics used {mem:.3f} GB of RAM"
                )
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, df_results_mint, track_mem=True):
        calc_results = CalculateResultsMint(df_results_mint=df_results_mint,
                                            groups=self.groups,
                                            dataset=self.dataset)
        res = calc_results.calculate_metrics()
        if track_mem:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024**3)
            self.logger_metrics.info(
                f"calculating error metrics used {mem:.3f} GB of RAM"
            )

        self.wall_time_total = time.time() - self.timer_start

        res["wall_time"] = {}
        res["wall_time"]["wall_time_train"] = self.wall_time_train
        res["wall_time"]["wall_time_predict"] = self.wall_time_predict
        res["wall_time"]["wall_time_total"] = self.wall_time_total
        return res
