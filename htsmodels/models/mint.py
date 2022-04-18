import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import numpy as np
from htsmodels.results.calculate_metrics import CalculateResultsMint
import pickle
from pathlib import Path
from datetime import datetime
import time


class MinT:

    def __init__(self, dataset, groups, aggregate_key=None, input_dir='./'):
        self.dataset = dataset
        self.groups = groups
        self.timer_start = time.time()
        self.wall_time_preprocess = None
        self.wall_time_build_model = None
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        # Ensure that keys are capitalized for every dataset
        self.groups['train']['groups_names'] = self._get_capitalized_keys(self.groups['train']['groups_names'])
        self.groups['train']['groups_idx'] = self._get_capitalized_keys(self.groups['train']['groups_idx'])
        self.groups['train']['groups_n'] = self._get_capitalized_keys(self.groups['train']['groups_n'])
        self.input_dir = input_dir
        self._create_directories()
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

    def _create_directories(self):
        # Create directory to store results if does not exist
        Path(f'{self.input_dir}results').mkdir(parents=True, exist_ok=True)

    def train(self, algorithm='ets', rec_method='mint'):
        """Train ETS or ARIMA with conventional bottom-up or MinT reconciliation strategies

        :param algorithm: ETS or ARIMA
        :param rec_method: reconciliation method (bottom_up or mint)
        :return: df with results
        """
        self.wall_time_preprocess = time.time() - self.timer_start
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
              } else if (time == 'day') {
                fn = ymd
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
                  model(base = ETS(Count),
                        base_arima = ARIMA(Count)) %>%
                  reconcile(
                    bottom_up_ets = bottom_up(base),
                    mint_ets = min_trace(base, method = "mint_shrink"),
                    bottom_up_arima = bottom_up(base_arima),
                    mint_arima = min_trace(base_arima, method = "mint_shrink")
                  )
              fc <- fit %>% forecast(h = h)
              
              fc_csv = fc %>% 
                # getting the 95% interval
                hilo(level = 95) %>%
                unpack_hilo('95%') %>% 
                as_tibble %>% 
                select(-Count) %>% 
                rename(lower='95%_lower') %>%
                rename(upper='95%_upper') %>%
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

        # filter the results to only contain the algorithm and reconciliation method chosen
        df_result = df_result[df_result['.model'] == rec_method+'_'+algorithm]

        df_result[['.mean']] = df_result[['.mean']].astype('float')
        df_result[['lower']] = df_result[['lower']].astype('float')
        df_result[['upper']] = df_result[['upper']].astype('float')
        self.wall_time_build_model = time.time() - self.timer_start - self.wall_time_preprocess
        self.wall_time_train = time.time() - self.timer_start - self.wall_time_build_model
        self.wall_time_predict = time.time() - self.timer_start - self.wall_time_train
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

    def store_metrics(self, res):
        with open(f'{self.input_dir}results/results_gp_cov_{self.dataset}.pickle', 'wb') as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, df_results_mint):
        calc_results = CalculateResultsMint(df_results_mint=df_results_mint,
                                            groups=self.groups)
        res = calc_results.calculate_metrics()
        self.wall_time_total = time.time() - self.timer_start

        res['mean'] = res
        #res['lower'] = lower
        #res['upper'] = upper
        res['wall_time'] = {}
        res['wall_time']['wall_time_preprocess'] = self.wall_time_preprocess
        res['wall_time']['wall_time_build_model'] = self.wall_time_build_model
        res['wall_time']['wall_time_train'] = self.wall_time_train
        res['wall_time']['wall_time_predict'] = self.wall_time_predict
        res['wall_time']['wall_time_total'] = self.wall_time_total
        return res
