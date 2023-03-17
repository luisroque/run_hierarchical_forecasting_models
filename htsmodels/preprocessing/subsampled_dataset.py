import numpy as np
import pandas as pd


def update_missing_values(groups: dict):
    """
    Update the missing values in the 'data', 'full_data', 'data_matrix', and 'dates' arrays with np.nan

    Args:
        groups: dictionary containing the data and metadata
    """
    for key in ['train', 'predict']:
        x_values = groups[key]['x_values']
        data = groups[key]['data']
        s = groups[key]['s']

        # Change the data type of the arrays to float
        data = data.astype(float)

        # Identify the missing indices in x_values
        missing_indices = sorted(set(range(max(x_values) + 1)) - set(x_values))

        # Update the 'data' array with np.nan for the missing values
        for missing_idx in missing_indices:
            insert_idx = missing_idx * s
            data = np.insert(data, insert_idx, [np.nan] * s)

        # Update the 'n' value in the metadata
        groups[key]['n'] = len(x_values) + len(missing_indices)
        groups[key]['data'] = data

        if key == 'train':
            full_data = groups[key]['full_data']
            full_data = full_data.astype(float)

            # Update the 'full_data' array with np.nan for the missing values
            for missing_idx in missing_indices:
                insert_idx = missing_idx * s
                full_data = np.insert(full_data, insert_idx, [np.nan] * s)

            groups[key]['full_data'] = full_data

        if key == 'predict':
            data_matrix = groups[key]['data_matrix']
            data_matrix = data_matrix.astype(float)

            # Update the 'data_matrix' array with np.nan for the missing values
            for missing_idx in missing_indices:
                insert_idx = missing_idx
                if insert_idx < data_matrix.shape[0]:
                    data_matrix = np.insert(data_matrix, insert_idx, [np.nan] * s, axis=0)

            groups[key]['data_matrix'] = data_matrix

        # Update the 'dates' array to fill in missing dates
        dates = groups['dates']
        start_date = dates[0]
        end_date = dates[-1]
        all_dates = pd.date_range(start_date, end_date, freq='QS')
        missing_dates = all_dates.difference(pd.to_datetime(dates))

        for missing_date in missing_dates:
            insert_idx = all_dates.get_loc(missing_date)
            dates.insert(insert_idx, missing_date)

        groups['dates'] = dates

    return groups

