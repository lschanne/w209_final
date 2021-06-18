#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import re

data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

# https://www.kaggle.com/mbogernetto/brazilian-amazon-rainforest-degradation?select=def_area_2004_2019.csv
def_area_data = pd.read_csv(f'{data_dir}/raw/def_area_2004_2019.csv')

# http://www.fao.org/faostat/en/#data/TP/metadata
# http://fenixservices.fao.org/faostat/static/bulkdownloads/Trade_Crops_Livestock_E_All_Data.zip
trade_crops_data = pd.read_csv(
    f'{data_dir}/raw/Trade_Crops_Livestock_E_All_Data/'
    f'Trade_Crops_Livestock_E_All_Data.csv'
)


regex = re.compile(r'^Y\d{4}$')
year_columns = []
new_columns = []
for iii, col_name in enumerate(trade_crops_data.columns):
    if re.match(regex, col_name):
        col_name = int(col_name.strip('Y'))
        year_columns.append(col_name)
    
    new_columns.append(col_name)

trade_crops_data.columns = new_columns

brazil_exports = pd.DataFrame(columns=['item', 'year', 'tonnes_exported'])
for item, group in (
    trade_crops_data.loc[
        (trade_crops_data['Area'] == 'Brazil') &
        (trade_crops_data['Element'] == 'Export Quantity')
    ].groupby('Item')
):
    item_data = pd.melt(
        group,
        value_vars=year_columns,
        var_name='year',
        value_name='tonnes_exported',
    )
    item_data['item'] = item
    brazil_exports = brazil_exports.append(
        item_data.loc[item_data['year'].notna()],
        ignore_index=True,
    )

total_deforestation_data = pd.DataFrame(
    {
        'year': def_area_data.iloc[:, 0],
        'deforested_area_km2': def_area_data.iloc[:, 1:-1].sum(axis=1),
    }
)

brazil_joined_data = brazil_exports.merge(
    total_deforestation_data,
    on='year',
    how='left',
)

brazil_corr_data = pd.DataFrame(columns=['item', 'n', 'corr'])
for item, group in brazil_joined_data.groupby('item'):
    group = group.loc[
        group['tonnes_exported'].notna() & group['deforested_area_km2'].notna()
    ]
    n = group.shape[0]
    correlation = group['tonnes_exported'].corr(group['deforested_area_km2'])
    if not np.isnan(correlation) and n >= 8:
        brazil_corr_data = brazil_corr_data.append(
            pd.DataFrame({'item': [item], 'n': [n], 'corr': [correlation]}),
            ignore_index=True,
        )

brazil_joined_data.to_csv(
    f'{data_dir}/processed/brazil_joined_data.csv',
)
brazil_corr_data.to_csv(
    f'{data_dir}/processed/brazil_corr_data.csv',
)
