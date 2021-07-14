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


year_regex = re.compile(r'^Y\d{4}$')
year_columns = [
    col for col in trade_crops_data.columns if re.match(year_regex, col)
]

word_regex = re.compile('\w+')
multi_word_items = {
    'fruit',
    'nuts',
    'oil, olive',
    'oil, palm',
    'other food',
    'potatoes', # listed here to avoid matching on sweet potatoes
    'sweet corn',
    'cotton',
    'cereal',
}
item_map = {
    word: {word} for word in (
        'beverages',
        'almonds',
        'flax',
        'apricots',
        'barley',
        'sunflower',
        'soybeans',
        'palm',
        'cashew',
        'peppers',
        'cocoa',
        'dates',
        'coconuts',
        'coffee',
        'eggs',
        'figs',
        'flour',
        'hazelnuts',
        'citrus',
        'lemon',
        'apple',
        'grape',
        'grapefruit',
        'groundnuts',
        'milk',
        'oats',
        'oilseeds',
        'orange',
        'pineapples',
        'rice',
        'roots',
        'rubber',
        'silk',
        'sheep',
        'sugar',
        'tea',
        'tobacco',
        'tomatoes',
        'vegetables',
        'watermelons',
        'wool',
        'peas',
        'offals',
        'mushrooms',
    )
}
item_map['chicken'] = {'chicken', 'poultry'}
item_map['beef'] = {'beef', 'cattle', 'bovine', 'whey'}
item_map['grape'] = {'grape', 'grapes'}
item_map['wheat'] = {'wheat', 'bread'}
item_map['pig'] = {'pig', 'bacon', 'pigmeat', 'pigs'}
item_map['pepper'] = {'pepper', 'peppers'}
def combine_item_names(item):
    item = item.lower()
    for multi_word in multi_word_items:
        if item.startswith(multi_word):
            return multi_word
    for word in re.findall(word_regex, item):
        for key, match_words in item_map.items():
            if word in match_words:
                return key
    return item

trade_crops_data['updated_item'] = trade_crops_data['Item'].apply(
    combine_item_names
)

brazil_exports = pd.DataFrame(columns=['item', 'year', 'tonnes_exported'])
for item, group in (
    trade_crops_data.loc[
        (trade_crops_data['Area'] == 'Brazil') &
        (trade_crops_data['Element'] == 'Export Quantity')
    ].groupby('updated_item')
):
    for col in year_columns:
        year = int(col.strip('Y'))
        value = group[col].sum(skipna=True)
        if not np.isnan(value):
            brazil_exports = brazil_exports.append(
                pd.DataFrame({
                    'item': [item],
                    'year': [year],
                    'tonnes_exported': [value],
                }),
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

total_deforestation_data.to_csv(
    f'{data_dir}/processed/brazil_deforested_area.csv',
)

min_year = total_deforestation_data['year'].min()
max_year = total_deforestation_data['year'].max()

brazil_exports = brazil_exports.loc[
    np.bitwise_and(
        np.bitwise_and(
            brazil_exports['year'] >= min_year,
            brazil_exports['year'] <= max_year,
        ),
        np.bitwise_not(np.isnan(brazil_exports['tonnes_exported'])),
    )
]

brazil_exports.to_csv(
    f'{data_dir}/processed/brazil_exports.csv',
)

# put everything for the plots in one df to use with vega-lite
brazil_corr_data['type'] = 'correlation'
brazil_corr_data.rename(
    columns={'item': 'label', 'corr': 'value'}, inplace=True,
)

total_deforestation_data['type'] = 'deforestation'
total_deforestation_data.rename(
    columns={'deforested_area_km2': 'value'}, inplace=True,
)

brazil_exports['type'] = 'deforestation'
brazil_exports.rename(
    columns={'item': 'label', 'tonnes_exported': 'value'}, inplace=True,
)

pd.concat(
    (
        brazil_corr_data,
        total_deforestation_data,
        brazil_exports,
    ),
).to_csv(
    f'{data_dir}/processed/plot_data.csv',
)
