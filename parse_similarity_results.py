import os
from pathlib import Path
import json

import pandas as pd
runs_dir = Path('cache')


paths_to_result_file = runs_dir.glob('**/eval.csv')
paths_to_result_file = sorted(paths_to_result_file, key=os.path.getmtime)
list_df = []
for i, path_to_csv_file in enumerate(paths_to_result_file):
    # if i > 3:
    #     break
    print(path_to_csv_file)
    exp_name = path_to_csv_file.parent.name
    model_name = path_to_csv_file.parent.parent.name
    df = pd.read_csv(path_to_csv_file)
    df = df.assign(exp_name=[exp_name], model=[model_name])
    list_df.append(df)
    if 'weighted' not in df.columns:
        df['weighted'] = (df['similarity'] + 1 - df['SDT norm']) / 2

df = pd.concat(list_df, axis=0)
column_names = ['model', 'exp_name', 'weighted', 'similarity', 'SDT norm', 'FDT', 'SDT', 'FDT norm']
df = df[column_names]
df.sort_values(by=['weighted'], inplace=True, ascending=False)
print(df)

# df.to_excel("results.xlsx")
writer = pd.ExcelWriter('results_similarity.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='all', index=False)
(max_row, max_col) = df.shape
wb = writer.book
worksheet = writer.sheets['all']
col_names = [{'header': col_name} for col_name in df.columns]
# add table with coordinates: first row, first col, last row, last col;
#  header names or formatting can be inserted into dict
worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
    'columns': col_names,
    # 'style' = option Format as table value and is case sensitive
    # (look at the exact name into Excel)
    'style': None
})
worksheet.autofit()
wb.close()

