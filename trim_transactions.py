import pandas as pd
import os

'''
As streamlit file size limit is 200 MB, please trim transaction table based on this code
('week' feature in causal table only starts from 43, hence why >42 is used here)
'''

base_dir = '..'

df = pd.read_csv(os.path.join(base_dir, 'dh_transactions.csv'))
df[df['week']>42].to_csv(os.path.join(base_dir, 'dh_transactions_trimmed.csv'), index=False)