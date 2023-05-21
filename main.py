import random
from functools import reduce

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f_oneway
from utils import *

st.set_page_config(layout='wide', page_title='test_20230511')
st.title('Test 20230511')

columns = {}

if not st.session_state.get('county', pd.DataFrame([[False]])).any(axis=None):
    st.session_state['county'] = pd.DataFrame(np.array([['Dummy']]), columns=['Geographic Area Name'])


def do_join_3(causal, transaction, product, drop_store=True):
    '''
    joins 3 main tables
    used by demographics and marketing campaign sections
    '''
    causal['upc_week_store'] = causal['upc'].astype(str) + causal['week'].astype(str).apply(lambda x: x.zfill(3)) + \
                               causal['store'].astype(str).apply(lambda x: x.zfill(4))
    causal = causal.drop(['upc', 'week', 'store'], axis=1)
    transaction = transaction[transaction['dollar_sales'] > 0.1]
    transaction['upc_week_store'] = transaction['upc'].astype(str) + transaction['week'].astype(str).apply(
        lambda x: x.zfill(3)) + transaction['store'].astype(str).apply(lambda x: x.zfill(4))
    transaction = transaction.drop(['store'] * int(drop_store) + ['time_of_transaction', 'basket', 'day', 'household'],
                                   axis=1)

    df = pd.merge(causal.drop('geography', axis=1), transaction, how='inner', left_on='upc_week_store',
                  right_on='upc_week_store').drop('upc_week_store', axis=1)
    df['unit_price'] = (df['dollar_sales'] / df['units']).apply(lambda x: round(x, 1))
    df = df.drop('dollar_sales', axis=1)
    df = pd.merge(
        df.drop('week', axis=1).groupby(
            ['store'] * int(not drop_store) + ['upc', 'feature_desc', 'display_desc', 'geography', 'coupon',
                                               'unit_price']).sum(),
        df.drop('units', axis=1).groupby(
            ['store'] * int(not drop_store) + ['upc', 'feature_desc', 'display_desc', 'geography', 'coupon',
                                               'unit_price']).count(),
        left_index=True, right_index=True, how='inner')

    df = df.reset_index()

    product['product_size'] = product['product_size'].apply(translate_product_size)
    product = product[product['product_size'] != -1]

    df = pd.merge(df, product[['upc', 'commodity', 'product_size']], how='inner', left_on='upc', right_on='upc')

    df = df.drop('upc', axis=1)
    df = df.groupby([col for col in df.columns if col not in ['units']]).sum().reset_index()
    df['sales_per_week'] = df['units'] / df['week']
    df['revenue_per_week'] = df['units'] * df['unit_price'] / df['week']
    df['weight_sold_per_week'] = df['units'] * df['product_size'] / df['week']
    return df


def do_explore_demographics(causal, transaction, product, store, county):
    '''
    do demographics features (based on store zip code) exploration
    '''

    zip_helper = ZipHelper()
    df = do_join_3(causal, transaction, product, drop_store=False)

    if st.session_state.get('income', False):
        store['household_count'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
            apply(lambda x:
                  county[county['Geographic Area Name'] == x]['Total!!Estimate!!Households'].tolist()[
                      0])

        store['median_household_income'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
            apply(lambda x:
                  county[county['Geographic Area Name'] == x]['Median income (dollars)!!Estimate!!Households'].tolist()[
                      0])

    if st.session_state.get('demographics', False):
        for race_col in ['Estimate!!RACE!!Total population', 'Estimate!!RACE!!White',
                         'Estimate!!RACE!!Black or African American',
                         'Estimate!!RACE!!American Indian and Alaska Native',
                         'Estimate!!RACE!!Asian',
                         'Estimate!!RACE!!Native Hawaiian and Other Pacific Islander',
                         'Estimate!!RACE!!Some other race',
                         'Estimate!!HISPANIC OR LATINO AND RACE!!Hispanic or Latino (of any race)']:
            store[race_col.split('!!')[2]] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][race_col].tolist()[0])
        store['Some other race'] = store['Some other race'] + store['Native Hawaiian and Other Pacific Islander']
        store = store.drop('Native Hawaiian and Other Pacific Islander', axis=1)

    for commute in ['residents_commute', 'workers_commute']:
        if st.session_state.get(commute, False):
            store[f'commuters_{commute[0]}'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][
                f'Estimate!!Total_{commute[0]}'].tolist()[0])
            store[f'no_vehicle_{commute[0]}'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][
                f'Estimate!!Total!!No vehicle available_{commute[0]}'].tolist()[0])
            store[f'1+_vehicle_{commute[0]}'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][
                [f'Estimate!!Total!!1 vehicle available_{commute[0]}',
                 f'Estimate!!Total!!2 vehicles available_{commute[0]}',
                 f'Estimate!!Total!!3 or more vehicles available_{commute[0]}']].to_numpy().sum())
            store[f'commuters_private_{commute[0]}'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][
                f'Estimate!!Total!!Car, truck, or van - drove alone_{commute[0]}'].tolist()[0])
            store[f'commuters_other_{commute[0]}'] = store['store_zip_code'].apply(zip_helper.zip_to_cs). \
                apply(lambda x: county[county['Geographic Area Name'] == x][
                [f'Estimate!!Total!!Car, truck, or van - carpooled_{commute[0]}',
                 f'Estimate!!Total!!Public transportation (excluding taxicab)_{commute[0]}',
                 f'Estimate!!Total!!Taxicab, motorcycle, bicycle, or other means_{commute[0]}',
                 f'Estimate!!Total!!Walked_{commute[0]}',
                 f'Estimate!!Total!!Worked at home_{commute[0]}']].to_numpy().sum())

    store['county'] = store['store_zip_code'].apply(zip_helper.zip_to_cs)
    store.groupby('county').count()
    county_count = []
    ids = []
    for value in store.groupby('county').groups.values():
        county_count += [len(value) for _ in range(len(value))]
        ids += list(value)
    store['county_store_count'] = pd.Series(county_count, index=ids)

    df = pd.merge(df, store, how='inner', left_on='store', right_on='store') \
        .drop(['store', 'store_zip_code', 'feature_desc', 'display_desc', 'geography', 'coupon'], axis=1)

    for measurement_choice in ['sales_per_week',
                               'revenue_per_week',
                               'weight_sold_per_week']:
        df[measurement_choice.rsplit('_', 2)[0]] = df[measurement_choice] * df['week']
    df = df.drop(['sales_per_week',
                  'revenue_per_week',
                  'weight_sold_per_week'], axis=1)
    df = df.groupby([col for col in df.columns if col not in ['units', 'week', 'sales',
                                                              'revenue',
                                                              'weight_sold']]).sum().reset_index()

    df = df.drop(['unit_price', 'week', 'units', 'product_size'], axis=1)
    df = df.groupby([col for col in df.columns if col not in [
        'sales', 'revenue', 'weight_sold'
    ]]).sum().reset_index()

    numerical_cols = [col for col in df.columns if col not in
                      ['county', 'units', 'sales', 'revenue', 'weight_sold', 'sales_per_week',
                       'revenue_per_week',
                       'weight_sold_per_week',
                       'week',
                       'feature_desc', 'display_desc', 'geography', 'coupon', 'commodity',
                       'upc', 'store']]

    st.plotly_chart(px.scatter(
        df.rename(columns={'Total population': 'county population'}),
        x='county population' if 'Total population' in df.columns else 'county_store_count',
        y='revenue', color='commodity', size='county_store_count' if 'Total population' in df.columns else None,
        hover_data=['sales', 'weight_sold'], trendline='ols'))
    numerical_commodity_tabs = st.tabs(['pasta', 'pasta sauce', 'syrups', 'pancake mixes'])
    for commodity, commodity_tab in zip(['pasta', 'pasta sauce', 'syrups', 'pancake mixes'], numerical_commodity_tabs):
        with commodity_tab:
            demographics_df_scatter(df, commodity, numerical_cols)


@st.cache_data
def do_explore_marketing(causal, transaction, product, measurement_choice):
    '''
    do marketing features (feature, display, coupon, etc.) exploration
    '''
    df = do_join_3(causal, transaction, product)
    categorical_commodity_tabs = st.tabs(['pasta', 'pasta sauce', 'syrups', 'pancake mixes'])
    for cctab, commodity in zip(categorical_commodity_tabs, ['pasta', 'pasta sauce', 'syrups', 'pancake mixes']):
        with cctab:
            dfc = df[df['commodity'] == commodity]
            categorical_tabs = st.tabs(['feature_desc', 'display_desc', 'geography', 'coupon'])
            for tab, category in zip(categorical_tabs, ['feature_desc', 'display_desc', 'geography', 'coupon']):
                dfca = dfc.copy()
                dfca[measurement_choice] = dfca[measurement_choice] * dfca['week']
                dfca = dfca[[category, measurement_choice, 'week']].groupby(category).sum().reset_index()
                dfca[measurement_choice] = dfca[measurement_choice] / dfca['week']
                dfca = dfca.sort_values(measurement_choice, ascending=False).drop('week', axis=1)
                with tab:
                    fig = go.Figure()
                    for feature in dfca[category].unique():
                        fig.add_trace(go.Box(
                            y=dfc[dfc[category] == feature][measurement_choice],
                            name=str(feature),
                            boxpoints='outliers'
                        ))
                    fig.update_yaxes(type="log")
                    fig.update_layout(title={'text': f'{category} {measurement_choice}'})
                    st.plotly_chart(fig)
                    columns[f'{category}-{measurement_choice}'] = st.columns([4, 6])
                    with columns[f'{category}-{measurement_choice}'][0]:
                        st.text(f'weighted mean per {category}')
                        st.dataframe(dfca.set_index(category), use_container_width=True)
                    with columns[f'{category}-{measurement_choice}'][1]:
                        st.text(f'anova p-value for {category}')
                        values = dfc[category].unique()
                        st.dataframe(
                            pd.concat([pd.Series([values[i], values[j],
                                                  f_oneway(dfc[measurement_choice][dfc[category] == values[i]],
                                                           dfc[measurement_choice][dfc[category] == values[j]])[1]])
                                       for j in range(len(values))
                                       for i in range(j)], axis=1)
                                .T
                                .rename(columns={0: 'value 1', 1: 'value 2', 2: 'anova p-value'})
                                .set_index(['value 1', 'value 2'])
                                .sort_values('anova p-value'),
                            use_container_width=True
                        )


with st.sidebar:
    '''
    the sidebar file uploaders
    '''
    st.markdown("""This app was initially designed to potentially run on 1 GB cloud memory limit. This constraint has since been lifted as the app expanded.
    
As a legacy of that constraint, **you only need to (and can only) upload once per file.**

If you uploaded the wrong file by mistake or need to start over for any reason, please reload the page.
    """)
    for file in ['causal', 'transactions', 'product', 'store']:
        globals()[f'{file}_file'] = st.file_uploader(f'Upload {file} csv')

        if globals()[f'{file}_file'] and not st.session_state.get(file, pd.DataFrame([[False]])).any(axis=None):
            st.session_state[file] = pd.read_csv(globals()[f'{file}_file'])
        del globals()[f'{file}_file']

    st.divider()

    for file in ['income', 'demographics', 'residents_commute', 'workers_commute']:
        globals()[f'{file}_file'] = st.file_uploader(f'Upload {file} csv')

        if globals()[f'{file}_file'] and not st.session_state.get(file, False):
            temp_df = pd.read_csv(globals()[f'{file}_file'], encoding='latin-1', skiprows=1)
            st.session_state['county'] = pd.merge(st.session_state['county'],
                                                  temp_df[list(filter(get_state_column_filter(file), temp_df.columns))],
                                                  how='right',
                                                  left_on='Geographic Area Name',
                                                  right_on='Geographic Area Name').rename(
                columns=lambda x: x if x[-2:] == '_r' or x[-2:] == '_w' or get_state_column_filter('income')(
                    x) or get_state_column_filter('demographics')(
                    x) or x == 'Geographic Area Name' else f'{x}_{file[0]}')
            st.session_state[file] = True
            del temp_df
        del globals()[f'{file}_file']

st.text('If you want to see the most exciting results first, start from the bottom section')
st.text('All plots are interactive, with at least hover info and the ability to zoom in')

with st.expander('table preview plots'):
    st.markdown("""
    ### Section Summary
    
    This section display bar plots of selected group-by features.
    
    The **store** section displays a heat map of store count on US states, and we find the stores are in 7 eastern US states, with most stores located in Georgia, Tennessee and Kentucky.
    """)
    exploration_tabs = st.tabs(['causal', 'transactions', 'product', 'store'])
    with exploration_tabs[0]:
        if st.session_state.get('causal', pd.DataFrame([[False]])).any(axis=None):
            causal = st.session_state['causal']
            try:
                plot_multiindex(causal
                                .groupby(reduce(list.__add__,
                                                [st.checkbox(f'group by {col}', key=f'explore_causal_{col}') * [col] for
                                                 col in causal.columns]))
                                .count().iloc[:, 0:1]
                                .rename(columns=lambda x: 'Count')
                                .sort_values('Count', ascending=False, key=np.vectorize(lambda x: x * random.random()))
                                .head(10)
                                .sort_values('Count'), n_cols=1)
            except Exception as e:
                print(e)

    with exploration_tabs[1]:
        if st.session_state.get('transactions', pd.DataFrame([[False]])).any(axis=None):
            transactions = st.session_state['transactions']
            try:
                plot_multiindex(transactions.copy().assign(units_sold=transactions.copy().pop('units'))
                                .assign(dollar_sale=transactions.copy().pop('dollar_sales')).drop(
                    ['time_of_transaction', 'week', 'day', 'household'], axis=1)
                                .groupby(reduce(list.__add__,
                                                [st.checkbox(f'group by {col}', key=f'explore_transactions_{col}') * [
                                                    col] for col in [x for x in transactions.columns if
                                                                     x not in ['units', 'dollar_sales',
                                                                               'time_of_transaction', 'week', 'day',
                                                                               'household']]]))
                                .sum()[['units_sold', 'dollar_sale']]
                                .sort_values(['units_sold', 'dollar_sale'], ascending=False,
                                             key=np.vectorize(lambda x: x * random.random()))
                                .head(10)
                                .sort_values(['units_sold', 'dollar_sale']), n_cols=2)
            except:
                pass

    with exploration_tabs[2]:
        if st.session_state.get('product', pd.DataFrame([[False]])).any(axis=None):
            product = st.session_state['product']
            try:
                plot_multiindex(
                    product.copy().assign(product_size=product.copy().pop('product_size').apply(translate_product_size))
                        .query('product_size != -1')
                        .groupby(reduce(list.__add__,
                                        [st.checkbox(f'group by {col}', key=f'explore_product_{col}') * [col] for col in
                                         [x for x in product.columns]]))
                        .count().iloc[:, 0:1]
                        .rename(columns=lambda x: 'Count')
                        .sort_values('Count', ascending=False, key=np.vectorize(lambda x: x * random.random()))
                        .head(10)
                        .sort_values('Count'), n_cols=1)
            except Exception as e:
                print(e)

    with exploration_tabs[3]:
        if st.session_state.get('store', pd.DataFrame([[False]])).any(axis=None):
            store = st.session_state['store']
            plot_states(store)
            plot_multiindex(store
                            .groupby('store_zip_code')
                            .count().iloc[:, 0:1]
                            .rename(columns=lambda x: 'Count')
                            .sort_values('Count', ascending=False, key=np.vectorize(lambda x: x * random.random()))
                            .head(10)
                            .sort_values('Count'), n_cols=1)

with st.expander('explore marketing campaigns'):
    st.text('(requires causal, transactions, and product tables)')
    st.markdown("""
    ## Section Summary
    
#### Key takeaways 
    
    1. geography of 2 has statistically higher measurement values than geography of 1
    2. coupon of 1 gives higher units sold and weight sold, but lower revenue
    3. feature and display effects are generally intuitive in units sold and weights sold,
    with front page feature selling significantly more than not on feature, etc.
    4. the effect is less clear and more confounded on revenue, probably because featured items tend to be on sale
    5. effect of being on promo/seasonal aisle is also intuitive, selling more items but not necessarily earning more
    
Overall this preliminary exploration results are within human intuition.

#### Methods

    1. Causal and Transactions are joined on upc-week-store
    2. The resulting join is then joined with Product on upc
    3. 'sales_per_week', 'revenue_per_week', and 'weight_sold_per_week' are created for use as potential metrics
    4. Each marketing feature is explored at a time
    5. ANOVA test is then performed
    6. Box plots are made to help visualize
    7. Group mean table is also created
        i. the metrics are remultiplied by week,
        ii. then the data is grouped by the selected feature
        iii. remultiplied metric and week are them summed up, and redivided again, to obtain the aggregated metric per week

""")

    if st.session_state.get('causal', pd.DataFrame([[False]])).any(axis=None) and \
            st.session_state.get('transactions', pd.DataFrame([[False]])).any(axis=None) and \
            st.session_state.get('product', pd.DataFrame([[False]])).any(axis=None):
        measurement_choice = st.selectbox('show',
                                          options=['sales_per_week', 'revenue_per_week', 'weight_sold_per_week'],
                                          index=1)
        do_explore_marketing(st.session_state['causal'].copy(),
                             st.session_state['transactions'].copy(),
                             st.session_state['product'].copy(),
                             measurement_choice)

with st.expander('explore demographics variables'):
    st.text('(requires all main tables + at least one additional table)')
    st.text('ML will appear at bottom of section')

    st.markdown("""
    ## Section Summary
    
    #### Key takeaways
    
Assuming no relationship of any kind, the key predictor of sales of each commodity type in a county should be either

    - Population of county
    - Store count in county (which should potentially be the better predictor, assuming not all stores in county is in our data tables)
    
However this is not always the case

    1. The null hypothesis only holds true for **pasta**.
    2. For **pasta sauce**, the second best linear predictor of sales is the number of 'White' population in county.
        - followed by the number of commuters that commute to work by private cars
    3. For **syrups**, the null hypothesis holds true for sales and weight sold, but not for revenue
        - Revenue is predicted better by White population and number of private-car commuters
        - So white consumers and motorists tend to buy more expensive syrups
        - However it is hard to quantify the statistical significance of this result
    4. For **pancake mixes**, reality differs even more from the null hypothesis
        - All metrics are best predicted by number of 'Black' population (0.9 pearson correlation)
        - Followed by number of commuters who doesn't own cars
        - then by number of commuters who commute by non-private means (public transport, walking, WfH, etc.)
        - Pearson correlation of county store count or of population are only ~0.8
        
Actionable insight from this section is that depending on what you want to sell, your store location matters.

Or for existing stores, stocking the right items might earn you more sales.

#### Methods

    1. All of upc (item code), selling price, and marketing factors such as features and displays are removed.
    2. The remaining columns are then aggregated to obtain the amount of sales for each commodity type in each county.
    3. Each county is concatenated with additional county demographics depending on additional tables
        - otherwise only county_store_count (the number of stores in county in the data tables) is used
    4. Pearson correlation is obtained for each commodity-county pair.
    5. Using the most correlated feature in order from Pearson correlation, linearly uncorrelated features are selected for Linear Regression models
        - not all combinations of features are used, but feature selection is guided by Pearson correlation
    6. The fitted Linear Regression models with Leave-One-Out data split obtained R-squared values of
        - 0.89 for pasta (1 feature)
        - 0.77 for pasta sauce (8 features)
        - 0.81 for syrups (8 features)
        - 0.76 for pancake mixes (1 feature)
        For such simple linear models, I consider these R-squared unexpectedly predictive.

    """)
    if not st.session_state.get('demographics_ml', False):
        st.session_state['demographics_ml'] = st.checkbox('do_demographics_ml')
    if st.session_state.get('causal', pd.DataFrame([[False]])).any(axis=None) and \
            st.session_state.get('transactions', pd.DataFrame([[False]])).any(axis=None) and \
            st.session_state.get('product', pd.DataFrame([[False]])).any(axis=None) and \
            st.session_state.get('store', pd.DataFrame([[False]])).any(axis=None):
        do_explore_demographics(st.session_state['causal'].copy(),
                                st.session_state['transactions'].copy(),
                                st.session_state['product'].copy(),
                                st.session_state['store'].copy(),
                                st.session_state['county'])
