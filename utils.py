import re
from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut


class ZipHelper:
    '''
    Legacy design (to minimize memory use)
    import library in a class object to recycle memory after use
    REQUIRES INTERNET CONNECTION
    '''

    def __init__(self):
        import uszipcode
        self.se = uszipcode.search.SearchEngine()
        state_df = pd.read_html('https://www.worldatlas.com/geography/usa-states.html')[0]
        self.state_dict = dict(zip(state_df['Abbreviation'].tolist(), state_df['US State'].tolist()))

    def zip_to_state(self, x):
        return self.se.by_zipcode(x).state

    def zip_to_cs(self, x):
        return self.se.by_zipcode(x).county + ', ' + self.state_dict[self.se.by_zipcode(x).state]


def plot_multiindex(df, n_cols=1):
    '''
    plot table preview bar charts
    '''
    df = df.reset_index()
    df['group'] = reduce(lambda x, y: x.apply(lambda z: z + '-') + y,
                         [f'{col}_' + df[col].astype(str) for col in list(df.columns)[:-n_cols]])

    return st.plotly_chart(
        px.bar(df.set_index('group')[df.columns.tolist()[-1 - n_cols:-1]], orientation='h', barmode='group',
               color_discrete_sequence=px.colors.qualitative.Safe))


def plot_states(df):
    '''
    plot heatmap of store count on US states map
    '''
    zip_helper = ZipHelper()
    store_state = df.groupby('store_zip_code').count().reset_index()
    store_state['state'] = store_state['store_zip_code'].apply(zip_helper.zip_to_state)
    store_state = store_state.drop('store_zip_code', axis=1)
    store_state = store_state.groupby('state').sum().reset_index()

    locations = store_state['state']
    z = store_state['store']
    fig = go.Figure(data=go.Choropleth(
        locations=locations,
        z=z,
        locationmode='USA-states',
        colorscale='Reds',
        colorbar_title="no. stores",
    ))

    fig.update_layout(
        title_text='Stores in DB by State',
        geo_scope='usa'
    )

    return st.plotly_chart(fig)


def demographics_df_scatter(df, commodity, numerical_cols, regression):
    '''
    display df and scatter plot for demographic features
    '''
    columns = st.columns(3)
    correlation_df = {}
    for mid, measurement_choice in enumerate(['sales',
                                              'revenue',
                                              'weight_sold']):
        correlation_df[measurement_choice] = pd.concat(
            [pd.Series([numeric, *pearsonr(df[df['commodity'] == commodity][numeric],
                                           df[df['commodity'] == commodity][
                                               measurement_choice])])
             for numeric in numerical_cols], axis=1) \
            .T \
            .rename(columns={0: 'predictor', 1: 'pearson_r', 2: 'p_value'}) \
            .set_index('predictor').sort_values('pearson_r', ascending=False)
        with columns[mid]:
            st.text(measurement_choice)
            st.dataframe(correlation_df[measurement_choice])
    st.plotly_chart(px.scatter(df[df['commodity'] == commodity],
                               x=correlation_df['revenue'].index[0], y='revenue',
                               color='commodity', size=correlation_df['revenue'].index[1] if len(
            correlation_df['revenue'].index) > 1 else None,
                               hover_data=['sales', 'weight_sold'], trendline='ols'))

    if not regression:
        st.text('to do Linear Regression, check the box at top of section')
    else:
        st.divider()
        do_demographics_regression(df[df['commodity'] == commodity].reset_index().drop('index', axis=1), commodity,
                                   correlation_df['revenue'])


def do_demographics_regression(df, commodity, correlations):
    '''
    do linear regression for revenue on demographic features
    '''
    predictors = get_top_predictors(correlations.index)
    for predictor in predictors:
        if 'non' in predictor:
            df[predictor] = df['Total population'] - df[
                list(df.columns)[[predictor.split('_')[1] in col for col in df.columns].index(True)]]
    r2s = []
    models = []
    for length in range(len(predictors)):
        x, y = df[predictors[:length + 1]], df[['revenue']]
        loos = LeaveOneOut().split(x)

        preds = []
        for i, (train_ind, test_ind) in enumerate(loos):
            model = LinearRegression()
            model.fit(X=x.iloc[train_ind], y=y.iloc[train_ind])
            preds.append(np.vectorize(lambda x: max(x, 0))(model.predict(x.iloc[test_ind]))[0])
        r2s.append(r2_score(y.to_numpy()[:, 0], np.array(preds)))
        models.append(model)

    st.text('Linear Regression from cumulatively adding linearly independent features in order of correlation,')
    st.text(f'best R-squared ({max(r2s)}) obtained at {np.argmax(r2s) + 1} features')
    st.text(f'these features are \'{", ".join(predictors[:np.argmax(r2s) + 1])}\'')

    features = predictors[:np.argmax(r2s) + 1]
    model = models[np.argmax(r2s)]
    x, y = df[features], df['revenue']
    true = pd.DataFrame([y]).T
    prediction = pd.concat([x, pd.Series(y), pd.Series(model.predict(x)[:, 0]).rename('predicted revenue')], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction['revenue'],
                             y=prediction['predicted revenue'], mode='markers', customdata=np.array(prediction),
                             name=f'Predicted {commodity} revenue',
                             hovertemplate='True revenue: %{customdata[' + f'{len(x.columns)}' + ']}<br>' +
                                           'Predicted revenue: %{customdata[' + f'{len(x.columns) + 1}' + ']}<br>' + '<br>'.join(
                                 [f'{x.columns[i]}: ' + '%{customdata[' + f'{i}' + ']}'
                                  for i in range(len(x.columns))])))
    fig.add_trace(go.Scatter(x=true['revenue'], y=true['revenue'], name=f'True {commodity} revenue'))
    st.plotly_chart(fig)


def get_top_predictors(correlations):
    '''
    cumulatively adds top correlated predictor as long as it's linearly independent to previous features
    '''

    def correlated(a, b):
        if any([x in ['county_store_count', 'median_household_income', 'household_income'] for x in [a, b]]):
            return False
        elif np.array([any([y in x for x in [a, b]]) for y in
                       ['Black', 'White', 'Asian', 'Hispanic', 'other race', 'Indian and Alaska']]).sum() > 1:
            return True
        elif any([x == 'Total population' for x in [a, b]]):
            return any([y in x for x in [a, b] for y in
                        ['Black', 'White', 'Asian', 'Hispanic', 'other race', 'Indian and Alaska']])
        elif any(['private_r' in x for x in [a, b]]):
            return any(['commuters_r' in x for x in [a, b]])
        elif any(['other_r' in x for x in [a, b]]):
            return any(['commuters_r' in x for x in [a, b]])
        elif any(['private_w' in x for x in [a, b]]):
            return any(['commuters_w' in x for x in [a, b]])
        elif any(['other_w' in x for x in [a, b]]):
            return any(['commuters_w' in x for x in [a, b]])
        elif any([all([f'commuters_{y}' in x for x in [a, b]]) for y in ['w', 'r']]):
            return True
        elif any([y in x for x in [a, b] for y in ['commuters_r', 'commuters_w']]):
            return any(['vehicle' in x for x in [a, b]])
        else:
            return False

    predictors = []
    for correlation in correlations:
        if all([not correlated(correlation, x) for x in predictors]):
            predictors.append(correlation)
            for word in ['Black', 'White', 'Asian', 'Hispanic', 'other race', 'Indian and Alaska']:
                if word in correlation:
                    predictors.append(f'non_{word}')
    return predictors


def translate_product_size(x):
    '''
    translates product size from Ounce and Lbs to KG
    '''
    if re.search('(Z)', x) is not None or re.search('(OUNCE)', x) is not None:
        return float(re.search('(\d+\.*\d*)', x).group(0)) / 35.274
    elif re.search('(LB)', x) is not None:
        return 16 * float(re.search('(\d+\.*\d*)', x).group(0)) / 35.274
    else:
        return -1


def get_state_column_filter(table):
    '''
    selects a subset of columns on each state data table
    '''
    if table in ['residents_commute', 'workers_commute']:
        return lambda x: x == 'Geographic Area Name' or ('Annotation' not in x and 'Margin' not in x)
    elif table == 'income':
        return lambda x: x == 'Geographic Area Name' or (x in ['Total!!Estimate!!Households',
                                                               'Median income (dollars)!!Estimate!!Households'])
    elif table == 'demographics':
        return lambda x: x == 'Geographic Area Name' or (x in ['Estimate!!RACE!!Total population',
                                                               'Estimate!!RACE!!White',
                                                               'Estimate!!RACE!!Black or African American',
                                                               'Estimate!!RACE!!American Indian and Alaska Native',
                                                               'Estimate!!RACE!!Asian',
                                                               'Estimate!!RACE!!Native Hawaiian and Other Pacific Islander',
                                                               'Estimate!!RACE!!Some other race',
                                                               'Estimate!!HISPANIC OR LATINO AND RACE!!Hispanic or Latino (of any race)'])
    else:
        assert False, 'Unknown state table'
