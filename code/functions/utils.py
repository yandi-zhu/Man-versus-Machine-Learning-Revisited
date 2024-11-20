import statsmodels.api as sm
import pandas as pd
import numpy as np
from functools import reduce
from . import summary2
### All of these function are tested in Pandas 1.2.0; Numpy 1.20.3; statsmodels 0.13.0

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

def merge_by_col(df1,df2):
    '''
    Merge two dataframes: (coefs) and (standard-errors) side-by-side
    '''
    new_columns = []
    
    # 遍历列名，交替添加
    for col_name in df1.columns:
        new_columns.append(df1[col_name])
        new_columns.append(df2[col_name])

    # 将列列表合并为一个DataFrame
    df_merged = pd.concat(new_columns, axis=1)

    # 由于直接合并会导致列名重复，我们需要重命名列
    new_col_names = []
    for i, col_name in enumerate(df1.columns):
        new_col_names.append(col_name)
        new_col_names.append('')

    df_merged.columns = new_col_names

    return df_merged
    
def cut_filter(df, cut_var, bins, filter_list=None, labels=None):
    '''
    Cut variables with filter
    -----------------------------
    df: DataFrame
    cut_var: str
        The cutted variable name
    filter_list: list
        A list of filter [filter variable name, filter value]
    bins: int
        The number of bins to cut
    labels: list
        Name of cutted bins' name
    '''
    ## Transform to rank to handle tie
    # 23/11/3 add first
    df[cut_var] = df[cut_var].transform(lambda x: x.rank(pct=True, method='first'))

    if filter_list:
    ## Get filter 
        filter_index = filter_list[0]
        filter_value = filter_list[1]

        ## filtered data
        temp = df[df[filter_index]==filter_value]

        ## calculate cut bins
        cut_value = temp[cut_var].quantile(bins).to_list()

    else:
        temp = df
        cut_value = bins        
    
    cut_value[0] = -np.inf
    cut_value[-1] = np.inf
    
    sort_var = pd.cut(df[cut_var], cut_value, labels=labels)
    
    return sort_var

def fama_macbeth(ret_monthly, formula_list, time_id, lags, WLS=None, float_format='%.4f',stars=True):
    '''
    Fama and MacBeth(1973) regression
    
    Parameters:
    ret_monthly: DataFrame
        the return data with characteristics
    formula_list: str or list
        the regression expressions to be tested
    time_id: str
        the time period id
    lags: int
        the New-West adjustment lag periods
    WLS: str
        Use Weighted-least Square in cross-sectional regression with `WLS` as weights
    float_format: str (optional)
        float format for coefficients and standard errors
        Default : '%.4f'
    -----------------------------
    
    '''
    df = ret_monthly.copy()
    
    if isinstance(formula_list,str):
        formula_list = [formula_list]
        
    result = []
    R2 = []
    N = []
    for idx, formula in enumerate(formula_list):
        ## for each formula
        
        # Step 1 Cross-sectional regression
        ### Input a regression formula 
        if WLS:
            cross_reg = ret_monthly.groupby(time_id).apply(lambda x: sm.WLS.from_formula(formula, data=x, weights=x[WLS], missing = 'drop') \
                                                                .fit() \
                                                    )
        else:
            cross_reg = ret_monthly.groupby(time_id).apply(lambda x: sm.OLS.from_formula(formula, data=x, missing = 'drop') \
                                                                .fit() \
                                                    )
        
        params = cross_reg.apply(lambda x:x.params)
        
        ## Other Statistics like R2
        N_i = int(cross_reg.apply(lambda x: x.nobs).sum())
        R2_i = cross_reg.apply(lambda x: x.rsquared_adj).mean() * 100
        
        # Step 2 Sequential summary
        ## Here I rename the series for later summary
        temp = params.agg(lambda x: sm.OLS(endog=x.rename(idx+1), exog=pd.Series([1]*len(x), index=x.index, name=x.name)).fit(cov_type = 'HAC', cov_kwds = {'maxlags':lags}))

        ## Summary Stats
        result_i = reduce(lambda x,y: pd.concat([x,y], axis=0),[summary2.summary_col(i,float_format,stars=stars) for i in temp])
        
        ## Reset index (for combination). may improve by other methods
        index = [[i,'%s_t' % i] for i in result_i.index if i !='']
        index = [x for l in index for x in l]
        result_i.index = index
        
        result.append(result_i)
        
        ## Other Statistics like R2
        N.append(N_i)
        R2.append(float_format % R2_i)
        
    result = pd.concat(result, axis=1)
    
    ## Reset index to drop t-stats
    index = [[result.index[i], ''] for i in range(len(result.index)) if i % 2 == 0]
    index = [x for l in index for x in l]
    result.index = index
    # result = result.T  
    
    R2 = pd.DataFrame({'R2':R2}, index = [str(i+1) for i in range(len(formula_list))]).transpose()
    N = pd.DataFrame({'N':N}, index = [str(i+1) for i in range(len(formula_list))]).transpose()
    # R2.round(3)
    
    result = pd.concat([result, N, R2])
    
    return result

def SingleSort(return_data, entity_id, time_id, sort_var, agg_var, num_level, weight_var, quantile_filter=None):
    """
    This function do single sort and return a new Dataframe contains a sorted group

    Parameters:
    -------------------------------------------
    return_data: DataFrame
    entity_id: str
        stock id like permno
    time_id: str
        time_id denotes the time dimension the panel dataset.
    sort_var: str
        the chosen variable to sort stocks into portfolios
    agg_var: str
        the name of the return column
    num_level: int or list
        int stands for the number of portfolios
        list of quantiles [0,0.3,0.7,1], these quantiles will be use if specificed
    weight_var: str
        the weight of each stock in its portfolio, if not assigned the portfolio will be equal weighted 
    quantile_filter : list, optional
        quantile_filter is a list = [column_name, value] with the first element 
        being the name of the column and the second being the value for which
        the quantiles should be calculated only. For example, if 
        quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for
        the estimation of quantiles. If it is None, all entities will be used.
    """ 
    df = return_data.copy()

    raw_row = df.shape[0]
    df = df.dropna(subset=[sort_var,agg_var])
    new_row = df.shape[0]
    
    print('Var:{}, Delete {} rows due to missing values, raw data {} rows --> new data {} rows'.format(sort_var, raw_row-new_row, raw_row, new_row))

    if isinstance(num_level, int):
        num_level = [i/num_level for i in range(num_level+1)]
    
    ## Cut into portfolio    
    df['port_var'] = df.groupby(time_id).apply(lambda x: cut_filter(x, sort_var, num_level, quantile_filter, labels=range(1, len(num_level)))).reset_index(level=0,drop=True)
    df['port_var'] = df['port_var'].astype('int')
    
    vwret = df.groupby([time_id,'port_var']).apply(wavg, agg_var, weight_var).reset_index().rename(columns={0: 'vwret'})
    vwret = vwret.pivot(index=time_id, columns='port_var', values='vwret') * 100
    # vwret.columns = [str(i) for i in range(1,num_level+1)] # Unless the columns are category type
    vwret['H-L'] = vwret.iloc[:,-1] - vwret.iloc[:,0]
    
    return df,vwret

def SingleSort_RetAna(df, vwret, time_id, factor_data, factor_dict, lag, float_format='%.2f'):
    '''
    This function do basic Single sort Portfolio analysis
    
    ---------------------------------------
    df: data with port_var as portfolio assignment
    vwret: vwret data
    agg_char: list
        other characteristics to aggregate
    factor_data: DataFrame
        the factor_data to be merged, the merged column MUST rename to time_id
    factor_dict: dict of lists
        the test factor model list {factor model name: [list of factors]}
        e.g.: {'Ret':['ones'],
               'CAPM':['ones','Mkt-RF']}
    lag: int
        the number of lags of Newey-West adjustment
    '''
    num_level = len(df.port_var.unique())
    if factor_data is not None:
        vwret = vwret.merge(factor_data, left_index=True, right_on=time_id)
    ## constant term for regression
    vwret['ones'] = 1
    result = []
    # For each test factor model
    for name, factor_list in factor_dict.items():
        model = []
        ## For each portfolio
        for i in list(range(1,num_level+1)) + ['H-L']:
            model_i = sm.OLS(endog=vwret[i], exog=vwret[factor_list], missing='drop')
            if lag:
                model_i = model_i.fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})
            else:
                model_i = model_i.fit(cov_type = 'HC0')
            model.append(model_i)
        model_result = summary2.summary_col(model, regressor_order=['ones'], drop_omitted=True, float_format=float_format)
        model_result.rename(index={'ones':name},inplace=True)
        result.append(model_result)
    result = pd.concat(result, axis=0)
    return result

def Factor_Exposure(vwret, num_level, time_id, factor_data, factor_dict, lag, regressor_order=['ones']):
    '''
    This function is used to get common risk factor exposure
    vwret is output from SingleSort
    e.g.: 
        Factor_Exposure(vwret1, num_level, 'YearMonth', 
                        all_factor, factor_dict2, 12, 
                        regressor_order = ['ones','Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'])
    '''
    vwret = vwret.merge(factor_data, left_index=True, right_on=time_id)
    vwret['ones'] = 1
    coefs = []
    coefs_t = []
    
    # For each test factor model
    for name, factor_list in factor_dict.items():
        
        model_i = sm.OLS(endog=vwret['H-L'], 
                         exog=vwret[factor_list], 
                         missing='drop')
        if lag:
            model_i = model_i.fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})
        else:
            model_i = model_i.fit(cov_type = 'HC0')
        
        coefs.append(summary2.summary_params(model_i)['Coef.'].rename(name))
        coefs_t.append(summary2.summary_params(model_i)['t'].rename(name))
        # model.append(model_i)
            
    # result = summary2.summary_col(model, regressor_order=regressor_order,  float_format='%.2f')
    coefs = pd.concat(coefs,axis=1)
    coefs_t = pd.concat(coefs_t,axis=1)
    result = merge_by_col(coefs, coefs_t)
    return result

def DoubleSort(return_data, time_id, sort_var1, sort_var2, num_level1, num_level2, agg_var, weight_var,lag, dependent=False, quantile_filter1=None, quantile_filter2=None):
    """
    This function do single sort and return a new Dataframe contains a sorted group

    Parameters:
    -------------------------------------------
    return_data: DataFrame
    entity_id: str
        stock id like permno
    time_id: str
        time_id denotes the time dimension the panel dataset.
    sort_var: str
        the chosen variable to sort stocks into portfolios
    agg_var: str
        the name of the return column
    num_level: int or list
        int stands for the number of portfolios
        list of quantiles [0,0.3,0.7,1], these quantiles will be use if specificed
    weight_var: str
        the weight of each stock in its portfolio, if not assigned the portfolio will be equal weighted 
    quantile_filter : list, optional
        quantile_filter is a list = [column_name, value] with the first element 
        being the name of the column and the second being the value for which
        the quantiles should be calculated only. For example, if 
        quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for
        the estimation of quantiles. If it is None, all entities will be used.
    """ 
    df = return_data.copy()
    
    raw_row = df.shape[0]
    df = df.dropna(subset=[sort_var1, sort_var2, agg_var])
    new_row = df.shape[0]
    print('Delete {} rows due to missing values, raw data {} rows --> new data {} rows'.format(raw_row-new_row, raw_row, new_row))
    
    if isinstance(num_level1, int):
        num_level1 = [i/num_level1 for i in range(num_level1+1)]
    if isinstance(num_level2, int):
        num_level2 = [i/num_level2 for i in range(num_level2+1)]
    
    ## Cut into portfolio
    df['port_var1'] = df.groupby(time_id)\
                        .apply(lambda x: cut_filter(x, sort_var1, num_level1, quantile_filter1, labels=range(1, len(num_level1))))\
                        .reset_index(level=0,drop=True)
    df['port_var1'] = df['port_var1'].astype('int')
    
    if dependent:        
        df['port_var2'] = df.groupby(['yyyymm','port_var1'])\
                            .apply(lambda x: cut_filter(x, sort_var2, num_level2, quantile_filter2, labels=range(1, len(num_level2)))) \
                            .reset_index(level=[0,1],drop=True)
    else:
        df['port_var2'] = df.groupby(['yyyymm'])\
                            .apply(lambda x: cut_filter(x, sort_var2, num_level2, quantile_filter2, labels=range(1, len(num_level2))))\
                            .reset_index(level=0,drop=True)
    df['port_var2'] = df['port_var2'].astype('int')
    
    vwret = df.groupby([time_id,'port_var1','port_var2']).apply(wavg, agg_var, weight_var).reset_index().rename(columns={0: 'vwret'})
    vwret['vwret'] = vwret['vwret']*100
    
    port_var1_levels = vwret.port_var1.unique()

    result = []
    ## For each group in PortSort 1
    for port_var1 in port_var1_levels:
        # port_var2_levels = vwret.port_var2.unique()
        ## Returns of each port2 and H-L
        sub_vwret = vwret[(vwret.port_var1==port_var1)].pivot(index=time_id, columns='port_var2', values='vwret')
        sub_vwret['H-L'] = sub_vwret.iloc[:,-1] - sub_vwret.iloc[:,0]
        ##
        models = sub_vwret.apply(lambda x: sm.OLS(endog=x, exog=[1]*len(x)).fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})).to_list()

        result_i = summary2.summary_col(models, float_format='%.3f')
        result_i.rename(index={'const':port_var1},inplace=True)

        result.append(result_i)
        
    result = pd.concat(result, axis=0)
    
    return df,vwret,result

def DoubleSort2(return_data, time_id, sort_var1, sort_var2, num_level1, num_level2, agg_var, weight_var,lag, dependent=False, 
                adjust_factor = None, factor_data = None,
                quantile_filter1=None, quantile_filter2=None):
    """
    This function do Double sort and return a new Dataframe contains a sorted group
    Update 12/29: This Version adds H-L in both side & Risk adjusted return
    Parameters:
    -------------------------------------------
    return_data: DataFrame
    entity_id: str
        stock id like permno
    time_id: str
        time_id denotes the time dimension the panel dataset.
    sort_var: str
        the chosen variable to sort stocks into portfolios
    agg_var: str
        the name of the return column
    num_level: int or list
        int stands for the number of portfolios
        list of quantiles [0,0.3,0.7,1], these quantiles will be use if specificed
    weight_var: str
        the weight of each stock in its portfolio, if not assigned the portfolio will be equal weighted 
    lag: int
        The Newey-West adjustment 
    adjust_factor: list
        The factors used to adjust return. e.g.: ['ones','MKT'] for CAPM
    factor_data: DataFrame
        Factor_data contains `time_id` for matching
    quantile_filter : list, optional
        quantile_filter is a list = [column_name, value] with the first element 
        being the name of the column and the second being the value for which
        the quantiles should be calculated only. For example, if 
        quantile_filter = ['EXCHCD', 1], then only NYSE stocks will be used for
        the estimation of quantiles. If it is None, all entities will be used.
    """ 
    df = return_data.copy()

    raw_row = df.shape[0]
    df = df.dropna(subset=[sort_var1, sort_var2, agg_var])
    new_row = df.shape[0]
    n_periods = len(df[time_id].unique())
    print('Var1:{} Var2:{}; Delete {} rows due to missing values, raw data {} --> new data {}; Number of Periods: {}'.format(sort_var1,sort_var2,raw_row-new_row, raw_row, new_row,n_periods))

    if isinstance(num_level1, int):
        num_level1 = [i/num_level1 for i in range(num_level1+1)]
    if isinstance(num_level2, int):
        num_level2 = [i/num_level2 for i in range(num_level2+1)]

    ## Cut into portfolio
    df['port_var1'] = df.groupby(time_id)\
                        .apply(lambda x: cut_filter(x, sort_var1, num_level1, quantile_filter1, labels=range(1, len(num_level1))))\
                        .reset_index(level=0,drop=True)
    df['port_var1'] = df['port_var1'].astype('int')

    if dependent:        
        df['port_var2'] = df.groupby([time_id,'port_var1'])\
                            .apply(lambda x: cut_filter(x, sort_var2, num_level2, quantile_filter2, labels=range(1, len(num_level2)))) \
                            .reset_index(level=[0,1],drop=True)
    else:
        df['port_var2'] = df.groupby([time_id])\
                            .apply(lambda x: cut_filter(x, sort_var2, num_level2, quantile_filter2, labels=range(1, len(num_level2))))\
                            .reset_index(level=0,drop=True)
    df['port_var2'] = df['port_var2'].astype('int')

    vwret = df.groupby([time_id,'port_var1','port_var2']).apply(wavg, agg_var, weight_var).reset_index().rename(columns={0: 'vwret'})
    vwret['vwret'] = vwret['vwret']*100

    ## Calculate H-L return
    # First sort by sort_var1 & H-L of var2
    vwret = vwret.pivot(index=[time_id,'port_var1'],columns='port_var2',values='vwret')
    vwret['H-L'] = vwret.iloc[:,-1] - vwret.iloc[:,0]
    # Reorder. Sort by sort_var2 & H-L of var1
    vwret = vwret.stack().unstack(level=1)
    vwret['H-L'] = vwret.iloc[:,-1] - vwret.iloc[:,0]
    vwret = vwret.stack().reset_index().rename(columns={0: 'vwret'})

    port_var1_levels = vwret.port_var1.unique()

    result = []

    ## Factor Data
    if adjust_factor:
        # factor_data = factor_data[(factor_data[time_id]>=vwret[time_id].min()) & (factor_data[time_id]<=vwret[time_id].max())].copy()
        factor_data = factor_data[factor_data[time_id].isin(vwret[time_id].unique())].copy()
        factor_data['ones'] = 1
        factor_data.set_index(time_id, inplace=True)

    ## For each group in PortSort 1
    for port_var1 in port_var1_levels:
        # port_var2_levels = vwret.port_var2.unique()
        ## Returns of each port2 and H-L
        sub_vwret = vwret[(vwret.port_var1==port_var1)].pivot(index=time_id, columns='port_var2', values='vwret')
        # sub_vwret['H-L'] = sub_vwret.iloc[:,-1] - sub_vwret.iloc[:,0]
        if adjust_factor:
            models = sub_vwret.apply(lambda x: sm.OLS(endog=x, exog=factor_data[adjust_factor].values).fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})).to_list()
        else:
            models = sub_vwret.apply(lambda x: sm.OLS(endog=x, exog=[1]*len(x)).fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})).to_list()

        result_i = summary2.summary_col(models, float_format='%.2f').iloc[:2,:]
        result_i.rename(index={'const':port_var1},inplace=True)
        result.append(result_i)
        
    result = pd.concat(result, axis=0)
    
    return df,vwret,result


def DoubleSort_factor(df,vwret,time_id, factor_data, factor_dict, lag):
    '''
    This function do factor alpha test for double-sort portfolios
    
    Parameters
    -----------------------------
    df: data with port_var1 and port_var2 as portfolio assignment
    vwret: vwret data
    time_id: str
        factor_data time id
    factor_data: DataFrame
        the factor_data to be merged
    factor_dict: dict of lists
        the test factor model list {factor model name: [list of factors]}
        e.g.: {'Ret':['ones'],
               'CAPM':['ones','Mkt-RF']}
    lag: int
        the number of lags of Newey-West adjustment
    
    '''
    port_var1_levels = vwret.port_var1.unique()
    
    result = []
    for port_var1 in port_var1_levels:
        ## Returns of each port2 and H-L
        sub_vwret = vwret[(vwret.port_var1==port_var1)].pivot(index=time_id, columns='port_var2', values='vwret')
        sub_vwret['H-L'] = sub_vwret.iloc[:,-1] - sub_vwret.iloc[:,0]
        sub_vwret = sub_vwret.merge(factor_data, left_index=True, right_on=time_id)
        sub_vwret['ones'] = 1

        # For each test factor model
        model = []
        for name, factor_list in factor_dict.items():
            model_i = sm.OLS(endog=sub_vwret['H-L'].rename(name), exog=sub_vwret[factor_list], missing='drop').fit(cov_type = 'HAC', cov_kwds = {'maxlags':lag})
            model.append(model_i)
        ## Combine different factor model alpha
        port_result = summary2.summary_col(model, regressor_order=['ones'], drop_omitted=True, float_format='%.3f')
        port_result.rename(index={'ones':port_var1},inplace=True)
        ## Combine different Port1 group
        result.append(port_result)
    result = pd.concat(result, axis=0)
    
    return result


def plot_bins(df, x_var, y_var, bins, xlabel=None, ylabel=None):
    """
    This function takes in a dataframe and the names of the x and y variables,
    along with the number of bins to group the x variable. It then plots a scatter
    plot of the mean x versus the mean y for each bin.
    
    :param df: pandas DataFrame containing the data
    :param x_var: string, the name of the x variable in df
    :param y_var: string, the name of the y variable in df
    :param bins: int, the number of bins to divide the x variable into
    """
    # Create a new column in the dataframe that assigns each value of x_var to a bin
    df['x_bin'] = pd.qcut(df[x_var], bins)
    
    # Calculate the mean of x_var and y_var for each bin
    grouped_means = df.groupby('x_bin').mean().reset_index()
    
    # Plotting
    # plt.figure(figsize=(10, 6))
    sns.regplot(data=grouped_means, x=x_var, y=y_var)
    # plt.scatter(grouped_means[x_var], grouped_means[y_var], color='blue')
    # plt.title('Scatter Plot of Mean Values of {} and {}'.format(x_var, y_var))
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # plt.grid(True)
    plt.show()