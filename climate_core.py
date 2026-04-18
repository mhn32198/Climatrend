import pandas as pd
import numpy as np

def load_data(file_obj):
    df = pd.read_csv(file_obj)
    df.columns = [col.strip().upper() for col in df.columns]
    required_cols = ['YEAR', 'MONTH', 'DAY', 'PRCP', 'TMAX', 'TMIN']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    # For percentile logic, map Feb 29 to Feb 28 to handle leap years smoothly across the full dataset
    df['CALENDAR_DAY'] = df['DATE'].dt.strftime('%m-%d').replace('02-29', '02-28')
    return df

def apply_qc(df):
    df_clean = df.copy()
    qc_report = {
        'missing_values': 0,
        'logical_errors': 0,
        'outliers_tmax': 0,
        'outliers_tmin': 0,
        'outliers_prcp': 0
    }
    for col in ['PRCP', 'TMAX', 'TMIN']:
        mask = df_clean[col] == -99.9
        qc_report['missing_values'] += int(mask.sum())
        df_clean.loc[mask, col] = np.nan
        
    logical_mask = df_clean['TMAX'] < df_clean['TMIN']
    qc_report['logical_errors'] = int(logical_mask.sum())
    df_clean.loc[logical_mask, ['TMAX', 'TMIN']] = np.nan
    
    for col, report_key in [('TMAX', 'outliers_tmax'), ('TMIN', 'outliers_tmin'), ('PRCP', 'outliers_prcp')]:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        outlier_mask = (df_clean[col] > mean + 3 * std) | (df_clean[col] < mean - 3 * std)
        qc_report[report_key] = int(outlier_mask.sum())
        df_clean.loc[outlier_mask, col] = np.nan
        
    return df_clean, qc_report

def calculate_percentiles(df):
    """Calculates 10th/90th temp percentiles per calendar day and overall precipitation percentiles."""
    tmax_10 = df.groupby('CALENDAR_DAY')['TMAX'].quantile(0.10).rename('TMAX_10p')
    tmax_90 = df.groupby('CALENDAR_DAY')['TMAX'].quantile(0.90).rename('TMAX_90p')
    tmin_10 = df.groupby('CALENDAR_DAY')['TMIN'].quantile(0.10).rename('TMIN_10p')
    tmin_90 = df.groupby('CALENDAR_DAY')['TMIN'].quantile(0.90).rename('TMIN_90p')
    temp_pct = pd.concat([tmax_10, tmax_90, tmin_10, tmin_90], axis=1)
    
    wet_days = df[df['PRCP'] >= 1]['PRCP']
    prcp_95p = wet_days.quantile(0.95) if not wet_days.empty else 0
    prcp_99p = wet_days.quantile(0.99) if not wet_days.empty else 0
    
    return temp_pct, prcp_95p, prcp_99p

# --- THRESHOLD COUNTS ---
def calc_FD(df, thresh_fd=0):
    return df[df['TMIN'] < thresh_fd].groupby('YEAR').size().rename('FD')

def calc_SU(df, thresh_su=25):
    return df[df['TMAX'] > thresh_su].groupby('YEAR').size().rename('SU')

def calc_ID(df, thresh_id=0):
    return df[df['TMAX'] < thresh_id].groupby('YEAR').size().rename('ID')

def calc_TR(df, thresh_tr=20):
    return df[df['TMIN'] > thresh_tr].groupby('YEAR').size().rename('TR')

def calc_Rnn(df, thresh_rnn=30):
    return df[df['PRCP'] >= thresh_rnn].groupby('YEAR').size().rename('Rnn')

def calc_GSL(df, hemisphere='Northern'):
    df_gsl = df.copy()
    df_gsl['TG'] = (df_gsl['TMAX'] + df_gsl['TMIN']) / 2.0
    if hemisphere == 'Southern':
        df_gsl['GSL_YEAR'] = df_gsl.apply(
            lambda row: row['YEAR'] if row['MONTH'] >= 7 else row['YEAR'] - 1, axis=1
        )
    else:
        df_gsl['GSL_YEAR'] = df_gsl['YEAR']
        
    gsl_results = {}
    for year, group in df_gsl.groupby('GSL_YEAR'):
        group = group.sort_values('DATE').reset_index(drop=True)
        tg = group['TG'].values
        dates = group['DATE'].values
        start_idx = None
        for i in range(len(tg) - 5):
            if np.all(tg[i:i+6] > 5) and not np.isnan(tg[i:i+6]).any():
                start_idx = i
                break
        end_idx = None
        if start_idx is not None:
            threshold_date = pd.Timestamp(year=int(year), month=7, day=1) if hemisphere == 'Northern' else pd.Timestamp(year=int(year + 1), month=1, day=1)
            for i in range(start_idx + 6, len(tg) - 5):
                if pd.Timestamp(dates[i]) > threshold_date:
                    if np.all(tg[i:i+6] < 5) and not np.isnan(tg[i:i+6]).any():
                        end_idx = i
                        break
        if start_idx is not None and end_idx is not None:
            gsl_length = end_idx - start_idx
        elif start_idx is not None and end_idx is None:
            gsl_length = len(tg) - start_idx
        else:
            gsl_length = 0
        gsl_results[year] = gsl_length
    return pd.Series(gsl_results, name='GSL')

# --- PERCENTILE INDICES ---
def calc_percentile_indices(df, temp_pct, prcp_95p, prcp_99p):
    df_pct = df.copy()
    df_pct = df_pct.merge(temp_pct, left_on='CALENDAR_DAY', right_index=True, how='left')
    
    valid_tmax = df_pct['TMAX'].notna().groupby(df_pct['YEAR']).sum()
    valid_tmin = df_pct['TMIN'].notna().groupby(df_pct['YEAR']).sum()
    
    # Using lambda handles cases where valid counts could be 0
    tn10p = (df_pct[df_pct['TMIN'] < df_pct['TMIN_10p']].groupby('YEAR').size() / valid_tmin * 100).rename('TN10p')
    tx10p = (df_pct[df_pct['TMAX'] < df_pct['TMAX_10p']].groupby('YEAR').size() / valid_tmax * 100).rename('TX10p')
    tn90p = (df_pct[df_pct['TMIN'] > df_pct['TMIN_90p']].groupby('YEAR').size() / valid_tmin * 100).rename('TN90p')
    tx90p = (df_pct[df_pct['TMAX'] > df_pct['TMAX_90p']].groupby('YEAR').size() / valid_tmax * 100).rename('TX90p')
    
    def count_spell_days(series):
        if len(series) == 0: return 0
        # Create groups for consecutive True values
        cumsum = (~series).cumsum()
        # Size of each True group
        spell_lengths = series.groupby(cumsum).transform('sum')
        return int((series & (spell_lengths >= 6)).sum())
    
    df_pct['IS_WSDI'] = df_pct['TMAX'] > df_pct['TMAX_90p']
    df_pct['IS_CSDI'] = df_pct['TMIN'] < df_pct['TMIN_10p']
    
    wsdi = df_pct.groupby('YEAR')['IS_WSDI'].apply(count_spell_days).rename('WSDI')
    csdi = df_pct.groupby('YEAR')['IS_CSDI'].apply(count_spell_days).rename('CSDI')
    
    r95p = df_pct[df_pct['PRCP'] > prcp_95p].groupby('YEAR')['PRCP'].sum().rename('R95p')
    r99p = df_pct[df_pct['PRCP'] > prcp_99p].groupby('YEAR')['PRCP'].sum().rename('R99p')
    
    return [tn10p, tx10p, tn90p, tx90p, wsdi, csdi, r95p, r99p]

# --- DURATION / INTENSITY ---
def calc_CDD(df):
    def max_consecutive_dry(series):
        is_dry = series < 1
        return is_dry.groupby((~is_dry).cumsum()).sum().max() if not is_dry.empty else 0
    return df.groupby('YEAR')['PRCP'].apply(max_consecutive_dry).rename('CDD')

def calc_CWD(df):
    def max_consecutive_wet(series):
        is_wet = series >= 1
        return is_wet.groupby((~is_wet).cumsum()).sum().max() if not is_wet.empty else 0
    return df.groupby('YEAR')['PRCP'].apply(max_consecutive_wet).rename('CWD')

def calc_PRCPTOT(df):
    return df[df['PRCP'] >= 1].groupby('YEAR')['PRCP'].sum().rename('PRCPTOT')

def calc_RX1day(df):
    return df.groupby('YEAR')['PRCP'].max().rename('RX1day')

def calc_RX5day(df):
    return df.groupby('YEAR')['PRCP'].apply(lambda x: x.rolling(5).sum().max()).rename('RX5day')

def calc_SDII(df):
    wet_days = df[df['PRCP'] >= 1].groupby('YEAR')['PRCP']
    return (wet_days.sum() / wet_days.count()).rename('SDII')

# --- ABSOLUTE EXTREMES ---
def calc_TXx(df): return df.groupby('YEAR')['TMAX'].max().rename('TXx')
def calc_TNx(df): return df.groupby('YEAR')['TMIN'].max().rename('TNx')
def calc_TXn(df): return df.groupby('YEAR')['TMAX'].min().rename('TXn')
def calc_TNn(df): return df.groupby('YEAR')['TMIN'].min().rename('TNn')
def calc_DTR(df):
    df_temp = df.copy()
    df_temp['DTR'] = df_temp['TMAX'] - df_temp['TMIN']
    return df_temp.groupby('YEAR')['DTR'].mean().rename('DTR')

# --- SEASONAL AVERAGES ---
def calc_seasonal_averages(df):
    df_seas = df.copy()
    mam = df_seas[df_seas['MONTH'].isin([3, 4, 5])].groupby('YEAR')['TMAX'].mean().rename('MAM_TMAX_Ave')
    jjas = df_seas[df_seas['MONTH'].isin([6, 7, 8, 9])].groupby('YEAR')['PRCP'].mean().rename('JJAS_PRCP_Ave')
    
    # DJF: December (prev year), January, February
    df_seas['DJF_YEAR'] = df_seas.apply(lambda row: row['YEAR'] + 1 if row['MONTH'] == 12 else row['YEAR'], axis=1)
    djf = df_seas[df_seas['MONTH'].isin([12, 1, 2])].groupby('DJF_YEAR')['TMAX'].mean().rename('DJF_TMAX_Ave')
    
    return pd.concat([mam, jjas, djf], axis=1)

# --- MAIN ORCHESTRATOR ---
def calculate_all_indices(df, hemisphere='Northern', 
                          thresh_fd=0, thresh_su=25, thresh_id=0, thresh_tr=20, 
                          thresh_rnn=30):
    years = sorted(df['YEAR'].unique())
    results = pd.DataFrame(index=years)
    results.index.name = 'Year'
    
    temp_pct, prcp_95p, prcp_99p = calculate_percentiles(df)
    
    indices = [
        calc_FD(df, thresh_fd), calc_SU(df, thresh_su), calc_TR(df, thresh_tr), calc_ID(df, thresh_id), calc_GSL(df, hemisphere),
        calc_Rnn(df, thresh_rnn),
        calc_CDD(df), calc_CWD(df), calc_PRCPTOT(df), calc_RX1day(df), calc_RX5day(df), calc_SDII(df),
        calc_TXx(df), calc_TNx(df), calc_TXn(df), calc_TNn(df), calc_DTR(df)
    ]
    
    pct_indices = calc_percentile_indices(df, temp_pct, prcp_95p, prcp_99p)
    indices.extend(pct_indices)
    
    seasonal = calc_seasonal_averages(df)
    indices.append(seasonal)
    
    for idx in indices:
        results = results.join(idx)
        
    return results.fillna(0).round(2)
