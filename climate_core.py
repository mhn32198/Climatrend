import pandas as pd
import numpy as np

def load_data(file_obj):
    filename = getattr(file_obj, 'name', '').lower()
    if filename.endswith('.txt'):
        df = pd.read_csv(file_obj, sep=r'\s+', engine='python')
    else:
        df = pd.read_csv(file_obj)

    df.columns = [col.strip().upper() for col in df.columns]

    # --- Format Handling: Support both DATE-only and YEAR/MONTH/DAY columns ---
    has_ymd = all(c in df.columns for c in ['YEAR', 'MONTH', 'DAY'])
    has_date = 'DATE' in df.columns

    if has_date and not has_ymd:
        # Single DATE column: parse and extract YEAR, MONTH, DAY
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['YEAR']  = df['DATE'].dt.year.astype('int16')
        df['MONTH'] = df['DATE'].dt.month.astype('int16')
        df['DAY']   = df['DATE'].dt.day.astype('int16')
    elif has_ymd:
        # Separate columns: build DATE from them
        for col in ['YEAR', 'MONTH', 'DAY']:
            df[col] = df[col].astype('int16')
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    else:
        raise ValueError("CSV must contain either a 'DATE' column or 'YEAR', 'MONTH', 'DAY' columns.")

    required_cols = ['PRCP', 'TMAX', 'TMIN']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # DOWNCASTING for memory efficiency
    for col in ['PRCP', 'TMAX', 'TMIN']:
        df[col] = df[col].astype('float32')

    # Sort by date and build CALENDAR_DAY for percentile logic
    df = df.sort_values('DATE').reset_index(drop=True)
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
    tmax_10 = df.groupby('CALENDAR_DAY')['TMAX'].quantile(0.10).rename('TMAX_10p')
    tmax_90 = df.groupby('CALENDAR_DAY')['TMAX'].quantile(0.90).rename('TMAX_90p')
    tmax_95 = df.groupby('CALENDAR_DAY')['TMAX'].quantile(0.95).rename('TMAX_95p')
    tmin_10 = df.groupby('CALENDAR_DAY')['TMIN'].quantile(0.10).rename('TMIN_10p')
    tmin_90 = df.groupby('CALENDAR_DAY')['TMIN'].quantile(0.90).rename('TMIN_90p')
    temp_pct = pd.concat([tmax_10, tmax_90, tmax_95, tmin_10, tmin_90], axis=1)
    
    wet_days = df[df['PRCP'] >= 1]['PRCP']
    prcp_95p = wet_days.quantile(0.95) if not wet_days.empty else 0
    prcp_99p = wet_days.quantile(0.99) if not wet_days.empty else 0
    
    return temp_pct, prcp_95p, prcp_99p

# --- THRESHOLD COUNTS ---
def calc_FD(df, thresh_fd=0, freq=['YEAR'], name='FD'): return df[df['TMIN'] < thresh_fd].groupby(freq).size().rename(name)
def calc_SU(df, thresh_su=25, freq=['YEAR'], name='SU'): return df[df['TMAX'] > thresh_su].groupby(freq).size().rename(name)
def calc_ID(df, thresh_id=0, freq=['YEAR'], name='ID'): return df[df['TMAX'] < thresh_id].groupby(freq).size().rename(name)
def calc_TR(df, thresh_tr=20, freq=['YEAR'], name='TR'): return df[df['TMIN'] > thresh_tr].groupby(freq).size().rename(name)
def calc_Rnn(df, thresh_rnn=30, freq=['YEAR']): return df[df['PRCP'] >= thresh_rnn].groupby(freq).size().rename('Rnn')
def calc_R10mm(df, freq=['YEAR']): return df[df['PRCP'] >= 10].groupby(freq).size().rename('R10mm')
def calc_R20mm(df, freq=['YEAR']): return df[df['PRCP'] >= 20].groupby(freq).size().rename('R20mm')
def calc_R1mm(df, freq=['YEAR']): return df[df['PRCP'] >= 1].groupby(freq).size().rename('R1mm')

def calc_GSL(df, hemisphere='Northern', freq=['YEAR']):
    df_gsl = df.copy()
    df_gsl['TG'] = (df_gsl['TMAX'] + df_gsl['TMIN']) / 2.0
    
    if freq == ['YEAR', 'MONTH']:
        warm_days = df_gsl['TG'] > 5
        return warm_days.groupby([df_gsl['YEAR'], df_gsl['MONTH']]).sum().rename('GSL')
        
    if hemisphere == 'Southern':
        df_gsl['GSL_YEAR'] = df_gsl.apply(
            lambda row: row['YEAR'] if row['MONTH'] >= 7 else row['YEAR'] - 1, axis=1
        )
    else:
        df_gsl['GSL_YEAR'] = df_gsl['YEAR']
        
    gsl_results = {}
    for year, group in df_gsl.groupby('GSL_YEAR'):
        group = group.sort_values('DATE').reset_index(drop=True)
        warm_spells = (group['TG'] > 5).rolling(window=6).sum() == 6
        start_idx = None
        if warm_spells.any():
            start_idx = warm_spells.idxmax() - 5
            
        end_idx = None
        if start_idx is not None:
            threshold_date = pd.Timestamp(year=int(year), month=7, day=1) if hemisphere == 'Northern' else pd.Timestamp(year=int(year + 1), month=1, day=1)
            valid_dates_mask = group['DATE'] > threshold_date
            valid_idx_mask = group.index >= (start_idx + 6)
            cold_spells = (group['TG'] < 5).rolling(window=6).sum() == 6
            valid_cold_spells = cold_spells & valid_dates_mask & valid_idx_mask
            if valid_cold_spells.any():
                end_idx = valid_cold_spells.idxmax() - 5
                
        if start_idx is not None and end_idx is not None:
            gsl_length = end_idx - start_idx
        elif start_idx is not None and end_idx is None:
            gsl_length = len(group) - start_idx
        else:
            gsl_length = 0
            
        gsl_results[year] = gsl_length
    series = pd.Series(gsl_results, name='GSL')
    series.index.name = 'YEAR'
    return series

# --- PERCENTILE INDICES ---
def calc_percentile_indices(df, temp_pct, prcp_95p, prcp_99p, freq=['YEAR']):
    df_pct = df.copy()
    df_pct = df_pct.merge(temp_pct, left_on='CALENDAR_DAY', right_index=True, how='left')
    
    valid_tmax = df_pct.groupby(freq)['TMAX'].count()
    valid_tmin = df_pct.groupby(freq)['TMIN'].count()
    
    tn10p = (df_pct[df_pct['TMIN'] < df_pct['TMIN_10p']].groupby(freq).size() / valid_tmin * 100).rename('TN10p')
    tx10p = (df_pct[df_pct['TMAX'] < df_pct['TMAX_10p']].groupby(freq).size() / valid_tmax * 100).rename('TX10p')
    tn90p = (df_pct[df_pct['TMIN'] > df_pct['TMIN_90p']].groupby(freq).size() / valid_tmin * 100).rename('TN90p')
    tx90p = (df_pct[df_pct['TMAX'] > df_pct['TMAX_90p']].groupby(freq).size() / valid_tmax * 100).rename('TX90p')
    
    def count_spell_days(series):
        if len(series) == 0: return 0
        cumsum = (~series).cumsum()
        spell_lengths = series.groupby(cumsum).transform('sum')
        return int((series & (spell_lengths >= 6)).sum())
    
    df_pct['IS_WSDI'] = df_pct['TMAX'] > df_pct['TMAX_90p']
    df_pct['IS_CSDI'] = df_pct['TMIN'] < df_pct['TMIN_10p']
    
    wsdi = df_pct.groupby(freq)['IS_WSDI'].apply(count_spell_days).rename('WSDI')
    csdi = df_pct.groupby(freq)['IS_CSDI'].apply(count_spell_days).rename('CSDI')
    
    r95p = df_pct[df_pct['PRCP'] > prcp_95p].groupby(freq)['PRCP'].sum().rename('R95p')
    r99p = df_pct[df_pct['PRCP'] > prcp_99p].groupby(freq)['PRCP'].sum().rename('R99p')
    
    is_hw = df_pct['TMAX'] > df_pct['TMAX_95p']
    hw_spell_lengths = is_hw.groupby((~is_hw).cumsum()).transform('sum')
    df_pct['IS_HW_DAY'] = is_hw & (hw_spell_lengths >= 3)
    
    hwdi = df_pct.groupby(freq)['IS_HW_DAY'].sum().rename('HWDI')
    
    df_pct['HW_EXCESS'] = (df_pct['TMAX'] - df_pct['TMAX_95p']).where(df_pct['IS_HW_DAY'], 0)
    hwi = df_pct.groupby(freq)['HW_EXCESS'].sum().rename('HWI')
    
    return [tn10p, tx10p, tn90p, tx90p, wsdi, csdi, r95p, r99p, hwdi, hwi]

# --- DURATION / INTENSITY ---
def calc_CDD(df, freq=['YEAR']):
    def max_consecutive_dry(series):
        is_dry = series < 1
        return is_dry.groupby((~is_dry).cumsum()).sum().max() if not is_dry.empty else 0
    return df.groupby(freq)['PRCP'].apply(max_consecutive_dry).rename('CDD')

def calc_CWD(df, freq=['YEAR']):
    def max_consecutive_wet(series):
        is_wet = series >= 1
        return is_wet.groupby((~is_wet).cumsum()).sum().max() if not is_wet.empty else 0
    return df.groupby(freq)['PRCP'].apply(max_consecutive_wet).rename('CWD')

def calc_PRCPTOT(df, freq=['YEAR']): return df[df['PRCP'] >= 1].groupby(freq)['PRCP'].sum().rename('PRCPTOT')
def calc_RX1day(df, freq=['YEAR']): return df.groupby(freq)['PRCP'].max().rename('RX1day')
def calc_RX5day(df, freq=['YEAR']): return df.groupby(freq)['PRCP'].apply(lambda x: x.rolling(5).sum().max()).rename('RX5day')
def calc_SDII(df, freq=['YEAR']):
    wet_days = df[df['PRCP'] >= 1].groupby(freq)['PRCP']
    return (wet_days.sum() / wet_days.count()).rename('SDII')

# --- ABSOLUTE EXTREMES ---
def calc_TXx(df, freq=['YEAR']): return df.groupby(freq)['TMAX'].max().rename('TXx')
def calc_TNx(df, freq=['YEAR']): return df.groupby(freq)['TMIN'].max().rename('TNx')
def calc_TXn(df, freq=['YEAR']): return df.groupby(freq)['TMAX'].min().rename('TXn')
def calc_TNn(df, freq=['YEAR']): return df.groupby(freq)['TMIN'].min().rename('TNn')
def calc_DTR(df, freq=['YEAR']):
    df_temp = df.copy()
    df_temp['DTR'] = df_temp['TMAX'] - df_temp['TMIN']
    return df_temp.groupby(freq)['DTR'].mean().rename('DTR')

# --- SEASONAL AVERAGES ---
def calc_seasonal_averages(df, freq=['YEAR']):
    df_seas = df.copy()
    mam = df_seas[df_seas['MONTH'].isin([3, 4, 5])].groupby(freq)['TMAX'].mean().rename('MAM_TMAX_Ave')
    jjas = df_seas[df_seas['MONTH'].isin([6, 7, 8, 9])].groupby(freq)['PRCP'].mean().rename('JJAS_PRCP_Ave')
    
    df_seas['DJF_YEAR'] = df_seas.apply(lambda row: row['YEAR'] + 1 if row['MONTH'] == 12 else row['YEAR'], axis=1)
    
    if freq == ['YEAR', 'MONTH']:
        freq_djf = ['DJF_YEAR', 'MONTH']
        djf = df_seas[df_seas['MONTH'].isin([12, 1, 2])].groupby(freq_djf)['TMAX'].mean()
        djf.index.names = ['YEAR', 'MONTH']
        djf = djf.rename('DJF_TMAX_Ave')
    else:
        djf = df_seas[df_seas['MONTH'].isin([12, 1, 2])].groupby('DJF_YEAR')['TMAX'].mean().rename('DJF_TMAX_Ave')
        
    return pd.concat([mam, jjas, djf], axis=1)

# --- MAIN ORCHESTRATOR ---
def calculate_all_indices(df, hemisphere='Northern', thresh_fd=0, thresh_su=25, thresh_id=0, thresh_tr=20, thresh_rnn=30, resolution='Annual'):
    freq = ['YEAR'] if resolution == 'Annual' else ['YEAR', 'MONTH']
    if resolution == 'Annual':
        years = sorted(df['YEAR'].unique())
        results = pd.DataFrame(index=years)
        results.index.name = 'Year'
    else:
        idx = df.groupby(freq).size().index
        results = pd.DataFrame(index=idx)
    
    temp_pct, prcp_95p, prcp_99p = calculate_percentiles(df)
    indices = [
        calc_FD(df, 0, freq, 'FD0'), calc_SU(df, 25, freq, 'SU25'), calc_TR(df, 20, freq, 'TR20'), calc_ID(df, 0, freq, 'ID0'),
        calc_FD(df, thresh_fd, freq, 'FD_user'), calc_SU(df, thresh_su, freq, 'SU_user'), calc_TR(df, thresh_tr, freq, 'TR_user'), calc_ID(df, thresh_id, freq, 'ID_user'),
        calc_GSL(df, hemisphere, freq),
        calc_Rnn(df, thresh_rnn, freq), calc_R10mm(df, freq), calc_R20mm(df, freq), calc_R1mm(df, freq),
        calc_CDD(df, freq), calc_CWD(df, freq), calc_PRCPTOT(df, freq), calc_RX1day(df, freq), calc_RX5day(df, freq), calc_SDII(df, freq),
        calc_TXx(df, freq), calc_TNx(df, freq), calc_TXn(df, freq), calc_TNn(df, freq), calc_DTR(df, freq)
    ]
    pct_indices = calc_percentile_indices(df, temp_pct, prcp_95p, prcp_99p, freq)
    indices.extend(pct_indices)
    
    seasonal = calc_seasonal_averages(df, freq)
    indices.append(seasonal)
        
    for idx in indices:
        if idx is not None and not idx.empty: results = results.join(idx)
        
    if resolution == 'Monthly':
        results.index = [f"{y}-{m:02d}" for y, m in results.index]
        results.index.name = 'Year-Month'
        
    return results.fillna(0).round(2)
