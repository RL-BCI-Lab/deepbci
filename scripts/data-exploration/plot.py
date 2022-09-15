import warnings
from pdb import set_trace

import numpy as np
import pandas as pd
import mne

import plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly import colors
from plotly.subplots import make_subplots

def get_subject_df(df, event, channel, epochs):
    """
        Return:
            A list of dataframes, one for each unique dataset.
    """
    group_data = []
    info = {}
    levels = ['group', 'dataset', 'subject']
    groups = df.groupby(level=levels)
    
    for group_name, group_df in groups:
        # Compress data into one data_utils.Data object if more than
        # one object is detected.
        d = group_df.ravel()
        if len(d) != 1:
            raise ValueError('d is more than 1')
        else:
            d = d[0]
            
        tags = group_df.index.droplevel(level='trial')[0]
        subject_df = prepare_subject_df(
            epochs=d.data, 
            event=event, 
            channel=channel, 
            epoch_counts=epochs, 
            tags=tags
        )
        # Store dataset for plotting
        for l,n in zip(levels, tags):
            subject_df[l] = n

        group_data.append(subject_df)
        
        info[tags] = d.data.info
        
    return pd.concat(group_data).set_index(levels), info

def prepare_subject_df(epochs, event, channel, tags, epoch_counts=None):
    epoch_len = len(epochs._times_readonly)
    subject_df = epochs.to_data_frame()

    # Rename condition column to event
    subject_df.rename({'condition': 'event'}, axis=1, inplace=True)
    # Get ONLY the passed events
    subject_df = subject_df[subject_df['event'].isin(map(str, event))]

    # Convert time to seconds
    subject_df.loc[:, 'time'] = subject_df.loc[:, 'time'] / 1000

    if epoch_counts is not None:
        for e in event:
            key_tuple =  tags + (e,)
            key = '-'.join(map(str, key_tuple))
            
            if key in epoch_counts:
                max_epochs = epoch_counts[key]
                event_df = subject_df[subject_df['event'] == e]
                event_epochs = event_df['epoch'].unique()
                print(type(e))
                print("test", epoch_counts, e, subject_df['event'].dtype)

                if max_epochs > len(event_epochs):
                    warnings.warn(f"max_epochs {max_epochs} is greater than total epochs {len(event_epochs)} for {key_tuple}. Using all epochs.")
                    max_epochs = len(event_epochs)
                    
                use_event_epochs = np.random.choice(event_epochs, size=max_epochs, replace=False)
                use_rows = event_df['epoch'].isin(use_event_epochs)
                drop_rows = use_rows[use_rows == False].index
                subject_df.drop(drop_rows, axis=0, inplace=True)

    # Drop unused channels
    drop_ch = list(set(epochs.ch_names).symmetric_difference(set(channel)))
    if len(drop_ch) != 0:
        subject_df.drop(drop_ch, axis=1, inplace=True, errors='ignore')

    # Reset indexes
    subject_df.reset_index(drop=True, inplace=True)

    # Melt channels into rows only leaving uV values
    keep_ch = list(set(epochs.ch_names) & set(channel))
    subject_df = pd.melt(subject_df, 
                    id_vars=['time', 'event', 'epoch'], 
                    value_vars=keep_ch,
                    var_name='channel',
                    value_name='uV')
    
    return subject_df

def combine_with_event(*args):
    for index, value in args[0].items():
        try:
            args[0][index] = str(int(value))
        except ValueError:
            args[0][index] = str(value)
            
    return '-'.join(args[0])

def mne_object_filter(data, low_pass=None, high_pass=None):
    """ Filter MNE data object given low pass and/or high pass bound
        
        MNE docs:
            https://mne.tools/0.15/generated/mne.Epochs.html#mne.Epochs.filter
            
            l_freq and h_freq are the frequencies below which and above which, 
            respectively, to filter out of the data. Thus the uses are:

            l_freq < h_freq: band-pass filter
            l_freq > h_freq: band-stop filter
            l_freq is not None and h_freq is None: high-pass filter
            l_freq is None and h_freq is not None: low-pass filter

    """

    iir_params = dict(order=2, ftype='butter', output='ba')

    data.filter(l_freq=high_pass, 
                 h_freq=low_pass, 
                 method='iir',
                 verbose=False,
                 iir_params=iir_params)
    
def mne_filter(data, sfreq, low_pass=None, high_pass=None,):
    """ Filter MNE data object given low pass and/or high pass bound
        
        MNE docs:
            https://mne.tools/0.15/generated/mne.Epochs.html#mne.Epochs.filter
            
            l_freq and h_freq are the frequencies below which and above which, 
            respectively, to filter out of the data. Thus the uses are:

            l_freq < h_freq: band-pass filter
            l_freq > h_freq: band-stop filter
            l_freq is not None and h_freq is None: high-pass filter
            l_freq is None and h_freq is not None: low-pass filter

    """

    iir_params = dict(order=2, ftype='butter', output='ba')

    return mne.filter.filter_data(
            data,
            sfreq=sfreq,
            l_freq=high_pass, 
            h_freq=low_pass,
            method='iir',
            verbose=False,
            iir_params=iir_params
    )
    
class ErrorBands():
    @staticmethod
    def standard_error(data: pd.DataFrame, n: int) -> pd.DataFrame:
        return data.std(ddof=1) / np.sqrt(n)
    @staticmethod
    def standard_deviation(data: pd.DataFrame)-> pd.DataFrame:
        return data.std(ddof=1)

def fill_multi_index(columns, values):
    cols_vals = {}
    for name, value in zip(columns, values):
        cols_vals[name] = value
    return cols_vals

def hex_to_rgba(color: str, opacity=0.15):
    rgb = colors.hex_to_rgb(color) + (opacity,)
    return f'rgba{rgb}'

class PlotAverages():
    def __init__(self, grps):
        self.grps = grps
        
    def plot_subject_average(
        self, group, dataset, subject, channel, event,
        epochs={}, mne_filter_kws=None, std=2
    ):
        event = [str(e) for e in event]
        indexes = ['group', 'dataset', 'subject', 'event', 'channel']
        avg_plots = self.plot_average(
            group, dataset, subject, channel, event, indexes,
            epochs=epochs, mne_filter_kws=mne_filter_kws
        )

        line_kws = dict(
            data_frame=avg_plots,
            x='time', 
            y='uV',
            error_y_minus='error',
            color='event',
            line_dash='channel',
            facet_row='subject',
            height=400 * len(subject),
        )
        fig = px.line(**line_kws)

        for trace in fig.data:
            se = trace.error_y['arrayminus']
            y_low = trace.y - std*se
            y_high = trace.y + std*se
            band = go.Scatter(
                name=trace.name,
                legendgroup=trace.legendgroup,
                x=list(trace.x) + list(trace.x[::-1]),
                y=list(y_high) + list(y_low[::-1]),
                fill='toself',
                fillcolor=hex_to_rgba(trace.line.color),
                line=dict(color=hex_to_rgba(trace.line.color)),
                hoverinfo="skip",
                showlegend=False,
                xaxis=trace.xaxis,
                yaxis=trace.yaxis
            )
            fig.add_trace(band)

        return fig

    def plot_grand_average(
        self, group, dataset, subject, channel, event,
        epochs={}, mne_filter_kws=None, std=2
    ):
        event = [str(e) for e in event]
        indexes = ['group', 'dataset', 'event', 'channel']
        avg_plots = self.plot_average(
            group, dataset, subject, channel, event, indexes,
            epochs=epochs, mne_filter_kws=mne_filter_kws
        )

        line_kws = dict(
            data_frame=avg_plots,
            x='time', 
            y='uV',
            error_y_minus='error',
            color='event',
            line_dash='channel',
            height=400,
        )
        fig = px.line(**line_kws)

        for trace in fig.data:
            se = trace.error_y['arrayminus']
            y_low = trace.y - std*se
            y_high = trace.y + std*se
            band = go.Scatter(
                name=trace.name,
                legendgroup=trace.legendgroup,
                x=list(trace.x) + list(trace.x[::-1]),
                y=list(y_high) + list(y_low[::-1]),
                fill='toself',
                fillcolor=hex_to_rgba(trace.line.color),
                line=dict(color=hex_to_rgba(trace.line.color)),
                hoverinfo="skip",
                showlegend=False,
                xaxis=trace.xaxis,
                yaxis=trace.yaxis
            )
            fig.add_trace(band)

        return fig

    def plot_average(
        self, group, dataset, subject, channel, event, indexes,
        epochs={}, error=None, mne_filter_kws=None, 
    ):

        if not isinstance(subject, (list, tuple)):
            subject = [subject]
        avg_plots = []
        selected_data = self.grps.get_series([group, dataset, subject])
        subjects_df, info = get_subject_df(selected_data, event, channel, epochs)
        sfreq = list(info.values())[0]['sfreq']
        subjects_df = subjects_df.reset_index()

        # Return emptry plot if there is nothing to plot
        if len(subjects_df) == 0:
            return px.line(), None
        
        compute_df = subjects_df.set_index(indexes).sort_index()
        combos = list(compute_df.index.unique())
        
        for combo in combos:
            print(f"Combo: {combo}")
            combo_mdf = compute_df.loc[[combo]]

            n_times = len(combo_mdf['time'].unique())
            n_epochs = (combo_mdf['epoch'].value_counts() / n_times).sum()
            
            combo_df = combo_mdf.reset_index()
            combo_df.set_index(['subject', 'epoch'], inplace=True)
            
            unqiue_idxs = np.unique(combo_df.index)
    #         size = epochs.get('-'.join(combo), len(unqiue_idxs))
    #         # Use all samples
    #         if size == len(unqiue_idxs):
    #             uv = combo_df.groupby('time')[['uV']]
    #         # Randomly select subset of samples
    #         else:
    #             use_samples = np.random.choice(unqiue_idxs, size=size, replace=False)
    #             uv = combo_df.loc[use_samples].groupby('time')[['uV']]
                
            print(f"\tEpoch count: {len(unqiue_idxs)}")
            uv = combo_df.groupby('time')[['uV']]
            # Compute mean
            uv_means = uv.mean().reset_index()

            columns = fill_multi_index(compute_df.index.names, combo)
            
            uv_means = uv_means.assign(**columns)
            uv_means = uv_means.set_index(compute_df.index.names)
            
            # Compute standard error
            error = ErrorBands.standard_error(uv, n_epochs)
            uv_means['error'] = error['uV'].values
            
            avg_plots.append(uv_means)
            
        avg_plots = pd.concat(avg_plots)

        if mne_filter_kws is not None:
            # Check if both parms are none, if so do NOT run filter as error will be thrown.
            if mne_filter_kws['high_pass'] is not None or mne_filter_kws['low_pass'] is not None:
                filter_grps = avg_plots.groupby(level=indexes)
                for multi_idx, grp_df in filter_grps:
                    # Extract sampling rate (i.e., sfreq) from MNE meta data
                    kwargs = dict(sfreq=sfreq, **mne_filter_kws)
                    filtered_uv = mne_filter(grp_df['uV'].values, **kwargs)
                    # Update current avg plot group with filtered uV values
                    avg_plots.loc[multi_idx, 'uV'] = filtered_uv
                avg_plots.reset_index(inplace=True)
                
        avg_plots = avg_plots.reset_index()
        avg_plots.loc[:, 'event'] = avg_plots[['dataset', 'event']].agg('-'.join, axis=1)
        
        return avg_plots
        

    def plot_subject_grand_average(
        self, group, dataset, subject, channel, event, 
        epochs={}, mne_filter_kws=None,
    ):
        # TODO: https://stackoverflow.com/questions/67179007/how-to-add-a-line-plot-on-top-of-a-stacked-bar-plot-in-plotly-express-in-python
        event = [str(e) for e in event]
        if not isinstance(subject, (list, tuple)):
            subject = [subject]

        print([group, dataset, subject, channel, event])
        selected_data = self.grps.get_series([group, dataset, subject])

        plot_df, info = get_subject_df(selected_data, event, channel, epochs)
        plot_df = plot_df.reset_index()
        
        # Randomly select info as all sfreq should be the same if average is being taken!
        info = list(info.values())[0]
        
        # Return emptry plot if there is nothing to plot
        if len(plot_df) == 0:
            return px.line(), None

        # Group by time, event, channel and subject then average all unqiue pairs to get an average uV.
        groupby = ['group', 'dataset', 'subject', 'time', 'event', 'channel']
        sa_df =  plot_df.groupby(groupby).mean().reset_index()
        sa_df.loc[:, 'event'] = sa_df[['dataset', 'event']].agg('-'.join, axis=1)
        
        # Group by time, event, and channel then average all unqiue pairs to get an average uV.
        groupby = ['group', 'dataset', 'time', 'event', 'channel']
        ga_df = plot_df.groupby(groupby).mean(numeric_only=True).reset_index()
        ga_df.loc[:, 'event'] = ga_df[['dataset', 'event']].agg('-'.join, axis=1)
        ga_df['subject'] = 'ga'
        
        saga_df = pd.concat([ga_df, sa_df], sort=True)
        saga_df.loc[:, 'event'] = saga_df[['subject', 'event']].agg('-'.join, axis=1)
        saga_df.drop(['epoch'], axis=1, inplace=True)
        
        # Filter averaged data
        if mne_filter_kws is not None:
            # Check if both parms are none, if so do NOT run filter as error will be thrown.
            if mne_filter_kws['high_pass'] is not None or mne_filter_kws['low_pass'] is not None:
                saga_df = saga_df.set_index(['group', 'dataset', 'subject', 'channel', 'event']).sort_index()
                filter_grps = saga_df.groupby(level=['group', 'dataset', 'subject', 'channel', 'event'])
                for n, grp_df in filter_grps:
                    kwargs = dict(sfreq=info['sfreq'], **mne_filter_kws)
                    filtered_uv = mne_filter(grp_df['uV'].values, **kwargs)
                    grp_df.loc[:, 'uV'] = filtered_uv
                    saga_df.loc[n] = grp_df
                saga_df.reset_index(inplace=True)
        
        # Create plot
        line_kws = dict(
            data_frame=saga_df,
            x='time', 
            y='uV',
            color='event',
            line_dash='channel',
            height=400
        )
        fig = px.line(**line_kws)
        
        # Update plot
        fig.for_each_trace(
            lambda trace: trace.update(line=dict(color='black', width=4,)) if 'ga' in trace.name else ()
        )

        # Output Event count
        for (dataset, event), ds_df in plot_df.groupby(['dataset', 'event']):
            print('Dataset: {} Event: {}'.format(dataset, event))
            ds_total = 0
            for subject, sub_df in ds_df.groupby('subject'):
                samples = sub_df[sub_df['event']==event].epoch.nunique()
                ds_total += samples
                print("\tSubject: {} Count: {}".format(subject, samples))
            print('\tTotal: {}'.format(ds_total))
            
        return fig, plot_df

