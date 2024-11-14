#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for making different plots for the bid-based LDRD project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime as datetime
import glob
import os

def str_to_pd_resample(sample_interval):
    """ Converts a string in ['day', 'hour', None] to pandas resampling option """
    assert sample_interval in ['day', 'hour', None]
    if 'day' in sample_interval.lower():
        si = "D"
    elif 'hour' in sample_interval.lower():
        si = 'H'
    else:
        si = None
    return si

def str_to_datetime(times):
    ''' YYYYmmddHHMM '''
    try:
        datetimes = [datetime.datetime.strptime(t, '%Y%m%d%H%M') for t in times]
    except:
        datetimes = pd.to_datetime(times)
    return datetimes

def group_by_day(times, values):
    ''' Sum values over each day and return values '''
    times = str_to_datetime(times)
    days = 0
    this_day = None
    out_values = list()
    # TODO: re-do with masking and array slicing
    for i, t in enumerate(times):
        if t.day != this_day:
            days += 1
            this_day = t.day
            out_values.append(0)
        out_values[days - 1] += values[i]
    return np.array(out_values)

class SurplusPlotter:
    
    def __init__(self, dirlist, mkt_fname='market_summary_TSRTM.csv', sample_interval='day',
                 stype='surplus_phys', mode='percent'):
        """ Initialize with a list of saved directories to use for plotting
        Each directory may contain several market surplus files
        Requires the same files in each directory
        """
        # Load inputs into class variables
        if type(dirlist) == str: # Handling for a single directory input
            dirlist = [dirlist]
        self.find_surplus_types(dirlist[0])
        # Type of surplus metric (from market_summary_TSRTM.csv)
        assert stype in ['surplus_phys', 'surplus_minus_str']
        self.stype = stype
        self.dirlist = dirlist
        self.mkt_fname = mkt_fname
        self.si = str_to_pd_resample(sample_interval)
        assert mode in ['percent', 'absolute']
        self.mode = mode
        # None defaults for class attributes
        self.surplus_types = None
        self.timeseries = None
        self.baseline_surplus = None
        self.option_surplus = None
        self.surplus_diff = None
        # Load data
        if self.mode == 'percent':
            self.scale = 1
        else:
            self.scale = 1000
        self.get_differences()
        self.os_stats = self.get_median_err()
        
    def find_surplus_types(self, firstdir):
        """ Checks for the different surplus types and loads them into a list """
        surplus_files = glob.glob(os.path.join(firstdir, 'market_summary*csv'))
        self.surplus_types = ['.'.join(fname.split('.')[:-1]) for fname in surplus_files]
        self.surplus_types = [fname.split('_')[-1] for fname in self.surplus_types]

    def get_baseline_surplus(self, baseline_dir='bbm_baseline'):
        """ Loads the baseline physical surplus and timeseries """
        root = os.path.split(self.dirlist[0])[0]
        surplus_df = pd.read_csv(os.path.join(root, baseline_dir, self.mkt_fname), index_col='time')
        surplus_df.index = pd.to_datetime(surplus_df.index, format='%Y%m%d%H%M')
        # Resample
        if self.si is not None:
            surplus_df = surplus_df.resample(self.si).sum()
        # Timeseries and surplus
        self.timeseries = surplus_df.index.values
        self.baseline_surplus = surplus_df[self.stype].values
        # If using surplus_minus_str, include degradation cost as storage surplus estimate
        if self.stype == 'surplus_minus_str':
            deg_df = pd.read_csv(os.path.join(root, baseline_dir, 'degradation_cost.csv'), index_col='Times')
            deg_cost = -deg_df.sum(axis=1).values # Apply a negative since this is a loss
            if len(self.baseline_surplus) != len(deg_cost): # Baseline gets 1 extra day sometimes
                self.baseline_surplus = self.baseline_surplus[:len(deg_cost)]
            self.baseline_surplus += deg_cost

    def get_option_surplus(self):
        """ Saves a dictionary to the object with the surplus for different options """
        self.option_surplus = dict()
        for directory in self.dirlist:
            if 'option' in directory:
                bname = os.path.basename(directory)
                key = '_'.join(bname.split('_')[1:])  # converts bbm_option_N -> option_N

                df = pd.read_csv(os.path.join(directory, self.mkt_fname), index_col='time')
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')
                # Resample
                if self.si is not None:
                    df = df.resample(self.si).sum()
                self.option_surplus[key] = df[self.stype].values
                # If using surplus_minus_str, include degradation cost as storage surplus estimate
                if self.stype == 'surplus_minus_str':
                    deg_df = pd.read_csv(os.path.join(directory, 'degradation_cost.csv'), index_col='Times')
                    deg_cost = -deg_df.sum(axis=1).values  # Apply a negative since this is a loss
                    if len(self.option_surplus[key]) != len(deg_cost):  # Baseline gets 1 extra day sometimes
                        self.option_surplus[key] = self.option_surplus[key][:len(deg_cost)]
                    self.option_surplus[key] += deg_cost

    def get_median_err(self, low=16, high=84, dec=2):
        """ Computes the median and low/high error """
        os_stats = {}
        for key, values in self.surplus_diff.items():
            median = np.median(values)
            # Compute low/high values. Convert to difference from median for saving
            value_sort = np.sort(values)
            low_range = value_sort[int(low / 100 * len(value_sort))]
            high_range = value_sort[int(high / 100 * len(value_sort))]
            low_err = median - low_range
            high_err = high_range - median
            scale = self.scale
            median = round(median/scale, dec)
            low_err = round(low_err/scale, dec)
            high_err = round(high_err/scale, dec)
            stat_dict = {'median':median, 'low_err':low_err, 'high_err':high_err}
            os_stats[key] = stat_dict
        return os_stats

    def get_differences(self):
        """ Loads baseline and options and makes a dictionary of the differences """
        # First load baseline and options surplus
        self.get_baseline_surplus()
        self.get_option_surplus()
        # Now compute differences
        self.surplus_diff = dict()
        for key, opt_surplus in self.option_surplus.items():
            if self.mode == 'percent':
                self.surplus_diff[key] = (opt_surplus - self.baseline_surplus)/self.baseline_surplus * 100
            elif self.mode == 'absolute':
                self.surplus_diff[key] = (opt_surplus - self.baseline_surplus)/self.scale

    def add_stat_text(self, ax, labels, incl_errs=False):
        """ Adds the stat summaries to the plot """
        xvals = ax.get_xticks()
        ylims = ax.get_ylim()
        yval = ylims[1] - 0.05*(ylims[1] - ylims[0])
        if self.mode == 'percent':
            unit, extra = '%', ''
        elif self.mode == 'absolute':
            unit, extra = 'k', '$'
        for i, xval in enumerate(xvals):
            # Load the relevant stat dict (e.g., mode = 'income', labels[i] = 'baseline')
            stat_dict = self.os_stats[labels[i]]
            if incl_errs:
                stat_text = f'{extra}{stat_dict["median"]}{unit}$^{{{stat_dict["high_err"]}}}_{{-{stat_dict["low_err"]}}}$'
            else:
                stat_text = f'{extra}{stat_dict["median"]}{unit}'
            ax.text(xval, yval, stat_text, ha='center', va='center', fontsize=12)

    def plot_differences(self, show=False):
        """ Creates a box and whisker plot of the surplus differences"""
        # Ensure the differences have been calculated
        if self.surplus_diff is None:
            self.get_differences()
        # Configure styles
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        fig, ax = plt.subplots()
        values = [v for v in self.surplus_diff.values()]
        xtick_labels = [' '.join(k.split('_')).capitalize() for k in self.surplus_diff.keys()]
        bplot = ax.boxplot(values, patch_artist=True, labels=xtick_labels)
        # Colors
        for patch, color, in zip(bplot['boxes'], sns.color_palette('hls', len(xtick_labels))):
            patch.set_facecolor(color)
        if self.mode == 'percent':
            ylabel = 'Percentage (%)'
        elif self.mode == 'absolute':
            ylabel = 'Thousand Dollars ($)'
        ax.set_ylabel(ylabel, fontsize=14)
        plt.ylim(-0.5, 0.1)
        if self.scale == 1000:
            # Add 'k' to the tick label (thousands of dollars)
            yticklabels = ax.get_yticklabels()
            for ytick in yticklabels:
                ytick.set_text(f'{ytick.get_text()}k')
            ax.set_yticklabels(yticklabels)
        self.add_stat_text(ax, list(self.surplus_diff.keys()))
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/surplus_bw_{self.mode}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

class RevenuePlotter:

    def __init__(self, dirlist, new_profits=False, sample_interval='day'):
        # Initialize class variables
        self.dirlist = dirlist
        self.si = str_to_pd_resample(sample_interval)
        self.products = ['EN', 'RGU', 'RGD', 'SPR', 'NSP', 'DEG']
        self.scale = 1000 # Scaling on dollars
        assert self.scale in [1, 1000]
        # Create/load profit summary spreadsheets
        self.get_profit_summary(new_profits=new_profits)

    def save_summary(self, save_dict, save_times):
        """ Saves the profit summary to an excel file """
        prods = ['revenue', 'degradation']
        data = np.zeros((len(save_times), len(save_dict) + 1), dtype=object)
        data[:, 0] = save_times
        dcols = ['time'] + list(save_dict.keys())
        with pd.ExcelWriter(f'daily_profit_summary.xlsx') as writer:
            for prod in prods:
                for i, mkt in enumerate(save_dict.keys()):
                    if prod == 'revenue':
                        data[:, i + 1] = save_dict[mkt]['EN'] + \
                                         save_dict[mkt]['RGU'] + \
                                         save_dict[mkt]['RGD'] + \
                                         save_dict[mkt]['SPR'] + \
                                         save_dict[mkt]['NSP']
                    else:
                        data[:, i + 1] = save_dict[mkt]['DEG']
                df = pd.DataFrame(data, columns=dcols)
                df.to_excel(writer, sheet_name=prod, index=False)

    def create_profit_summary(self):
        """ Creates a profit summary based on the input directories """
        # Pids and products to match profit summary
        pids = ['p00001', 'p00002', 'p00003', 'p00004', 'p00005', 'p00006', 'p00007', 'p00008',
                'p00009', 'p00010', 'p00011', 'p00012']

        product_profits = dict()
        tunit = 'day'
        times = None
        for directory in self.dirlist:
            dbase = os.path.basename(directory)
            key = '_'.join(dbase.split('_')[1:])  # converts bbm_option_N -> option_N
            product_profits[key] = dict()
            for product in self.products:
                for pid in pids:
                    try: #Try/except captures both new and legacy naming convention
                        path = os.path.join(directory,f'profit_{pid}_total.csv')
                        df = pd.read_csv(path, index_col='time')
                    except FileNotFoundError:
                        path = os.path.join(directory, f'profit_{pid}.csv')
                        df = pd.read_csv(path, index_col='time')
                    if 'dccommit' in directory:
                        df.index = pd.to_datetime(df.index, format='%m/%d/%y %H:%M')
                    else:
                        df.index = pd.to_datetime(df.index)
                    df = df.resample(self.si).sum()
                    # Save times (just once)
                    if times is None:
                        times = df.index[:-1]
                    cols = df.columns
                    cidx = None
                    for i, c in enumerate(cols):
                        if product in c:
                            cidx = i
                    if cidx is None:
                        raise KeyError(f"Product {product} not found in column headers {cols}")
                    prod_vals = df[cols[cidx]].values
                    # Adjust lengths - drop last day for non-deg, drop 1st day for deg
                    if 'dccommit' not in directory: # Latest update to scheduler does not include extra DAM
                        if product == 'DEG':
                            prod_vals = prod_vals[1:]
                        else:
                            prod_vals = prod_vals[:-1]
                    print(f"For directory {os.path.basename(directory)} found {len(prod_vals)} {product}")
                    if product in product_profits[key].keys():
                        product_profits[key][product] += prod_vals
                    else:
                        product_profits[key][product] = prod_vals
        self.save_summary(product_profits, times)

    def get_median_err(self, values, low, high, dec=1):
        """ Computes the median and low/high error """
        median = np.median(values)
        # Compute low/high values. Convert to difference from median for saving
        value_sort = np.sort(values)
        low_range = value_sort[int(low / 100 * len(value_sort))]
        high_range = value_sort[int(high / 100 * len(value_sort))]
        low_err = median - low_range
        high_err = high_range - median
        scale = self.scale
        median = round(median/scale, dec)
        low_err = round(low_err/scale, dec)
        high_err = round(high_err/scale, dec)
        stat_dict = {'median':median, 'low_err':low_err, 'high_err':high_err}
        return stat_dict

    def calc_summary_stats(self, low=16, high=84):
        """ Computes median and low/high bounds for each of revenue, deg, and net """
        self.ps_stats = {'revenue':{}, 'degradation':{}, 'net':{}}
        for key in self.ps_stats.keys():
            if key == 'net':
                cols = self.ps['revenue'].columns
                for col in cols:
                    if col == 'time':
                        continue
                    values = self.ps['revenue'][col].values + self.ps['degradation'][col].values
                    stat_dict = self.get_median_err(values, low, high)
                    self.ps_stats[key][col] = stat_dict
            else:
                cols = self.ps[key].columns
                for col in cols:
                    if col == 'time':
                        continue
                    values = self.ps[key][col].values
                    stat_dict = self.get_median_err(values, low, high)
                    self.ps_stats[key][col] = stat_dict

    def get_profit_summary(self, new_profits=False):
        """ Checks for or creates profit summary and loads """
        if not os.path.exists('daily_profit_summary.xlsx') or new_profits:
            self.create_profit_summary()
        self.ps = pd.read_excel('daily_profit_summary.xlsx', sheet_name=None)
        self.calc_summary_stats()

    def add_stat_text(self, ax, mode, labels, incl_errs=False, pos='top'):
        """ Adds the stat summaries to the plot """
        assert pos in ['top', 'bottom']
        xvals = ax.get_xticks()
        ylims = ax.get_ylim()
        if pos == 'bottom':
            yval = ylims[0] + 0.1*(ylims[1] - ylims[0])
        elif pos == 'top':
            yval = ylims[1] - 0.05*(ylims[1] - ylims[0])
        for i, xval in enumerate(xvals):
            # Load the relevant stat dict (e.g., mode = 'revenue', labels[i] = 'baseline')
            stat_dict = self.ps_stats[mode][labels[i]]
            if incl_errs:
                stat_text = f'${stat_dict["median"]}k$^{{{stat_dict["high_err"]}}}_{{-{stat_dict["low_err"]}}}$'
            else:
                stat_text = f'${stat_dict["median"]}k'
            ax.text(xval, yval, stat_text, ha='center', va='center', fontsize=12)

    def make_boxplot(self, data, mode, show=False):
        """ Creates and styles boxplots """
        # Configure styles
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        # Make figure and add each column to the plot
        fig, ax = plt.subplots()
        values = [data.values[:,i]/self.scale for i in range(data.values.shape[1]) if i != 0]
        labels = data.columns[1:]
        xtick_labels = [' '.join(l.split('_')).capitalize() for l in labels] # Remove underscore, capitalize
        bplot = ax.boxplot(values, patch_artist=True, labels=xtick_labels)
        # Colors
        for patch, color, in zip(bplot['boxes'], sns.color_palette('hls', len(labels))):
            patch.set_facecolor(color)
        ax.set_ylabel('Dollars ($)', fontsize=14)
        if mode == 'revenue':
            plt.ylim(-250, 150)
        elif mode == 'net':
            plt.ylim(-800, 0)
        if self.scale == 1000:
            # Add 'k' to the tick label (thousands of dollars)
            yticklabels = ax.get_yticklabels()
            for ytick in yticklabels:
                ytick.set_text(f'{ytick.get_text()}k')
            ax.set_yticklabels(yticklabels)
        pos = 'bottom'
        if mode == 'net':
            pos = 'top'
        self.add_stat_text(ax, mode, labels, pos=pos)
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/profit_{mode}.png', bbox_inches='tight')
        if show:
            plt.title(f"Average Daily {mode.capitalize()}")
            plt.show()
        plt.close(fig)

    def plot_profits(self, show=False):
        """ Plots profit summary for revenue, degradation, and sum """
        modes = ['revenue', 'degradation', 'net']
        for mode in modes:
            if mode == 'net':
                net_revenue = self.ps['revenue']
                # Add degradation to revenue, skipping 1st column (times)
                net_revenue.iloc[:,1:] += self.ps['degradation'].iloc[:,1:]
                self.make_boxplot(net_revenue, mode, show=show)
            else:
                self.make_boxplot(self.ps[mode], mode, show=show)

if __name__ == '__main__':
    id_str = 'bbm'
    # Gets all directories in scenarios folder with 'bbm' in name
    dirs = glob.glob(os.path.join('../scenarios', f'{id_str}*'))
    dirs = sorted(dirs)
    # Convert to absolute paths
    dirs = [os.path.abspath(d) for d in dirs]
    # dirs = ['../scenarios/bbm_baseline', '../scenarios/bbm_option_2', '../scenarios/bbm_option_3',
    #         '../scenarios/bbm_option_5']
    splotter = SurplusPlotter(dirs, sample_interval='day', stype='surplus_minus_str')
    splotter.plot_differences(show=True)
    rplotter = RevenuePlotter(dirs, new_profits=True)
    rplotter.plot_profits(show=True)