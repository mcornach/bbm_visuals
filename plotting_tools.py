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
                 stype='surplus_phys'):
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
        # None defaults for class attributes
        self.surplus_types = None
        self.timeseries = None
        self.baseline_surplus = None
        self.option_surplus = None
        self.surplus_diff = None
        # Load data
        self.get_differences()
        
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
                key = '_'.join(directory.split('_')[1:])  # converts bbm_option_N -> option_N

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

    def get_differences(self):
        """ Loads baseline and options and makes a dictionary of the differences """
        # First load baseline and options surplus
        self.get_baseline_surplus()
        self.get_option_surplus()
        # Now compute differences
        self.surplus_diff = dict()
        for key, opt_surplus in self.option_surplus.items():
            self.surplus_diff[key] = (opt_surplus - self.baseline_surplus)/self.baseline_surplus * 100

    def plot_differences(self, show=False):
        """ Creates a box and whisker plot of the surplus differences"""
        # Ensure the differences have been calculated
        if self.surplus_diff is None:
            self.get_differences()
        # Configure styles
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        fig = plt.figure()
        ax = plt.gca()
        for option, data in self.surplus_diff.items():
            sns.boxplot(y=data, ax=ax)
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/surplus_bw.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

class RevenuePlotter:

    def __init__(self, dirlist, new_profits=False, sample_interval='day'):
        # Initialize class variables
        self.dirlist = dirlist
        self.si = str_to_pd_resample(sample_interval)
        self.products = ['EN', 'RGU', 'RGD', 'SPR', 'NSP', 'DEG']
        # Create/load profit summary spreadsheets
        self.get_profit_summary(new_profits=new_profits)

    def save_summary(self, save_dict, save_times):
        """ Saves the profit summary to an excel file """
        prods = ['income', 'degradation']
        data = np.zeros((len(save_times), len(save_dict) + 1), dtype=object)
        data[:, 0] = save_times
        dcols = ['time'] + list(save_dict.keys())
        with pd.ExcelWriter(f'daily_profit_summary.xlsx') as writer:
            for prod in prods:
                for i, mkt in enumerate(save_dict.keys()):
                    if prod == 'income':
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
            key = '_'.join(directory.split('_')[1:])  # converts bbm_option_N -> option_N
            product_profits[key] = dict()
            for product in self.products:
                for pid in pids:
                    path = os.path.join(directory,f'profit_{pid}.csv')
                    df = pd.read_csv(path, index_col='time')
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
                    if product == 'DEG':
                        prod_vals = prod_vals[1:]
                    else:
                        prod_vals = prod_vals[:-1]
                    if product in product_profits[key].keys():
                        product_profits[key][product] += prod_vals
                    else:
                        product_profits[key][product] = prod_vals
        self.save_summary(product_profits, times)

    def get_profit_summary(self, new_profits=False):
        """ Checks for or creates profit summary and loads """
        if not os.path.exists('daily_profit_summary.xlsx') or new_profits:
            self.create_profit_summary()
        self.ps = pd.read_excel('daily_profit_summary.xlsx', sheet_name=None)

    def make_boxplot(self, data, mode, show=False):
        """ Creates and styles boxplots """
        # Configure styles
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        # Make figure and add each column to the plot
        fig, ax = plt.subplots()
        values = [data.values[:,i] for i in range(data.values.shape[1]) if i != 0]
        labels = data.columns[1:]
        bplot = ax.boxplot(values, patch_artist=True, labels=labels)
        # Colors
        for patch, color, in zip(bplot['boxes'], sns.color_palette('hls', len(labels))):
            patch.set_facecolor(color)
        ax.set_ylabel('Dollars ($)', fontsize=14)
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/profit_{mode}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def plot_profits(self, show=False):
        """ Plots profit summary for income, degradation, and sum """
        modes = ['income', 'degradation', 'net']
        for mode in modes:
            if mode == 'net':
                net_revenue = self.ps['income']
                # Add degradation to income, skipping 1st column (times)
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
    dirs = ['../scenarios/bbm_baseline', '../scenarios/bbm_option_3']
    splotter = SurplusPlotter(dirs, sample_interval='day', stype='surplus_minus_str')
    # splotter.plot_differences()
    rplotter = RevenuePlotter(dirs, new_profits=True)
    rplotter.plot_profits(show=True)