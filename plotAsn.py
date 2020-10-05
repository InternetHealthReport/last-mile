import sys
import re
import json
import glob
from collections import defaultdict
import sqlite3
import pandas as pd
from matplotlib import pylab as plt
import welch
import numpy as np
from rftb import plot as rfplt

individual_plots = False
individual_plot_ylim = [0,27]

try: 
    asn = int(sys.argv[1])
except:
    sys.exit(f'usage: {sys.argv[0]} ASN')

# Hardcoded parameters :(
delay_db_pattern = ['data/raclette/results_20*T00:00.sql']
directory = 'fig/'
anchor = False
if anchor:
    probe_label = 'anchor'
else:
    probe_label = 'prb.'

# in this script all signals are re-sampled to 30min
fs = 2.0
nperseg = 3*24*fs
noverlap = None
minRMS = 0 
wel = welch.Welch(fs=fs, nperseg=nperseg, noverlap=noverlap, minRMS=minRMS)

# Load external data (RIPE probe info and APNIC eyeball estimates)
with open('data/probe_info.json', 'r') as fp:
    probe_raw = json.load(fp)

    asns = defaultdict(list)
    for probe in probe_raw['probes']:
        # ignore anchors
        if probe['is_anchor']==anchor: 
            if not probe.get('asn_v4'):
                continue
            asns[probe['asn_v4']].append(probe)

probes = asns[asn]
fnames = [fname for pattern in delay_db_pattern for fname in glob.glob(pattern)]
fnames.sort()

for delay_fname in fnames:
    match = re.search(r'results_(.+)T([0-9]+:[0-9]+)\.sql', delay_fname) 
    if not match:
        sys.exit("Could not parse the date in the input file name")
    date = match.group(1)

    con = sqlite3.connect(delay_fname)
    sql_where = ' OR '.join(
            [" (startpoint = 'PB{}' AND (endpoint = 'AS{}v4' or endpoint='AS0v4')) ".format(
                probe['id'], probe['asn_v4']) 
                for probe in probes
            ])

    df = pd.read_sql_query(
        "SELECT ts, median, startpoint \
                FROM diffrtt WHERE {} AND nbtracks>2".format(sql_where),
        con
        )

    if len(df)<7:
        continue

    # Re-arrange data
    df.index = pd.to_datetime(df.ts, unit='s')
    observed_probes = df.startpoint.unique()

    # Normalization
    df['min_values'] = df.groupby('startpoint')['median'].transform('min')
    df['norm_median'] = df['median']-df['min_values']
    df['timebin'] = df['ts'].floordiv(3600/fs)
    signal = df.groupby(['timebin']).median()

    if individual_plots:
        ind_signal = signal.copy()
        ind_signal.index = pd.to_datetime(ind_signal.index*3600/fs, unit='s')
        plt.figure(figsize=(6,3))
        plt.plot(ind_signal['norm_median'])
        plt.xticks(rotation=45)
        plt.ylabel('Agg. last-mile queuing delay (ms)')
        plt.ylim(individual_plot_ylim)
        plt.tight_layout()
        plt.savefig(directory+f'/norm_rtt_AS{asn}_{date}.png', dpi=300)

    # Plot corresponding spectrum
    f,pspec = wel.analyze(signal['norm_median'])
    freqMax = f[pspec.argmax()]
    # peak-to-peak amplitude = 2*RMS amp*sqrt(2)
    ampMax = 2*np.sqrt(pspec.max())*np.sqrt(2)
    print('Max. amplitude: ',ampMax)

    # plot the spectrogram
    plt.figure(10, figsize=(4,3))
    plt.plot(f, 2*np.sqrt(pspec)*np.sqrt(2))
    # plt.ylim([1e-3, 1e1])
    # plt.xlim([0, 1])
    plt.xlabel('Frequency (cycles per hour)')
    plt.xticks([1/i for i in [24, 4, 2, 1]], [f'1/{i}' for i in [24, 4, 2, 1]])
    plt.ylabel('Amplitude (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory+f'/power_spectrum_AS{asn}_{date}.png')
    plt.savefig(directory+f'/power_spectrum_AS{asn}_{date}.pdf')

    # Get a full week
    df['week'] = df.to_period('W').index
    weeks = list(df.week.unique())
    weeks.sort()
    if len(weeks)>2:
        df = df[df.week == weeks[1]]
    else:
        df = df[df.week == weeks[0]]
    # Normalization
    df['min_values'] = df.groupby('startpoint')['median'].transform('min')
    df['norm_median'] = df['median']-df['min_values']
    signal = df.groupby(['timebin']).median()
    # signal = df[['timebin','norm_median']].groupby(['timebin']).quantile(.75)

    # Plot RTT signal
    nb_probes = len(df.startpoint.unique())
    label = date.rpartition('-')[0]+f' ({nb_probes} {probe_label})'
    fig = plt.figure(1, figsize=(7,3))
    plt.plot(range(len(signal)), signal['norm_median'], label=label)


    # Distribution of the daily maximum median RTT
    df['day'] = df.to_period('D').index
    df_max_probe = df.groupby(['startpoint','day']).max()
    avg_max = df_max_probe.groupby('startpoint').median()['norm_median']
    fig = plt.figure(34, figsize=(4,3))
    rfplt.ecdf(avg_max, label=label)
    plt.title(f'AS{asn}')
    # plt.legend(loc='upper right',  ncol=3, fontsize='small')
    plt.autoscale()
    plt.xscale('log')
    plt.savefig(directory+f'/median_max_delay_per_probe_AS{asn}.pdf')
    plt.xscale('linear')
    plt.xlim([0,5])
    plt.ylim([0.4,0.9])
    plt.savefig(directory+f'/median_max_delay_per_probe_AS{asn}_zoomed.pdf')

    avg_max = df_max_probe.groupby('startpoint').mean()['norm_median']
    fig = plt.figure(35, figsize=(4,3))
    rfplt.ecdf(avg_max, label=label)
    plt.title(f'AS{asn}')
    # plt.legend(loc='upper right',  ncol=3, fontsize='small')
    plt.autoscale()
    plt.xscale('log')
    plt.savefig(directory+f'/avg_max_delay_per_probe_AS{asn}.pdf')
    plt.xscale('linear')
    plt.xlim([0,5])
    plt.ylim([0.0,1])
    plt.savefig(directory+f'/avg_max_delay_per_probe_AS{asn}_zoomed.pdf')

    avg_max = df_max_probe.groupby('startpoint').min()['norm_median']
    fig = plt.figure(36, figsize=(4,3))
    rfplt.ecdf(avg_max, label=label)
    plt.title(f'AS{asn}')
    # plt.legend(loc='upper right',  ncol=3, fontsize='small')
    plt.autoscale()
    plt.xscale('log')
    plt.savefig(directory+f'/min_max_delay_per_probe_AS{asn}.pdf')
    plt.xlim([5,10])
    plt.ylim([0.6,1])
    plt.xscale('linear')
    plt.savefig(directory+f'/min_max_delay_per_probe_AS{asn}_zoomed.pdf')

# Save the aggregated last-mile queuing delay plot
fig = plt.figure(1, figsize=(7,3))
df['dayname'] = df.index.day_name()
plt.xticks([i*24*fs for i in range(7)], df['dayname'].unique())
plt.ylabel('Agg. queuing delay (ms)')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', bbox_to_anchor= (0.0, 1.01), ncol=3,
        borderaxespad=0, frameon=False, fontsize='small')
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(directory+f'/norm_rtt_AS{asn}_all.png')
plt.savefig(directory+f'/norm_rtt_AS{asn}_all.pdf')
plt.close()

