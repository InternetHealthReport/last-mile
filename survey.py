import json
import numpy as np
import welch
from collections import defaultdict
import sqlite3
import sys
import pandas as pd
from matplotlib import pylab as plt
import os
import re
from flask import render_template
import flask
import country_converter as coco

if len(sys.argv) < 2:
    sys.exit(f'usage: {sys.argv[0]} data/raclette/results_XXXX.sql')

delay_fname = sys.argv[1]
match = re.search(r'results_(.+)T([0-9]+:[0-9]+)\.sql', delay_fname) 
if not match:
    sys.exit("Could not parse the date in the input file name")
date = match.group(1)
print("Start analaysis for: ", date)

app = flask.Flask('Last Mile RTT')

with app.app_context():

    CLASS_THRESHOLD = [
        ('low', 0.5), 
        ('mild', 1.0), 
        ('severe', 3.0), 
        ]
    MIN_NB_PROBES = 3

    fs = 2.0
    nperseg = 3*24*fs
    noverlap = None
    minRMS = 0 
    wel = welch.Welch(fs=fs, nperseg=nperseg, noverlap=noverlap, minRMS=minRMS)

    # Load external data (RIPE probe info and APNIC eyeball estimates)
    with open('data/probe_info.json', 'r') as fp:
        probe_raw = json.load(fp)

        asns = defaultdict(list)
        probes_info = defaultdict(dict)
        for probe in probe_raw['probes']:
            # ignore old probes and anchors
            if probe['is_anchor']==False: 
                if not probe.get('asn_v4'):
                    continue

                asns[probe['asn_v4']].append(probe)
                probes_info[probe['id']] = probe

    with open('data/eyeball.json', 'r') as fp:
        eyeball = json.load(fp)


    stats = {
            'unknown': defaultdict(list), 
            'low': defaultdict(list),
            'mild': defaultdict(list),
            'severe': defaultdict(list)
            }

    index_data = {key: defaultdict(list) for key in stats.keys()}

    # fetch raclette results
    con = sqlite3.connect(delay_fname)
    for asn, probes in asns.items():

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

        # Re-arrange data
        df.index = pd.to_datetime(df.ts, unit='s')
        observed_probes = df.startpoint.unique()
        if len(observed_probes) < MIN_NB_PROBES:
            continue

        # Normalization
        df['min_values'] = df.groupby('startpoint')['median'].transform('min')
        df['norm_median'] = df['median']-df['min_values']

        # Frequency Analysis
        df['timebin'] = df['ts'].floordiv(3600/fs)
        signal = df.groupby(['timebin']).median()

        welch_results = wel.analyze(signal['norm_median'])
        dailyFluc = False
        if welch_results:
            f,pspec = welch_results
            freqMax = f[pspec.argmax()]
            # peak-to-peak amplitude = 2*RMS amp*sqrt(2)
            ampMax = 2*np.sqrt(pspec.max())*np.sqrt(2)
            if freqMax < 1.0/23 and freqMax > 1.0/25:
                dailyFluc = True
                # ips, _, timebin = link.rpartition("_")
                # ip = ips[1:-1].split(",")
                # ampMax = np.sqrt(pspec.max())
                # fi.write("%s, %s, %s, %s, %s\n" % (ip[0], ip[1], freqMax, ampMax, timebin))
        else:
            freqMax = None
            ampMax = -1


        # Make profile
        df['hour'] = df.index.hour
        profile = df.groupby(['hour']).median()

        # Skip ASes with incomplete profile
        if len(profile) < 24:
            continue

        low = profile['norm_median'].quantile(0.125)
        high = profile['norm_median'].quantile(0.875)
        deviation = (high-low)/low 

        # Classify the asn based on its peak-to-peak amplitude
        classification = 'unknown'
        for label, threshold in CLASS_THRESHOLD:
            if ampMax > threshold and dailyFluc:
                classification = label

        # Record overall statistics
        eye = eyeball.get(str(asn), defaultdict(str))
        asn_stats = {
            'asn': asn,
            # 'snr': signal['median'].mean()/signal['median'].std(),
            # 'snr_welch': np.mean(pspec)/np.std(pspec),
            'deviation': deviation,
            'diff_ms': high-low,
            'cc': eye['cc'],
            'name': eye['name'],
            'eyeball_rank': eye['rank'],
            'nb_probes': len(observed_probes),
            'freqMax': freqMax,
            'ampMax': ampMax,
            'dailyFluc': dailyFluc,
            'classification': classification
            }
        stats[classification][asn] = asn_stats

        directory = f'fig/survey_{date}_minProbes{MIN_NB_PROBES}/AS{asn}/'
        os.makedirs(directory, exist_ok=True)

        details = [
            f'Number of probes: {len(observed_probes)}',
            f'APNIC eyeball rank: {eye["rank"]}',
            f'Daily fluctuations: {dailyFluc}',
            f'Main frequency: {freqMax:.4f}',
            f'Average peak-to-peak amplitude: {ampMax:.2f}ms',
            # f'Low RTT: {low:.2f}ms',
            # f'High RTT: {high:.2f}ms ',
            # f'snr: {signal["median"].mean()/signal["median"].std()}',
            # f"snr_welch: {np.mean(pspec)/np.std(pspec)}",
            # f'deviation: {deviation*100:.2f}%',
            # f'signal length: {signal["norm_median"].count()}',
            ]

        index_data[classification][coco.convert(names=[eye['cc']], to='short')].append( 
                {
                'asn': f'AS{asn}',
                'name': eye['name'],
                'cc': eye['cc'],
                'details': details,
                })

        if classification == 'unknown':
            continue

        # print details
        print('\nAS', asn)
        for det in details:
            print(det)

        plt.figure(figsize=(5,4))
        plt.ylabel('Median aggregated queuing delay')
        plt.xlabel('Hour of the day')
        plt.plot(profile.index, profile['norm_median'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(directory+f'/profile_AS{asn}_{date}.png')
        # plt.title(f'AS{asn}')
        # plt.tight_layout()
        # plt.savefig(directory+f'/profile_AS{asn}_{date}.pdf')
        plt.close()

        # Plot the normalized median RTT for the ASN
        fig = plt.figure(figsize=(7,3))
        plt.ylabel('Aggregated queuing delay')
        plt.xlabel('Time')
        plt.plot(pd.to_datetime(signal.ts, unit='s'), signal['norm_median'])
        plt.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(directory+f'/med_rtt_AS{asn}_{date}.png')
        # plt.title(f'AS{asn}')
        # plt.tight_layout()
        # plt.savefig(directory+f'/med_rtt_AS{asn}_{date}.pdf')
        plt.close()

        # plot the spectrogram
        plt.figure(figsize=(5,4))
        plt.plot(f, 2*np.sqrt(pspec)*np.sqrt(2))
        # plt.ylim([1e-3, 1e1])
        plt.xlim([0, 1])
        plt.xlabel('Frequency (cycles per hour)')
        plt.ylabel('Amplitude (ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(directory+f'/power_spectrum_AS{asn}_{date}.png')
        # plt.savefig(directory+f'/power_spectrum_AS{asn}_{date}.pdf')
        plt.close()

        for probe in observed_probes:
            pid = int(probe[2:])
            city = probes_info[pid]["city"]
            if len(city)>2:
                city = city[2:]
            fig = plt.figure(figsize=(7,3))
            plt.title(f'AS{asn}, {probe}, {city}')
            plt.ylabel('Median last-mile RTT')
            dftmp = df[df['startpoint'] == probe]
            plt.plot(dftmp.index, dftmp['median'], label=probe)
            plt.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            # plt.savefig(directory+f'/signals_AS{asn}_{probe}_{date}.pdf')
            plt.savefig(directory+f'/signals_AS{asn}_{probe}_{date}.png')
            plt.close()


        # Generate report page
        rendered = render_template( 'as_report.html', 
            title = f'Last-mile delay survey / {date} / AS{asn}' ,
            asn = asn,
            name = eye['name'],
            cc = eye['cc'],
            date = date,
            details = details,
            probes = observed_probes,
            classification= classification
            )

        with open(directory+f'/AS{asn}.html', 'w') as fp:
            fp.write(rendered)


    summary = [
        f"{len(stats['severe'])} ASes severely congested ASes ({len(index_data['severe'])} countries)",
        f"{len(stats['mild'])} ASes mildly congested ASes ({len(index_data['mild'])} countries)",
        f"{len(stats['low'])} ASes with low daily fluctuations ({len(index_data['low'])} countries)",
        f"{len(stats['unknown'])} ASes without daily fluctuations ({len(index_data['unknown'])} countries)",
        ]

    # Generate index page
    rendered = render_template( 'index.html', 
        severe = index_data['severe'],
        mild = index_data['mild'],
        low = index_data['low'],
        date = date,
        summary = summary
        )

    directory = f'fig/survey_{date}_minProbes{MIN_NB_PROBES}/'
    with open(directory+f'/index.html', 'w') as fp:
        fp.write(rendered)


    # Print the classification breakdown
    for line in summary:
        print(line)

    # store stats:
    with open(f'data/survey_{date}.json', 'w') as fp:
        json.dump(stats, fp)

