import sys
from matplotlib import pylab as plt
import matplotlib.patheffects as PathEffects
import json
from collections import defaultdict
import numpy as np
from rftb import plot as rfplt


with open('data/survey_2019-09-01.json', 'r') as fp:
    survey_results2019 = json.load(fp)

with open('data/survey_2020-04-01.json', 'r') as fp:
    survey_results2020 = json.load(fp)

asns2020 = {res['asn']: res 
        for classif, classif_results in survey_results2020.items()
            for res in classif_results.values()}

asns2019 = {res['asn']: res 
        for classif, classif_results in survey_results2019.items()
            for res in classif_results.values()}

asnsDaily2020 = defaultdict(float, {asn: res['ampMax'] 
        for asn, res in asns2020.items()
            if res['dailyFluc']})

asnsDaily2019 = defaultdict(float,{asn: res['ampMax'] 
        for asn, res in asns2019.items()
            if res['dailyFluc']})


def plot_survey(fig, ax, results, category_names, data_label, offset=0, ymax=40):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = np.array(list(results.keys()))
    x_ticks = np.arange(len(list(results.keys())))
    data_sum = np.sum(list(results.values()), axis=1)
    data_raw = np.array(list(results.values()))
    data = np.array(list(results.values()))
    for i, s in enumerate(data_sum):
        data[i] = 100*data[i]/s 

    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    # fig, ax = plt.subplots(num=fignum, figsize=(7, 4))
    # ax.yaxis.set_visible(False)
    ax.set_ylim(0, ymax) #np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.bar(x_ticks+offset, widths, align='center', bottom=starts, width=0.35,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        text_outline = 'white' if r * g * b >= 0.5 else 'k'
        for x, (y, c) in enumerate(zip(xcenters, data_raw[:,i])):
            c = int(c)
            if c==0:
                continue
            if y > ymax - 5:
                y = ymax - 5
                ax.text(x+offset, y, data_label+'\n\n\n\n\n\n', ha='center', va='center',
                        color='k', alpha=0.5)

            txt = ax.text(x+offset, y, str(c), ha='center', va='center',
                    color=text_color, fontsize='small')
            txt.set_path_effects([PathEffects.withStroke(linewidth=0.7, foreground=text_outline)])

    return fig, ax



# plot rank vs. daily amplitude

rank = [int(res['eyeball_rank'])
    for res in asns2020.values()
        if res['eyeball_rank']!='' 
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification']!='unknown']
amp = [float(res['ampMax'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification']!='unknown']

rank_new = [int(res['eyeball_rank'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification'] in ['unknown']]
amp_new = [float(res['ampMax'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification'] in ['unknown']]

rank_jp = [int(res['eyeball_rank'])
    for res in asns2020.values()
        if res['eyeball_rank']!='' 
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification']!='unknown'
        and res['cc']=='JP'
        ]
amp_jp = [float(res['ampMax'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification']!='unknown'
        and res['cc']=='JP'
        ]

rank_new_jp = [int(res['eyeball_rank'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification'] in ['unknown']
        and res['cc']=='JP'
        ]
amp_new_jp = [float(res['ampMax'])
    for res in asns2020.values()
        if res['eyeball_rank']!=''
        and res['classification']!='unknown'
        and res['asn'] in asns2019
        and asns2019[res['asn']]['classification'] in ['unknown']
        and res['cc']=='JP'
        ]


# Plot histogram for the percentage of congested networks
congestion_labels = ['severe', 'mild', 'low']
bins = np.array([10**i for i in range(6)])
bins[0] = 0
bins_label = ['1-10', '10-100', '100-1k', '1k-10k', '+10k']
congested20 = [res['eyeball_rank'] 
        for asn, res in asns2020.items() 
            if res['classification'] in congestion_labels and res['eyeball_rank']!='']  
all20 = [res['eyeball_rank'] 
        for res in asns2020.values() 
            if res['eyeball_rank']!='']
congested19 = [res['eyeball_rank'] 
        for asn, res in asns2019.items() 
            if res['classification'] in congestion_labels and res['eyeball_rank']!='']  
all19 = [res['eyeball_rank'] 
        for res in asns2019.values() if res['eyeball_rank']!='']

c20h = np.histogram(congested20, bins)
print(c20h)
a20h = np.histogram(all20, bins)
print(a20h)
perc20 = 100* np.histogram(congested20, bins)[0] / np.histogram(all20, bins)[0]
perc19 = 100* np.histogram(congested19, bins)[0] / np.histogram(all19, bins)[0]

width = 0.35
x = np.arange(len(bins_label))

fig, ax = plt.subplots(figsize=(6,3))
ax.bar(x-width/2, perc19, align='center', width=width, label='Sep. 2019')
ax.bar(x+width/2, perc20, align='center', width=width, label='Apr. 2020')
ax.set_ylabel('Percentage of congested ASes')
ax.set_xlabel('APNIC eyeball rank')
ax.set_xticks(x)
ax.set_xticklabels(bins_label)
ax.legend()
plt.tight_layout()
fig.savefig('fig/perc_congested.pdf')

# Plot breakdown of survey
category_names = ['severe', 'mild', 'low', 'unknown']
pretty_category_names = ['Severe', 'Mild', 'Low', 'None']
rank_labels = ['1 to 10', '11 to 100', '101 to 1k', '1k to 10k', 'more than 10k']
fignum = 12345
offset = -0.4/2
fig, ax = plt.subplots(num=fignum, figsize=(6, 4))
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(False)
for i, (data_label, data) in enumerate([('Sep.\n2019', asns2019), ('Apr.\n2020', asns2020)]):
    results =  {key: np.zeros(len(category_names)) for key in rank_labels}
    for asn, res in data.items():
        if res['eyeball_rank'] == '':
            continue
        results[rank_labels[int(np.log10(res['eyeball_rank']))]][category_names.index(res['classification'])] += 1

    plot_survey(fig, ax, results, pretty_category_names, data_label, 
            offset=offset, ymax=39)
    offset += 0.4 

    if i == 0:
        ax.legend(ncol=len(pretty_category_names), bbox_to_anchor=(0, 1.2),
                    loc='lower left') #, fontsize='small')

x_ticks = np.arange(len(rank_labels))
ax.set_xticks(x_ticks)
ax.set_xticklabels(rank_labels)
plt.xlabel('APNIC eyeball rank')
plt.ylabel('Percentage of ASes')
plt.tight_layout()

plt.savefig(f'fig/survey_breakdown.pdf')

# Plot APNIC rank vs. amplitude
plt.figure(figsize=(4,3))
plt.plot(rank_new, amp_new, '^', label='Apr. 2020 only', alpha=0.7)
plt.plot(rank, amp, 's', label='Sep. 2019 & Apr. 2020', alpha=0.7)
plt.plot(rank_new_jp, amp_new_jp, 'ko', ms=10, fillstyle='none', label='Japanese ASN', alpha=0.3)
plt.plot(rank_jp, amp_jp, 'ko', ms=10, fillstyle='none', label=None, alpha=0.3)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('APNIC eyeball rank')
plt.ylabel('Daily amplitude (ms)')
plt.legend(loc='lower left', bbox_to_anchor= (-0.1, 1.02), ncol=2, prop={'size': 9})
plt.tight_layout()
plt.savefig('fig/survey_eyeballrank_vs_amp.pdf')

# plot amplitude before/after  vs. rank
plt.figure(figsize=(4,3))
rank = [int(res['eyeball_rank'])
    for res in asns2019.values()
        if res['eyeball_rank']!='']
ampDiff = [asnsDaily2020[asn] - asnsDaily2019[asn]
    for asn, res in asns2019.items()
        if res['eyeball_rank']!='']

plt.plot(rank, ampDiff, '.')
plt.xscale('log')
plt.yscale('log')
plt.savefig('fig/survey_eyeballrank_vs_ampDiff.pdf')


# plot amplitude distribution 
plt.figure()
# Make CDF
for data_label, data in [('Sep.2019', asnsDaily2019), ('Apr.2020', asnsDaily2020)]:
    dist = np.array([v for v in data.values() if v>0.5])
    rfplt.ecdf(dist, label=data_label)

plt.legend()
plt.grid(True, alpha=0.3)
# plt.yscale('log')
# plt.xscale('log')
plt.xlim([0.5,3])
plt.tight_layout()
plt.savefig('fig/survey_daily_amp_cdf.pdf')


# Print stats
totalAsns = set([asn for data in [asns2019, asns2020] for asn in data.keys()])
print(f'Total number of ASNs: {len(totalAsns)}')
