# last-mile
Last-mile delay analysis using RIPE Atlas. 

This repository contains a set of scripts used for the IMC'20 paper entitled:
['Persistent Last-mile Congestion: Not so Uncommon'](https://last-mile-congestion.github.io/last-mile-imc20-CR.pdf)
Please see also the accompanying website for more results and pre-processed datasets: https://last-mile-congestion.github.io


## Pre-process RIPE Atlas data 
First fetch and process RIPE Atlas traceroute data with [raclette](https://github.com/InternetHealthReport/raclette). This is an 
example configuration file to use with raclette: https://github.com/InternetHealthReport/raclette/blob/master/conf/lastmile.conf
You can reuse this and just change the start/end dates or the Atlas measurement IDs.
See [raclette's README](https://github.com/InternetHealthReport/raclette/blob/master/README.md) for instructions on how to run raclette.

Raclette will produce sqlite files that are directly read by the scripts available in this repository.
Alternatively you can also download sqlite files we produced for the IMC 20 publication: https://last-mile-congestion.github.io/data.html


## Analysis across all ASes

To compute last mile latency for all ASes found in raclette's sqlite file 
(hereafter referred as data/raclette/results_2020-04-01T00:00.sql), run the following commands:
```
python3 getEyeballEstimates.py
python3 survey.py data/raclette/results_2020-04-01T00:00.sql
```

The last command will produce a directory fig/survey_2020-04-01_minProbes3/ that
contains a directory for each AS and an index.html file to browse results.
It will also produce a json file (data/survey_2020-04-01.json) with all results
formated in JSON.

## Plot results for a single AS

To compute and monitor last mile latency for a single AS, run the following command:
```
python3 plotAsn.py ASN
```
with ASN set as the AS number of your choice. The results are stored in the fig directory.
This script assumes that your sqlite files are located in data/raclette/. Feel free to change
this at the beginning of the script.


## Feedback
Please report problems or send us your comments via github issues: https://github.com/InternetHealthReport/last-mile/issues
