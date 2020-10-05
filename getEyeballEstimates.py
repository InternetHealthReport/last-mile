import sys
import re
import json
import requests
from bs4 import BeautifulSoup

class HTMLTableParser:

    def parse_url(self, url):
        # Download the page, parse it, and extract the text containing the data
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        pattern = re.compile(
                r'new google\.visualization\.arrayToDataTable\((.+?)\);', 
                re.MULTILINE | re.DOTALL
                )
        script = soup.find("script", text=pattern)
        data_txt = pattern.search(str(script))

        if data_txt:
            return data_txt.group(1)
        else:
            return None

    def parse_html_table(self, table):
        res = {}
        table_list = eval(table)
        for row in table_list:

            rank, asn, name, cc, est_users, per_country, per_internet, samples = row
            try:
                if asn.startswith('AS'):
                    asn = int(asn[2:])
                else:
                    asn = int(asn)
            except ValueError:
                print(f'Could not parse ASN: {asn}')
                continue

            cc = cc.partition('>')[2].partition('<')[0]

            # the ASN may appear twice in the table (but with different country
            # code, probably a bug) keep only the highest ranked
            if asn not in res:
                res[asn] = {'rank': rank, 'asn': asn, 'name': name, 'cc': cc, 
                    'est_users': est_users, 'per_country': per_country, 
                    'per_internet': per_internet, 'samples': samples}

        return res

if __name__ == '__main__':

    url = 'https://stats.labs.apnic.net/aspop/'

    htp = HTMLTableParser()
    table = htp.parse_url(url)
    res = htp.parse_html_table(table)

    with open('data/eyeball.json', 'w') as fp:
        json.dump(res, fp)

