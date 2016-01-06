# Grab a Graphite JSON formatted timeseries and image for use in Crucible
# usage:
# graphite-grab.py --target "http://your_graphite_host/render/?from=-24hour&target=stats.server1.cpu.system"
# graphite-grab.py --quiet --target "http://your_graphite_host/render/?from=1451952000&until=1451977200&target=stats.server1.cpu.system"

import json
import sys
import requests
import urlparse
import os
from os.path import dirname, join, abspath, isfile
import urllib2
import traceback

import argparse
parser = argparse.ArgumentParser(description='Retrieve a timeseries from graphite')
parser.add_argument('--quiet', action='store_true', default=False,
                    help='Execute without printing output')
parser.add_argument('--target', required=True,
                    help='The graphite render URL for the target')
args = parser.parse_args()

quiet = args.quiet
url = args.target

parsed = urlparse.urlparse(url)
target = urlparse.parse_qs(parsed.query)['target'][0]
data_folder = abspath(join(dirname(__file__), '..', 'data'))

# Get graphite timeseries image
image_url = url.replace("&format=json", "")
graphite_image_file = join(data_folder + "/" + target + '.png')
if "width" not in image_url:
    image_url += "&width=586"
if "height" not in image_url:
    image_url += "&height=308"

image_data = None
if image_data is None:
        try:
            image_data = urllib2.urlopen(image_url).read()
        except urllib2.URLError:
            image_data = None
if image_data is not None:
    with open(graphite_image_file, 'w') as f:
        f.write(image_data)
        f.close()
    if not quiet:
        print "graphite image saved as " + graphite_image_file

# Get graphite timeseries json
if "&format=json" not in url:
    url += "&format=json"

r = requests.get(url)
js = r.json()
datapoints = js[0]['datapoints']

converted = []
for datapoint in datapoints:
    try:
        new_datapoint = [float(datapoint[1]), float(datapoint[0])]
        converted.append(new_datapoint)
    except:
        continue

json_data_file = join(data_folder + "/" + target + '.json')
crucible_parameter_file = join(data_folder + "/" + target + '.crucible.parameters')

with open(json_data_file, 'w') as f:
    f.write(json.dumps(converted))
    f.close()
with open(json_data_file, 'r') as fr:
    timeseries = json.loads(fr.read())

if os.path.isfile(json_data_file):
    try:
        start_timestamp = int(timeseries[0][0])
        end_timestamp = int(timeseries[-1][0])
        full_duration = end_timestamp - start_timestamp
        if not quiet:
            print "target retrieved, run: python src/crucible.py --debug --end_timestamp " + str(end_timestamp) + " --full_duration " + str(full_duration)
            print "runtime arguments:"
            print "--end_timestamp " + str(end_timestamp) + " --full_duration " + str(full_duration)
    except:
        print("Graphite timeseries data error: " + traceback.format_exc())
        print("bad timeseries data: " + str(timeseries))
        if os.path.isfile(json_data_file):
            os.remove(json_data_file)
            print("removed: " + json_data_file)
        if os.path.isfile(graphite_image_file):
            os.remove(graphite_image_file)
            print("removed: " + graphite_image_file)
        if os.path.isfile(crucible_parameter_file):
            os.remove(crucible_parameter_file)
            print("removed: " + crucible_parameter_file)

        print("OK - cleaned up data from " + target)

if os.path.isfile(json_data_file) and end_timestamp:
    parameters_string = join("--end_timestamp " + str(end_timestamp) + " --full_duration " + str(full_duration))
    with open(crucible_parameter_file, 'w') as fw:
        fw.write(parameters_string)
        fw.close()
