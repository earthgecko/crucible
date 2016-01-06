import logging
import time
from multiprocessing import Process
import os
from os.path import dirname, join, abspath, isfile
from sys import exit, version_info
import traceback
from settings import ALGORITHMS
import json
import shutil
from os import getcwd, listdir, makedirs

import gzip

from algorithms import run_algorithms

import argparse
parser = argparse.ArgumentParser(description='Analyse a timeseries.')
parser.add_argument('--end_timestamp', required=True, type=int,
                    help='The epoch end timestamp of your timeseries')
parser.add_argument('--full_duration', required=True, type=int,
                    help='The FULL_DURATION of your timeseries in seconds')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Execute without printing output')
args = parser.parse_args()

user_end_timestamp = args.end_timestamp
user_full_duration = args.full_duration
debug = args.debug


class Crucible():

    def run(self):
        """
        Called when the process intializes.
        """
        __data__ = abspath(join(dirname(__file__), '..', 'data'))
        files = [f for f in listdir(__data__) if isfile(join(__data__, f))]

        __results__ = abspath(join(dirname(__file__), '..', 'results'))

        python_version = '.'.join(map(str, version_info[:3]))
        if debug:
            print 'python verison - ' + python_version

        # Spawn processes
        pids = []
        for index, timeseries_filename in enumerate(files):
            if not timeseries_filename.endswith('.json'):
                continue

            timeseries_name = timeseries_filename.replace('.json', '')
            timeseries_dir = timeseries_name.replace('.', '/')

            with open(join(__data__ + "/" + timeseries_filename), 'r') as f:
                timeseries = json.loads(f.read())
                start_timestamp = int(timeseries[0][0])
                end_timestamp = int(timeseries[-1][0])
                full_duration = end_timestamp - start_timestamp
                results_dir = join(__results__ + "/" + timeseries_dir + "/" + str(end_timestamp))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                if debug:
                    print 'Spawning process to analyse ' + timeseries_name
                p = Process(target=run_algorithms, args=(timeseries, timeseries_filename, end_timestamp, full_duration, debug, timeseries_filename))
                pids.append(p)
                p.start()

        # Send wait signal to zombie processes
        for p in pids:
            p.join()

        for index, timeseries_filename in enumerate(files):
            if not timeseries_filename.endswith('.json'):
                continue

            timeseries_name = timeseries_filename.replace('.json', '')
            timeseries_dir = timeseries_name.replace('.', '/')

            with open(join(__data__ + "/" + timeseries_filename), 'r') as f:
                timeseries = json.loads(f.read())
                end_timestamp = int(timeseries[-1][0])
                results_dir = join(__results__ + "/" + timeseries_dir + "/" + str(end_timestamp))

                # Archive all artefacts to the results_dir for this timeseries
                results_data_file = join(results_dir + '/data.json')
                if not os.path.isfile(results_data_file):
                    try:
                        shutil.move(join(__data__ + '/' + timeseries_filename), results_data_file)
                    except:
                        print 'Failed to move data file to the results dir'
                        traceback.print_exc()

                # gzip the json timeseries
                if os.path.isfile(results_data_file):
                    results_data_gz = join(results_data_file + '.gz')
                    try:
                        f_in = open(results_data_file)
                        f_out = gzip.open(results_data_gz, 'wb')
                        f_out.writelines(f_in)
                        f_out.close()
                        f_in.close()
                        os.remove(results_data_file)
                    except:
                        print 'Failed to gzip data file with python ' + python_version
                        traceback.print_exc()

                # Move any graphite image file associated with the json
                graphite_target_metric = os.path.splitext(timeseries_filename)[0]
                graphite_image = join(__data__ + '/' + graphite_target_metric + '.png')
                if debug:
                    print "graphite graph - " + graphite_image
                if os.path.isfile(graphite_image):
                    graphite_image_results_file = join(results_dir + '/' + graphite_target_metric + '.png')
                    if not os.path.isfile(graphite_image_results_file):
                        try:
                            shutil.move(graphite_image, graphite_image_results_file)
                        except:
                            print 'Failed to move graphite image file to the results dir'
                            traceback.print_exc()

                # Move the parameters file if it exists
                parameters_file = join(__data__ + '/' + graphite_target_metric + '.crucible.parameters')
                if os.path.isfile(parameters_file):
                    results_crucible_file = join(results_dir + '/crucible.parameters')
                    if not os.path.isfile(results_crucible_file):
                        try:
                            shutil.move(parameters_file, results_crucible_file)
                        except:
                            print 'No crucible.parameters file to move'

                print 'Results image files and the gzipped json timeseries saved to: ' + results_dir + '/'


if __name__ == "__main__":
    """
    Start Crucible.
    """

    int_end_timestamp = int(user_end_timestamp)
    int_full_duration = int(user_full_duration)

    # Make sure we have data file/s to test
    __data__ = abspath(join(dirname(__file__), '..', 'data'))
    if not os.listdir(__data__):
        print 'No files found in the data/ directory to process.'
        exit(0)

    # Make sure we can run all the algorithms
    try:
        from algorithms import *

        # However if the full_duration of the timeseries is > 86400 we only test
        # at 86400 to save time as if an algorithm can run on a 24hr data set,
        # it should be able to be analysed at a greater full duration as well
        if int_full_duration >= 86400:
            if debug:
                print "full duration of " + str(int_full_duration) + " was passed but only testing at 86400"
            int_full_duration = 86400
        timeseries = map(list, zip(map(float, range(int_end_timestamp - int_full_duration, int_end_timestamp + 1)), [1] * (int_full_duration + 1)))
        ensemble = [globals()[algorithm](timeseries, int_end_timestamp, int_full_duration, debug) for algorithm in ALGORITHMS]
    except KeyError as e:
        print 'Algorithm %s deprecated or not defined; check settings.ALGORITHMS' % e
        exit(1)
    except Exception as e:
        print 'Algorithm test run failed.'
        traceback.print_exc()
        exit(1)

    __results__ = abspath(join(dirname(__file__), '..', 'results'))

    if not os.path.exists(__results__):
        os.makedirs(__results__)

    crucible = Crucible()
    crucible.run()
