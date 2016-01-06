import pandas
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import traceback
import os
import time
from multiprocessing import Process
from settings import ALGORITHMS
from os.path import dirname, join, abspath
from os import makedirs

"""
This is no man's land. Do anything you want in here,
as long as you return a boolean that determines whether the input
timeseries is anomalous or not.

To add an algorithm, define it here, and add its name to settings.ALGORITHMS.
It must be defined required parameters (even if your algorithm/function does not
need them), as the run_algorithms function passes them to all ALGORITHMS defined
in settings.ALGORITHMS.
"""


def tail_avg(timeseries, end_timestamp, full_duration, debug):
    """
    This is a utility function used to calculate the average of the last three
    datapoints in the series as a measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity and increases the delay
    to detection.
    """
    try:
        t = (timeseries[-1][1] + timeseries[-2][1] + timeseries[-3][1]) / 3
        return t
    except IndexError:
        return timeseries[-1][1]


def median_absolute_deviation(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the deviation of its latest datapoint with
    respect to the median is X times larger than the median of deviations.
    """

    series = pandas.Series([x[1] for x in timeseries])
    median = series.median()
    demedianed = np.abs(series - median)
    median_deviation = demedianed.median()

    # The test statistic is infinite when the median is zero,
    # so it becomes super sensitive. We play it safe and skip when this happens.
    if median_deviation == 0:
        return False

    test_statistic = demedianed.iget(-1) / median_deviation

    # Completely arbitary...triggers if the median deviation is
    # 6 times bigger than the median
    if test_statistic > 6:
        return True


def grubbs(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    """

    series = scipy.array([x[1] for x in timeseries])
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(timeseries, end_timestamp, full_duration, debug)
    z_score = (tail_average - mean) / stdDev
    len_series = len(series)
    threshold = scipy.stats.t.isf(.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

    return z_score > grubbs_score


def first_hour_average(timeseries, end_timestamp, full_duration, debug):
    """
    Calcuate the simple average over 60 datapoints (maybe one hour),
    FULL_DURATION seconds ago.
    A timeseries is anomalous if the average of the last three datapoints
    are outside of three standard deviations of this value.
    """

    int_end_timestamp = int(timeseries[-1][0])
    int_start_timestamp = int(timeseries[0][0])
    int_full_duration = int_end_timestamp - int_start_timestamp

# Determine data resolution
#    last_hour_threshold = int_end_timestamp - (int_full_duration - 3600)
    int_second_last_end_timestamp = int(timeseries[-2][0])
    resolution = int_end_timestamp - int_second_last_end_timestamp
    ten_data_point_seconds = resolution * 10
    ten_datapoints_ago = int_end_timestamp - ten_data_point_seconds
    sixty_data_point_seconds = resolution * 60
    sixty_datapoints_ago = int_end_timestamp - sixty_data_point_seconds
    last_hour_threshold = int_end_timestamp - (int_full_duration - sixty_datapoints_ago)

    series = pandas.Series([x[1] for x in timeseries if x[0] < last_hour_threshold])
    mean = (series).mean()
    stdDev = (series).std()
    t = tail_avg(timeseries, end_timestamp, full_duration, debug)

    return abs(t - mean) > 3 * stdDev


def stddev_from_average(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than one standard
    deviation of the average. This does not exponentially weight the MA and so
    is better for detecting anomalies with respect to the entire series.
    """
    series = pandas.Series([x[1] for x in timeseries])
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries, end_timestamp, full_duration, debug)

    return abs(t - mean) > 3 * stdDev


def stddev_from_moving_average(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than one standard
    deviation of the moving average. This is better for finding anomalies with
    respect to the short term trends.
    """
    series = pandas.Series([x[1] for x in timeseries])
    expAverage = pandas.stats.moments.ewma(series, com=50)
    stdDev = pandas.stats.moments.ewmstd(series, com=50)

    return abs(series.iget(-1) - expAverage.iget(-1)) > 3 * stdDev.iget(-1)


def mean_subtraction_cumulation(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the value of the next datapoint in the
    series is farther than a standard deviation out in cumulative terms
    after subtracting the mean from each data point.
    """

    series = pandas.Series([x[1] if x[1] else 0 for x in timeseries])
    series = series - series[0:len(series) - 1].mean()
    stdDev = series[0:len(series) - 1].std()
    expAverage = pandas.stats.moments.ewma(series, com=15)

    return abs(series.iget(-1)) > 3 * stdDev


def least_squares(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the average of the last three datapoints
    on a projected least squares model is greater than three sigma.
    """

    x = np.array([t[0] for t in timeseries])
    y = np.array([t[1] for t in timeseries])
    A = np.vstack([x, np.ones(len(x))]).T
    results = np.linalg.lstsq(A, y)
    residual = results[1]
    m, c = np.linalg.lstsq(A, y)[0]
    errors = []
    for i, value in enumerate(y):
        projected = m * x[i] + c
        error = value - projected
        errors.append(error)

    if len(errors) < 3:
        return False

    std_dev = scipy.std(errors)
    t = (errors[-1] + errors[-2] + errors[-3]) / 3

    return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


def histogram_bins(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the average of the last three datapoints falls
    into a histogram bin with less than 20 other datapoints (you'll need to tweak
    that number depending on your data)

    Returns: the size of the bin which contains the tail_avg. Smaller bin size
    means more anomalous.
    """

    int_end_timestamp = int(timeseries[-1][0])
    int_start_timestamp = int(timeseries[0][0])
    int_full_duration = int_end_timestamp - int_start_timestamp

    series = scipy.array([x[1] for x in timeseries])
    t = tail_avg(timeseries, int_end_timestamp, int_full_duration, debug)
    h = np.histogram(series, bins=15)
    bins = h[1]
    for index, bin_size in enumerate(h[0]):
        if bin_size <= 20:
            # Is it in the first bin?
            if index == 0:
                if t <= bins[0]:
                    return True
            # Is it in the current bin?
            elif t >= bins[index] and t < bins[index + 1]:
                    return True

    return False


def ks_test(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 datapoints (might be 10 minutes) is
    different from the last 60 datapoints (might be an hour).
    It produces false positives on non-stationary series so Augmented
    Dickey-Fuller test applied to check for stationarity.
    """

    int_end_timestamp = int(timeseries[-1][0])

    hour_ago = int_end_timestamp - 3600
    ten_minutes_ago = int_end_timestamp - 600
# Determine resolution of the data set
#    reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
#    probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])
    int_second_last_end_timestamp = int(timeseries[-2][0])
    resolution = int_end_timestamp - int_second_last_end_timestamp
    ten_data_point_seconds = resolution * 10
    ten_datapoints_ago = int_end_timestamp - ten_data_point_seconds
    sixty_data_point_seconds = resolution * 60
    sixty_datapoints_ago = int_end_timestamp - sixty_data_point_seconds
    reference = scipy.array([x[1] for x in timeseries if x[0] >= sixty_datapoints_ago and x[0] < ten_datapoints_ago])
    probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_datapoints_ago])

    if reference.size < 20 or probe.size < 20:
        return False

    ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        adf = sm.tsa.stattools.adfuller(reference, 10)
        if adf[1] < 0.05:
            return True

    return False


def detect_drop_off_cliff(timeseries, end_timestamp, full_duration, debug):
    """
    A timeseries is anomalous if the average of the last ten datapoints is <trigger>
    times greater than the last data point.  This algorithm is most suited to
    timeseries with most datapoints being > 100 (e.g high rate).  The arbitrary
    <trigger> values become more noisy with lower value datapoints, but it still
    matches drops off cliffs.
    """

    if len(timeseries) < 21:
        return False

    int_end_timestamp = int(timeseries[-1][0])
    # Determine resolution of the data set
    int_second_last_end_timestamp = int(timeseries[-2][0])
    resolution = int_end_timestamp - int_second_last_end_timestamp
    ten_data_point_seconds = resolution * 10
    ten_datapoints_ago = int_end_timestamp - ten_data_point_seconds

    ten_datapoint_array = scipy.array([x[1] for x in timeseries if x[0] <= int_end_timestamp and x[0] > ten_datapoints_ago])
    ten_datapoint_array_len = len(ten_datapoint_array)
    if ten_datapoint_array_len > 3:
        # DO NOT handle if negative integers in range, where is the bottom of
        # of the cliff if a range goes negative? The maths does not work either
        ten_datapoint_min_value = np.amin(ten_datapoint_array)
        if ten_datapoint_min_value < 0:
            return False
        ten_datapoint_max_value = np.amax(ten_datapoint_array)
        if ten_datapoint_max_value < 10:
            return False
        ten_datapoint_array_sum = np.sum(ten_datapoint_array)
        ten_datapoint_value = int(ten_datapoint_array[-1])
        ten_datapoint_average = ten_datapoint_array_sum / ten_datapoint_array_len
        ten_datapoint_value = int(ten_datapoint_array[-1])
        ten_datapoint_max_value = np.amax(ten_datapoint_array)
        if ten_datapoint_max_value == 0:
            return False
        if ten_datapoint_max_value < 101:
            trigger = 15
        if ten_datapoint_max_value < 20:
            trigger = ten_datapoint_average / 2
        if ten_datapoint_max_value < 1:
            trigger = 0.1
        if ten_datapoint_max_value > 100:
            trigger = 100
        if ten_datapoint_value == 0:
            # Cannot divide by 0, so set to 0.1 to prevent error
            ten_datapoint_value = 0.1
        if ten_datapoint_value == 1:
            trigger = 1
        if ten_datapoint_value == 1 and ten_datapoint_max_value < 10:
            trigger = 0.1
        if ten_datapoint_value == 0.1 and ten_datapoint_average < 1 and ten_datapoint_array_sum < 7:
            trigger = 7
        # Filter low rate and variable between 0 and 100 metrics
        if ten_datapoint_value <= 1 and ten_datapoint_array_sum < 100 and ten_datapoint_array_sum > 1:
            all_datapoints_array = scipy.array([x[1] for x in timeseries])
            all_datapoints_max_value = np.amax(all_datapoints_array)
            if all_datapoints_max_value < 100:
                # print "max_value for all datapoints at - " + str(int_end_timestamp) + " - " + str(all_datapoints_max_value)
                return False
        ten_datapoint_result = ten_datapoint_average / ten_datapoint_value
        if int(ten_datapoint_result) > trigger:
            if debug:
                print "ANOMALY - at " + str(int_end_timestamp) + ", ten_datapoint_value = " + str(ten_datapoint_value) + ", ten_datapoint_array_sum = " + str(ten_datapoint_array_sum) + ", ten_datapoint_average = " + str(ten_datapoint_average) + ", trigger = " + str(trigger) + ", ten_datapoint_result = " + str(ten_datapoint_result)
            return True

    return False


"""
This is no longer no man's land, but feel free to play and try new stuff
"""


def run_algorithms(timeseries, timeseries_name, end_timestamp, full_duration, debug, timeseries_filename):
    """
    Iteratively run algorithms.
    """

    timeseries_name = timeseries_name.replace('.json', '')
    timeseries_dir = timeseries_name.replace('.', '/')
    __results__ = abspath(join(dirname(__file__), '..', 'results'))
    results_dir = join(__results__ + "/" + timeseries_dir + "/" + str(end_timestamp))
    __data__ = abspath(join(dirname(__file__), '..', 'data'))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    start_analysis = int(time.time())
    if debug:
        print "Analysing " + timeseries_name

    try:
        for algorithm in ALGORITHMS:
            detected = ""
            x_vals = np.arange(len(timeseries))
            y_vals = np.array([y[1] for y in timeseries])
            # Match default graphite graph size
            plt.figure(figsize=(5.86, 3.08), dpi=100)
            plt.plot(x_vals, y_vals)

            # Start a couple datapoints in for the tail average
            for index in range(10, len(timeseries)):
                sliced = timeseries[:index]
                anomaly = globals()[algorithm](sliced, end_timestamp, full_duration, debug)

                # Point out the datapoint if it's anomalous
                if anomaly:
                    plt.plot([index], [sliced[-1][1]], 'ro')
                    detected = "DETECTED"

            if detected == "DETECTED":
                results_filename = join(results_dir + "/" + algorithm + "." + detected + ".png")
                if debug:
                    print "ANOMALY DETECTED - " + algorithm
            else:
                results_filename = join(results_dir + "/" + algorithm + ".png")

            plt.savefig(results_filename, dpi=100)
            if debug:
                print algorithm + " - " + results_filename
    except:
        print("Algorithm error: " + traceback.format_exc())

    end_analysis = int(time.time())
    seconds_to_run = end_analysis - start_analysis
    if debug:
        print "Analysis of " + timeseries_name + "at a full duration of " + str(full_duration) + " took " + str(seconds_to_run) + " seconds"
