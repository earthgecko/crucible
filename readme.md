## Crucible

![ts2.first_hour_average.DETECTED](ts2.first_hour_average.DETECTED.png?raw=true)

Crucible is a refinement and feedback suite for algorithm testing. It was
designed to be used to create anomaly detection algorithms, but it is very
simple and can probably be extended to work with your particular domain. It
evolved out of a need to test and rapidly generate standardized feedback for
iterating on anomaly detection algorithms.

## How it works

Crucible uses its library of timeseries in `data/` and tests all the
algorithms in algorithms.py on all these data. It builds the timeseries
datapoint by datapoint, and runs each algorithm at every step, as a way of
simulating a production environment. For every anomaly it detects, it draws a
red dot on the x value where the anomaly occurred. It then saves each graph to
disk in `results/<timeseries_dir>/<run_timestamp>/` where the
`<timeseries_dir>` are dotted namespace directories like the graphite
namespace directory structure with the plots from each algorithm, the json data
file (gzipped) and a graphite image file if the timeseries was surfaced from
graphite.

To be as fast as possible, Crucible launches a new process for each timeseries.

If you want to add an algorithm, simply create your algorithm in algorithms.py
and add it to settings.py as well so Crucible can find it. Crucible comes
loaded with a bunch of stock algorithms from an early
[Skyline](http://github.com/etsy/skyline) release and an additional newer
algorithm [detect_drop_off_cliff](examples/detect_drop_off_cliff/readme.md)
algorithm, but it's designed for you to write your own and test them.

## Dependencies
Standard python data science suite - everything is listed in algorithms.py

1. Install numpy, scipy, pandas, patsy, statsmodels, matplotlib.

2. You may have trouble with SciPy. If you're on a Mac, try:

* `sudo port install gcc48`
* `sudo ln -s /opt/local/bin/gfortran-mp-4.8 /opt/local/bin/gfortran`
* `sudo pip install scipy`

On Debian, apt-get works well for Numpy and SciPy. On CentOS, yum should do the
trick. If not, hit the Googles, yo.

## Instructions

Originally crucible was fixed to a `FULL_DURATION` of 86400 and removed the
results folder on every run, crucible has now been modified to run in a more
automated fashion on one timeseries at a time or many timeseries that have the
similar parameters.  This version is focused on automated discovery of the
timeseries parameters and graphite integration (see below).  This allows for
processing any historical timeseries for any `FULL_DURATION` period.
Just call:
`python src/crucible.py --debug --end_timestamp <epoch end timestamp of your timeseries> --full_duration <full_duration second>`
or the less verbose more automated mode:
`python src/crucible.py --end_timestamp <epoch end timestamp of your timeseries> --full_duration <full_duration second>`
In normal testing scenarios you may want to run `--debug`, the addition of this
option is related to enabling the automated running of crucible on demand.
The timestamp and full duration need to be passed to crucible solely to test
that the parameters are valid for algorithms, crucible will determine the
timeseries runtime parameters from the timeseries data itself.
Each run will put the data file/s, resultant plots and the original graphite
image (if there was one) in the `results/` folder.
Happy algorithming!!

## To add a timeseries:

It is possible to add timeseries manually, just create a json array of the form
`[[timestamp, datapoint], [timestamp, datapoint]]`. Put it in the `data/` folder
and pass the end_timestamp and full_duration parameters to crucible.py.

## Graphite integration:
There's a small tool to easily grab Graphite data and analyze it. Just call:
`python utils/graphite-grab.py --target "http://your_graphite.com/render/?<query_string>"`
or:
`python utils/graphite-grab.py --quiet --target "http://your_graphite.com/render/?<query_string>" --quiet`

Using a normal graphite graph url, the graphite-grab util will surface the data
from graphite for the metric and format it correctly and put it into `data/` for
you, along with a crucible.parameter file for the timeseries which calculates
the `end_timestamp` and `FULL_DURATION` from the data itself.  The required crucible
runtime parameters are outputted to pass to crucible.

## Contributions

It would be fantastic to have a robust library of canonical timeseries data.
Please, if you have a timeseries that you think a good anomaly detection
algorithm should be able to handle, share the love and add the timeseries to
the suite!

![forge](metalworker.jpg?raw=true)
