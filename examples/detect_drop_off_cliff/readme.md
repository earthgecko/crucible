## detect_drop_off_cliff algorithm

![detect_drop_off_cliff](detect_drop_off_cliff.detect_drop_off_cliff.ts.json.png?raw=true)

The standard crucible (and skyline) algorithms are not great at detecting a
timeseries that "drops off a cliff".  Further to this in the skyline context,
even if one of the standard alogrithms did trigger a drop off cliff, it may not
be considered anomalous in the context of `CONSENSUS`.

## How it works

A timeseries is anomalous if the average of the last ten datapoints is `<trigger>`
times greater than the last data point.  This algorithm is most suited to
timeseries with most datapoints being > 100 (e.g high rate).  The arbitrary
`<trigger>` values become more noisy with lower value datapoints, but it still
matches drops off cliffs.

This has been tested across a larger number of variying timeseries and is very
effective at detecting a drop off cliff in high rate/volume metrics.

## The effectiveness of the standard algorithms in detecting cliff drops

The example timeseries used (detect_drop_off_cliff.ts.json) is a real world
data set and one can see towards the end of the timeseries the values drop off
a cliff.

```
[1446570420.0, 538630.0],
[1446570480.0, 410285.0],
[1446570540.0, 393762.0],
[1446570600.0, 391828.0],
[1446570660.0, 373251.0],
[1446570720.0, 363756.0],
[1446570780.0, 107053.0],
[1446570840.0, 95708.0],
[1446570900.0, 94761.0],
[1446570960.0, 94198.0],
[1446571020.0, 6964.0],
[1446571080.0, 8.0],
[1446571140.0, 16.0],
[1446571200.0, 10.0],
[1446571260.0, 5.0],
[1446571320.0, 5.0],
[1446571380.0, 14.0],
[1446571440.0, 3.0],
[1446571500.0, 9.0],
[1446571560.0, 6.0],
[1446571620.0, 5.0],
[1446571680.0, 3.0],
[1446571740.0, 4.0],
[1446571800.0, 2.0],
[1446571860.0, 7.0],
```

Analysing this same timeseries with each standard crucible (skyline) algorithm
highlights that no algorithm actually detects the drop off cliff pattern towards
the end of the timeseries, apart from a new experimental `detect_drop_off_cliff`

### detect_drop_off_cliff - __DETECTED__

![detect_drop_off_cliff](detect_drop_off_cliff.detect_drop_off_cliff.ts.json.png?raw=true)

### first_hour_average - not detected

![first_hour_average](first_hour_average.detect_drop_off_cliff.ts.json.png?raw=true)

### grubbs - not detected

![grubbs](grubbs.detect_drop_off_cliff.ts.json.png?raw=true)

### histogram_bins - not detected

![histogram_bins](histogram_bins.detect_drop_off_cliff.ts.json.png?raw=true)

### ks_test - not detected

![ks_test](ks_test.detect_drop_off_cliff.ts.json.png?raw=true)

### least_squares - not detected

![least_squares](least_squares.detect_drop_off_cliff.ts.json.png?raw=true)

### mean_subtraction_cumulation - not detected

![mean_subtraction_cumulation](mean_subtraction_cumulation.detect_drop_off_cliff.ts.json.png?raw=true)

### median_absolute_deviation - not detected

![median_absolute_deviation](median_absolute_deviation.detect_drop_off_cliff.ts.json.png?raw=true)

### stddev_from_average - not detected

![stddev_from_average](stddev_from_average.detect_drop_off_cliff.ts.json.png?raw=true)

### stddev_from_moving_average - not detected

![stddev_from_moving_average](stddev_from_moving_average.detect_drop_off_cliff.ts.json.png?raw=true)

## detect_drop_off_cliff in lower ranges

### detect_drop_off_cliff - working in the 100 range with skyline/boundary

skyline/boundary alert on a not so high range timeseries in the 100 range.

![detect_drop_off_cliff.in.the.100.range.boundary.alert](detect_drop_off_cliff.in.the.100.range.boundary.alert.png?raw=true)

The same timeseries hours later, the drop off cliff can be seen at ~02h12 and
the later timeseries the it is a variable data set.

![detect_drop_off_cliff.in.the.100.range.hours.later.comparison](detect_drop_off_cliff.in.the.100.range.hours.later.comparison.png?raw=true)
