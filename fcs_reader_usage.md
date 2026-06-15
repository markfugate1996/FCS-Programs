# fcs_reader.py — Usage Guide

## Requirements

The script requires Python 3.10 or later and two third-party packages:

```
pip install numpy pandas
```

`pandas` is optional — everything works without it except `to_dataframe()`.

---

## Setup

Place `fcs_reader.py` in the same folder as your data files, or anywhere
on your Python path. There is nothing to install; it is a single file you
import directly.

---

## Basic usage

```python
from fcs_reader import read_fcs

d = read_fcs("experiment.fcs")
print(d)
```

`print(d)` gives a formatted summary of the measurement:

```
FCSData — experiment.fcs
────────────────────────────────────────────────────
[ Measurement ]
  Timestamp              : 4/20/2026 7:44:13 PM
  Duration               : 120.000 s  (2.000 min)
  Clock frequency        : 20.194705 MHz
  Laser period           : 49.5179 ns
  Microtime bin width    : 12.0893 ps

[ Photon Statistics ]
  Ch0 photons            : 700,215
  Ch1 photons            : 716,167
  Ch0 count rate         : 5.84 kHz
  Ch1 count rate         : 5.97 kHz

[ Instrument ]
  Objective Mag             : 60
  Excitation Laser          : 600 nm, 100%
  ...
```

---

## The clock frequency

The reader automatically extracts the true measured clock frequency from the
binary header (offset `0x022`). You do not need to supply it manually.

The binary header also contains a separate rounded nominal clock value
(offset `0x050`, e.g. exactly 20,000,000 Hz) which the reader ignores.
The `clock_hz` parameter to `read_fcs()` is available if you ever need
to override the value, but in normal use this should not be necessary.

---

## Accessing the data

After loading, the `FCSData` object exposes the following:

### Experimental parameters

```python
d.params                  # dict of all metadata
d.params["clock_hz"]      # laser clock frequency in Hz
d.params["timestamp"]     # acquisition date and time
d.params["objective_mag"] # microscope objective magnification
d.params["channel_ch1"]   # dict of Ch1 detector settings
d.params["channel_ch2"]   # dict of Ch2 detector settings
```

### Photon arrival times

```python
d.ch0_times_s             # absolute arrival times for Ch0, in seconds
d.ch1_times_s             # absolute arrival times for Ch1, in seconds
```

These are computed from the stored macrotime differences and the clock
frequency. They are calculated on demand each time you access them, so
assign to a variable if you need them more than once:

```python
t0 = d.ch0_times_s        # do this rather than calling d.ch0_times_s twice
```

### Microtimes (fluorescence lifetime)

```python
d.ch0_micro               # TCSPC bin index per Ch0 photon, 0–4095
d.ch1_micro               # TCSPC bin index per Ch1 photon, 0–4095
d.ch0_micro_ns            # same, converted to nanoseconds
d.ch1_micro_ns            # same, converted to nanoseconds
```

### Timing properties

```python
d.laser_period_ns         # duration of one laser clock cycle in ns
d.microtime_resolution_ns # width of one TCSPC bin in ns (~12 ps typically)
d.duration_s              # total measurement duration in seconds
d.count_rate_ch0_hz       # mean photon count rate on Ch0 in Hz
d.count_rate_ch1_hz       # mean photon count rate on Ch1 in Hz
```

---

## Built-in analysis methods

### Binned intensity trace

Converts the photon arrival times into a regular time series of counts
per time bin — useful for plotting intensity vs time or for computing
FCS correlation functions.

```python
time_s, I0, I1 = d.bin_intensity(bin_width_s=1e-3)   # 1 ms bins
```

Returns three arrays of equal length: the left edge of each bin in
seconds, and the photon counts per bin for Ch0 and Ch1 respectively.
The bin width defaults to 1 ms if not specified.

### Lifetime histogram

Builds a TCSPC decay histogram from the microtime data — the starting
point for fluorescence lifetime analysis.

```python
bin_times_ns, counts = d.lifetime_histogram(channel=0)
```

`channel` is 0 or 1. Returns the time axis in nanoseconds (within one
laser period) and the photon counts per bin. By default uses all 4096
microtime bins; pass `n_bins` to coarsen.

### pandas DataFrame

Returns a DataFrame of the binned intensity trace with columns
`time_s`, `ch0`, `ch1`:

```python
df = d.to_dataframe(bin_width_s=1e-3)
```

Requires pandas to be installed.

---

## Running from the command line

The script can also be run directly to inspect a file without writing
any Python:

```
python fcs_reader.py experiment.fcs
python fcs_reader.py experiment.fcs 20194704.968582
```

The optional second argument is the clock frequency in Hz. This prints
the summary, array shapes, and the first few photon arrival times.

---

## A note on channel numbering

The instrument labels its detectors Channel 1 and Channel 2. In the
binary file and in this reader, these are indexed as Ch0 and Ch1
respectively — zero-based rather than one-based. So `d.ch0_times_s`
corresponds to the Channel 1 PMT, and `d.ch1_times_s` to Channel 2.
