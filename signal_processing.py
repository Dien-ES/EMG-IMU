import numpy as np
from scipy.signal import iirnotch, butter, lfilter
import matplotlib.pyplot as plt
import neurokit2 as nk
import seaborn as sns
from tqdm import tqdm
sns.set_style("whitegrid")

SENSORS = ['TP', 'LD', 'TFL', 'QF', 'GC', 'TA']
COLUMNS = [f'R_{s}'for s in SENSORS] + [f'L_{s}'for s in SENSORS]


class Signal:
    def __init__(self, data, xrange, name, fs=1259):
        self.data = data
        self.xrange = xrange
        self.name = name
        self.fs = fs

    def plot(self, color, alpha, label=None, axes=None, sharey=True):
        if axes is None:
            _, axes = plt.subplots(2, 6, figsize=(30, 10),
                                     sharex=True,
                                     sharey=sharey,
                                     constrained_layout=True)

        for idx in range(12):
            col = idx
            if idx > 5:
                idx -= 6
                ax = axes[1, idx]
            else:
                ax = axes[0, idx]

            if label is None:
                ax.set_title(COLUMNS[col], fontsize=25)
            else:
                ax.set_title(label, fontsize=25)
            ax.tick_params(labelsize=15)
            if idx == 0:
                ax.set_ylabel('Amplitude [mV]', fontsize=20)
            ax.set_xlabel('Time [s]', fontsize=20)
            sns.lineplot(ax=ax,
                         label=self.name,
                         color=color,
                         alpha=alpha,
                         x=self.xrange[:self.data.shape[0]],
                         y=self.data[:, col])


class EMG:
    def __init__(self, data, preprocessing=None, fs=1259):
        self.columns = COLUMNS
        xrange = np.arange(0, len(data), 1) / 1259

        proc_data = Processing(np.array(data), fs)
        self.raw = Signal(proc_data.data, xrange, 'Raw', fs)

        if preprocessing:
            proc_data.normalization()
            self.norm = Signal(proc_data.data,
                               xrange, 'Normalization', fs)

            try:
                proc_data.filter(method='despike')
            except:
                pass
            self.filtering = Signal(proc_data.data,
                                    xrange, f'Filter', fs)

            proc_data.rectification()
            self.rect = Signal(proc_data.data,
                               xrange, 'Rectification', fs)

#             try:
#                 proc_data.filter(method='despike')
#             except:
#                 pass
            proc_data.rms()
            self.rms = Signal(proc_data.data,
                              xrange * int(fs * 0.25), f'RMS', fs)

    def prep_comparison_plot(self):
        fig, axes = plt.subplots(2, 6, figsize=(30, 10),
                                 sharex=True,
                                 sharey=True,
                                 constrained_layout=True)

        self.norm.plot(color='grey', alpha=0.5, axes=axes)
        self.filtering.plot(color='yellow', alpha=0.5, axes=axes)
        self.rect.plot(color='green', alpha=0.5, axes=axes)
        self.rms.plot(color='red', alpha=1.0, axes=axes)


class IMU:
    def __init__(self, data, preprocessing=None, fs=148):
        self.data = data
        self.fs = fs

    def preprocessing(self):
        #
        self.data = None

    def plot(self):
        return None


class Processing:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs

    def normalization(self):
        self.data = self.data - self.data.mean(axis=0)

    def filter(self, method='bandpass'):
        if method == "notch":
            self.notch_pass_filter(60, 5)
        elif method == "bandpass":
            self.butter_bandpass_filter(20, 500, 4)
        elif method == "nsigma":
            self.sigma_filter()
        elif method == "despike":
            self.despike_filter()
        elif method == 'nk_clean':
            self.nk_clean()

    def rectification(self):
        self.data = abs(self.data)

    # filter
    def nk_clean(self):
        self.data = np.apply_along_axis(nk.emg_clean, axis=0,
                                        arr=self.data, sampling_rate=1259)

    def notch_pass_filter(self, center, interval):
        b, a = iirnotch(center, center / interval, self.fs)
        self.data = lfilter(b, a, self.data)

    def butter_bandpass_filter(self, lowcut, highcut, order):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.data = lfilter(b, a, self.data)

    # RMS
    def rms(self):
        frame = int(self.fs * 0.5)
        step = int(self.fs * 0.25)
        rms = []
        for i in range(frame, self.data.size, step):
            x = self.data[i - frame:i, :]
            rms.append(np.sqrt(np.mean(x ** 2, axis=0)))
        self.data = np.array(rms)

    def sigma_filter(self):
        filter_list = []
        for idx in range(self.data.shape[1]):
            data = self.data[:, idx].copy()
            x = np.arange(data.size)

            loop_data = data.copy()
            prev_size = 0
            nsigma = 2
            while prev_size != loop_data.size:
                mean = loop_data.mean()
                std = loop_data.std()
                mask = (loop_data < mean + nsigma * std) & (
                        loop_data > mean - nsigma * std)
                prev_size = loop_data.size
                loop_data = loop_data[mask]
                x = x[mask]
                nsigma *= 2

            # Reconstruct the mask
            mask = np.zeros_like(data, dtype=np.bool)
            mask[x] = True
            # This destroys the original data somewhat
            data[~mask] = data[mask].mean()
            filter_list += [data]

        self.data = np.array(filter_list).T

    def despike_filter(self, th=1.e-3):
        filter_list = []
        for idx in range(self.data.shape[1]):
            yi = self.data[:, idx].copy()
            y = np.copy(yi)  # use y = y1 if it is OK to modify input array
            n = len(y)
            x = np.arange(n)
            c = np.argmax(y)
            d = abs(np.diff(y))
            try:
                l = c - 1 - np.where(d[c - 1::-1] < th)[0][0]
                r = c + np.where(d[c:] < th)[0][0] + 1
                # for fit, use area twice wider then the spike
                if (r - l) <= 3:
                    l -= 1
                    r += 1
                s = int(round((r - l) / 2.))
                lx = l - s
                rx = r + s
                # make a gap at spike area
                xgapped = np.concatenate((x[lx:l], x[r:rx]))
                ygapped = np.concatenate((y[lx:l], y[r:rx]))
                # quadratic fit of the gapped array
                z = np.polyfit(xgapped, ygapped, 2)
                p = np.poly1d(z)
                y[l:r] = p(x[l:r])
                filter_list += [y]

            except:  # no spike, return unaltered array
                filter_list += [y]

        self.data = np.array(filter_list).T
