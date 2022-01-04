import numpy as np
from scipy.signal import iirnotch, butter, lfilter


class Signal:
    def __init__(self, data, xrange, name, fs=1259):
        self.data = data
        self.xrange = xrange
        self.name = name
        self.fs = fs

    def plot(self, color, alpha, axes=None, sharey=True):
        if not axes:
            _, axes = plt.subplots(2, 6, figsize=(30, 10),
                                     sharex=True,
                                     sharey=sharey,
                                     constrained_layout=True)

        for idx in range(12):
            col = idx
            ax = axes[0, idx]
            if idx > 5:
                idx -= 6
                ax = axes[1, idx]

            ax.set_title(data.iloc[:, col].columns[0], fontsize=25)
            ax.tick_params(labelsize=15)
            if idx == 0:
                ax.set_ylabel('Amplitude (mV)', fontsize=20)
            ax.set_xlabel('Time (s)', fontsize=20)
            sns.lineplot(ax=ax,
                         label=self.name,
                         color=color,
                         alpha=alpha,
                         x=self.xrange[:self.data.shape[0]],
                         y=self.data.iloc[:, col])


class EMG:
    def __init__(self, data, filter='despike', method='1', fs=1259):
        xrange = np.arange(0, len(data) * 1 / 1259, 1 / 1259)
        proc_data = Processing(data, fs)

        self.raw = Signal(data, xrange, 'Raw', fs)
        self.norm = Signal(proc_data.normalization(),
                           xrange, 'Normalization', fs)

        self.filtering = Signal(proc_data.filter(filter),
                                xrange, f'Filter_({filter})', fs)
        self.rms = Signal(proc.data.rms(method),
                          xrange * int(fs * 0.2), f'RMS_({method})', fs)


class IMU:
    def __init__(self, data, fs=148):
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
        return self.data

    def filter(self, method='despike'):
        if method == "notch":
            return self.notch_pass_filter(60, 5)
        elif method == "bandpass":
            return self.butter_bandpass_filter(10, 500, 4)
        elif method == "nsigma":
            return self.sigma_filter()
        elif method == "despike":
            return self.despike_filter()

    def rectification(self):
        self.data = abs(self.data)
        return self.data

    def rms(self, method='rms_1'):
        if method == '1':
            return self.rms_1()
        else:
            return self.rms_2()

    # filter
    def notch_pass_filter(self, center, interval):
        b, a = signal.iirnotch(center, center / interval, self.fs)
        self.data = signal.lfilter(b, a, self.data)
        return self.data

    def butter_bandpass_filter(self, lowcut, highcut, order):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.data = lfilter(b, a, self.data)
        return self.data

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
        return self.data

    def despike_filter(self, th=1.e-8):
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
        return self.data

    # RMS
    def rms_1(self):
        frame = int(self.fs * 0.5)
        step = int(self.fs * 0.2)
        rms = []
        for i in range(frame, self.data.size, step):
            x = self.data[i - frame:i, :]
            rms.append(np.sqrt(np.mean(x ** 2, axis=0)))
        self.data = np.array(rms)
        return self.data

    def rms_2(self, window_size=int(1259/6)):
        data = np.power(self.data, 2)
        window = np.ones(window_size) / float(window_size)
        window_rms = np.sqrt(np.convolve(data, window, 'same'))
        self.data = np.apply_along_axis(window_rms, 1, data, window_size)
        return self.data