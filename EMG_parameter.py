import numpy as np
from tqdm import tqdm
import pickle
from signal_processing import *
from data_load import *

MV2SENS = {'habd': 'TFL', 'ke': 'QF', 'apf': 'GC', 'adf': 'TA'}
SENS2MV = {'TFL': 'habd', 'QF': 'ke', 'GC': 'apf', 'TA': 'adf'}
SENSORS = ['TP', 'LD', 'TFL', 'QF', 'GC', 'TA']
COLUMNS = [f'R_{s}'for s in SENSORS] + [f'L_{s}'for s in SENSORS]


class ContRefer:
    def __init__(self, data):
        self.max_contr = {col: np.nan for col in COLUMNS}
        self.sub_contr = {col: np.nan for col in COLUMNS}
        self.referece(data)

    def referece(self, data):
        for i in range(len(data)):
            emg = data[i].signals[0].emg
            rms = emg.rms.data

            motion_split = data[i].motion.split('_')
            side = motion_split[0].upper()
            mv = motion_split[1]
            power = motion_split[2]

            col = f'{side}_{MV2SENS[mv]}'
            cols = emg.columns
            value = np.nanmax(rms, axis=0)[cols.index(col)]
            if power == 'max':
                self.max_contr[col] = value
            else:
                self.sub_contr[col] = value


class Parameter:
    def __init__(self, rms, contrf, info):
        xrange = np.arange(0, len(rms) * 1 / 1259, 1 / 1259) * int(1259 * 0.25)
        self.info = info
        self.rms = Signal(rms, xrange, 'RMS')
        self.mvc = Signal(self.ratio(contrf.max_contr), xrange, 'MVC')
        self.submvc = Signal(self.ratio(contrf.sub_contr), xrange, 'subMVC')
        self.rms_amp = self.amplitude(self.rms)
        self.mvc_amp = self.amplitude(self.mvc)
        self.submvc_amp = self.amplitude(self.submvc)

    def ratio(self, contraction):
        return self.rms.data/np.array(list(contraction.values()))

    def amplitude(self, signal):
        amps = []
        for i in [1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
            amp = signal.data[:60].mean(axis=0) * i <= \
                  np.nanmax(signal.data, axis=0)
            amps += [amp]
        return np.array(amps)

    def comparison_plot(self):
        fig, axes = plt.subplots(2, 6, figsize=(30, 10),
                                 sharex=True,
                                 sharey=True,
                                 constrained_layout=True)

        self.rms.plot(color='blue', alpha=1.0, axes=axes)
        self.mvc.plot(color='red', alpha=1.0, axes=axes)
        self.submvc.plot(color='green', alpha=1.0, axes=axes)


def subject_movement_params(group, sid):
    subject = Subject(group, sid)
    print(f"subject load complete...")

    indiv_params = []
    func_params = []
    for data in subject.days:
        print(f"{data.day} day start...")
        contrf = ContRefer(data.indivs)

        indiv_params = movement_params(indiv_params, data.indivs,
                                       contrf, data.BBS)
        func_params = movement_params(func_params, data.funcs,
                                      contrf, data.BBS)
    print()
    return indiv_params, func_params


def movement_params(params, movements, contrf, BBS):
    if movements is None:
        return params

    for mv in movements:
        for signal in mv.signals:
            rms = signal.emg.rms.data
            info = {'sid': signal.sid, 'group': signal.group,
                    'BBS': BBS, 'day': signal.day, 'motion': signal.motion}
            parameter = Parameter(rms, contrf, info)
            params += [parameter]
    return params


def total_params(is_save=True):
    total_indiv_params, total_func_params = [], []
    for group in ['Healthy', 'Disabled']:
        if group == 'Healthy':
            end_sid = 20
            sid_list = range(1, end_sid + 1)
        else:
            end_sid = 22
            sid_list = list(range(1, end_sid + 1))
            sid_list.remove(13)
            sid_list.remove(17)
        for sid in sid_list:
            print(f"Group: {group},"
                  f" Subject id: {sid} of {end_sid} subject loading...")
            indiv_params, func_params = subject_movement_params(group, sid)
            total_indiv_params += indiv_params
            total_func_params += func_params

    if is_save:
        with open('../parameter/total_indiv_params.pkl', 'wb') as f:
            pickle.dump(total_indiv_params, f)
        with open('../parameter/total_func_params.pkl', 'wb') as f:
            pickle.dump(total_func_params, f)

    return total_indiv_params, total_func_params
