import numpy as np

MV2SENS = {'habd': 'TFL', 'ke': 'QF', 'apf': 'GC', 'adf': 'TA'}
SENS2MV = {'TFL': 'habd', 'QF': 'ke', 'GC': 'apf', 'TA': 'adf'}
SENSORS = ['TP', 'LD', 'TFL', 'QF', 'GC', 'TA']
COLUMNS = [f'R_{s}'for s in SENSORS] + [f'L_{s}'for s in SENSORS]


class ContRefer:
    def __init__(self, data):
        self.max_contr = {col: 1.0 for col in COLUMNS}
        self.sub_contr = {col: 1.0 for col in COLUMNS}
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


class RMS:
    def __init__(self, rms, contrf):
        self.rms = rms
        self.mvc = rms/np.array(list(contrf.max_contr.values()))
        self.submvc = rms/np.array(list(contrf.sub_contr.values()))

