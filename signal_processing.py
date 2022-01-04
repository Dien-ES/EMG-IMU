import glob
import os
data_path = glob.glob("../../../../Database/EMG_IMU")


class Signal:
    def __init__(self):
        self.data = None


class EMG(Signal):
    def __init__(self):
        super().__init__()


class IMU(Signal):
    def __init__(self):
        super().__init__()