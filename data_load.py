from signal_processing import *


class Subject:
    def __init__(self, group, sid):
        self.group = group
        self.sid = sid
        self.days = []

    def data_load(self):
        return 0


class Day:
    def __init__(self):
        self.BBS = None
        self.funcs = []
        self.indivs = []


class Movement:
    def __init__(self):
        self.motion = None
        self.EMGs = []
        self.IMUs = []


class FunctionalMovement(Movement):
    def __init__(self):
        super().__init__()


class IndividualMovement(Movement):
    def __init__(self):
        super().__init__()


