from signal_processing import *


class Subject:
    def __init__(self, group, sid):
        self.group = group
        self.sid = sid
        self.days = []

    def data_load(self):
        return 0


class Day:
    def __init__(self, day):
        self.day = day
        self.BBS = None
        self.funcs = []
        self.indivs = []

    def data_load(self):
        return 0


class Movement:
    def __init__(self, motion, session):
        self.session = session
        self.motion = motion
        self.signals = []

    def data_load(self):
        return 0


class FunctionalMovement(Movement):
    def __init__(self):
        super().__init__()


class IndividualMovement(Movement):
    def __init__(self):
        super().__init__()


