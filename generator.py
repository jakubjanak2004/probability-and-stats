import math


class Generator:
    def __init__(self, seed, a=1103515245, c=12345, m=1 << 31):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m

    def next_int(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return int(self.seed)

    def next_int_from_to(self, start=0, end=10):
        return int(math.fabs(self.next_int()) % end)
