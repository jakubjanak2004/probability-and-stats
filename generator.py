import math


class Generator:
    def __init__(self, seed):
        self.seed = seed

    def next_int(self) -> int:
        """ Generate next random integer"""
        pass

    def next_double(self) -> float:
        """ Generate next random double"""
        pass

    def next_int_from_to(self, start=0, end=10):
        """ generate random integer from range [start, end] both start and end are inclusive"""
        rng = end - start + 1
        return start + int(math.fabs(self.next_int()) % rng)

    def next_double_from_to(self, start=0, end=10):
        """ generates random double from range [start, end] both start and end are inclusive"""
        rng = end - start
        return start + self.next_double() * rng

    def next_from_exp(self, lmbda):
        """ generate samples from exponential distribution with parameter lambda defined as lmbda"""
        u = self.next_double()
        while u == 0:
            u = self.next_double()
        return -math.log(u) / lmbda

    def next_from_bernoulli(self, p):
        """generates samples from bernoulli distribution with parameter p. Bernoulli distribution is also known as
    Alternating distribution"""
        return 1 if self.next_double() < p else 0

    def next_from_binomial(self, n, p):
        return sum(self.next_from_bernoulli(p) for _ in range(n))

    def next_from_geometric(self, p):
        u = self.next_double()
        return math.floor(math.log(1 - u) / math.log(1 - p))

    # todo check if this makes sense
    def next_from_normal(self, mu=0, sigma=1):
        u1 = self.next_double()
        u2 = self.next_double()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z

    def next_from_poisson(self, lmbda):
        """ uses Knuths algorithm, tangible for small lambda"""
        L = math.exp(-lmbda)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            u = self.next_double()
            p *= u
        return k - 1

    def choices(self, values=(), weights=(), k=1):
        """Returns a random sample from custom discrete distribution specified by weights"""
        if not values or not weights or len(weights) != len(values):
            raise ValueError("Values and Weights must be non-empty and of same length")

        # Normalise the weights if not
        total_weight = sum(weights)
        n_weights = [w / total_weight for w in weights]

        # Create cumulative distribution
        cumulative = []
        cum_sum = 0
        for w in n_weights:
            cum_sum += w
            cumulative.append(cum_sum)

        samples = []
        for _ in range(k):
            r = self.next_double()
            for i, threshold in enumerate(cumulative):
                if r <= threshold:
                    samples.append(values[i])
                    break
        return samples


class LCGenerator(Generator):
    def __init__(self, seed, a=1103515245, c=12345, m=1 << 31):
        super().__init__(seed)
        self.a = a
        self.c = c
        self.m = m

    def next_int(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return int(self.seed)

    def next_double(self):
        return math.fabs(self.next_int()) / self.m


# todo implement MT random numbers generator
class MTGenerator(Generator):
    def __init__(self, seed):
        # Constants for MT19937
        super().__init__(seed)
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253

        # Masks
        self.lower_mask = (1 << self.r) - 1  # Lower r bits
        self.upper_mask = (~self.lower_mask) & 0xFFFFFFFF  # Upper w - r bits

        # State array
        self.mt = [0] * self.n
        self.index = self.n  # Force twist on first use
        self.mt[0] = seed

        for i in range(1, self.n):
            self.mt[i] = (self.f * (self.mt[i - 1] ^ (self.mt[i - 1] >> (self.w - 2))) + i) & 0xFFFFFFFF

    def twist(self):
        for i in range(self.n):
            x = (self.mt[i] & self.upper_mask) + (self.mt[(i + 1) % self.n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        if self.index >= self.n:
            self.twist()

        y = self.mt[self.index]

        # Tempering
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= (y >> self.l)

        self.index += 1
        return y & 0xFFFFFFFF

    def random(self):
        """Return a floating point number in [0, 1)."""
        return self.extract_number() / 2 ** 32

    def next_int(self):
        value = self.extract_number()
        # Convert to signed 32-bit int if needed
        if value >= 0x80000000:
            value -= 0x100000000
        return value

    def next_double(self):
        # Combine two 32-bit values to get 53 bits of randomness
        a = self.extract_number() >> 5  # 27 bits
        b = self.extract_number() >> 6  # 26 bits
        return (a * (2 ** 26) + b) / float(2 ** 53)


# todo implement quasi random number generator
class QuasiGenerator(Generator):
    def __init__(self, seed):
        super().__init__(seed)

    def next_int(self):
        pass

    def next_double(self):
        pass
