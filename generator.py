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

    def next_double(self):
        return math.fabs(self.next_int()) / self.m

    """ generate random integer from range [start, end] both start and end are inclusive"""
    def next_int_from_to(self, start=0, end=10):
        rng = end - start + 1
        return start + int(math.fabs(self.next_int()) % rng)

    """ generates random double from range [start, end] both start and end are inclusive"""
    def next_double_from_to(self, start=0, end=10):
        rng = end - start
        return start + self.next_double() * rng

    """ generate samples from exponential distribution with parameter lambda defined as lmbda"""
    def next_from_exp(self, lmbda):
        u = self.next_double()
        while u == 0:
            u = self.next_double()
        return -math.log(u) / lmbda

    """generates samples from bernoulli distribution with parameter p. Bernoulli distribution is also known as 
    Alternating distribution"""
    def next_from_bernoulli(self, p):
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

    """ uses Knuths algorithm, tangible for small lambda"""
    def next_from_poisson(self, lmbda):
        L = math.exp(-lmbda)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            u = self.next_double()
            p *= u
        return k - 1

    """Returns a random sample from custom discrete distribution specified by weights"""
    def choices(self, values=(), weights=(), k=1):
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

