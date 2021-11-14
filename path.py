class Path:
    def cast(self):
        h = 1

    def evaluate(self):
        h = 1

    def accum_radiance(self):
        h = 1

    def update_throughput(self):
        h = 1

    def russian_roulette(self):
        h = 1

    def increment(self):
        self.next_task()
        return self.is_alive

    def __main__(self):
        self.next_task = self.cast
        self.is_alive = True
