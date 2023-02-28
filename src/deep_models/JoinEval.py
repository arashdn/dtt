class JoinEval:

    #  remove_extra_in_source -> remove (a, b) from joined set if neither a nor b exists in golden.
    def __init__(self, joined, golden, remove_extra_in_source=True, is_directed=False, case_sensitive=False):
        for j in joined:
            assert len(j) == 2
        for g in golden:
            assert len(g) == 2

        self.is_directed = is_directed

        all_golden = set()

        new_golden = set()
        for g in golden:
            if not case_sensitive:
                g = (g[0].lower(), g[1].lower())
            all_golden.add(g[0])
            all_golden.add(g[1])
            l = sorted(list(g)) if not is_directed else g
            new_golden.add((l[0], l[1]))

        new_joined = set()
        for j in joined:
            if not case_sensitive:
                j = (j[0].lower(), j[1].lower())

            if remove_extra_in_source and j[0] not in all_golden and j[1] not in all_golden:
                continue
            l = sorted(list(j)) if not is_directed else j
            new_joined.add((l[0], l[1]))

        self.golden = new_golden
        self.joined = new_joined

        self._p = None
        self._r = None
        self._tp = None

    @property
    def tp(self):
        if self._tp is None:
            self._tp = self.joined.intersection(self.golden)
        return self._tp

    @property
    def precision(self):
        if self._p is None:
            if len(self.joined) == 0:
                self._p = 1
            else:
                self._p = len(self.tp) / len(self.joined)
        return self._p

    @property
    def recall(self):
        if self._r is None:
            if len(self.golden) == 0:
                self._r = 1
            else:
                self._r = len(self.tp) / len(self.golden)
        return self._r

    @property
    def f1(self):
        if self.precision + self.recall == 0:
            return 0.0
        return (2 * self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self):
        return f"P={self.precision}, R={self.recall}, F1={self.f1}, |joined|={len(self.joined)}, |golden|={len(self.golden)}, tp={len(self.tp)}"

    def short_str(self):
        return f"{self.precision},{self.recall},{self.f1}"
