import unittest

import numpy as np

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
from zipftok.zipf import fit_loglog, r_squared, rank_frequency, segmented_fit, zipf_score


class TestRankFrequency(unittest.TestCase):
    def test_sorts_and_drops_zeros(self):
        r, f = rank_frequency([3, 0, 10, 1, 0, 5])
        np.testing.assert_array_equal(f, [10, 5, 3, 1])
        np.testing.assert_array_equal(r, [1, 2, 3, 4])

    def test_min_count(self):
        _, f = rank_frequency([3, 1, 10, 1, 5], min_count=2)
        np.testing.assert_array_equal(f, [10, 5, 3])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            rank_frequency([1, -1])


class TestFit(unittest.TestCase):
    def test_exact_power_law(self):
        for k in (0.8, 1.0, 1.3):
            ranks = np.arange(1, 5001)
            counts = 1e7 * ranks ** (-k)
            fit = zipf_score(counts)
            self.assertAlmostEqual(fit.r2, 1.0, places=10)
            self.assertAlmostEqual(fit.slope, -k, places=8)
            self.assertAlmostEqual(fit.exponent, k, places=8)
            self.assertEqual(fit.n_types, 5000)

    def test_order_of_counts_is_irrelevant(self):
        rng = np.random.default_rng(0)
        counts = rng.zipf(1.7, size=3000).astype(float)
        a = zipf_score(counts)
        b = zipf_score(rng.permutation(counts))
        self.assertAlmostEqual(a.r2, b.r2, places=12)

    def test_r2_matches_squared_pearson(self):
        rng = np.random.default_rng(1)
        counts = np.round(1e5 * np.arange(1, 2001) ** -1.1 * rng.lognormal(0, 0.3, 2000)) + 1
        r, f = rank_frequency(counts)
        fit = fit_loglog(r, f)
        rho = np.corrcoef(np.log10(r), np.log10(f))[0, 1]
        self.assertAlmostEqual(fit.r2, rho ** 2, places=10)

    def test_curvature_lowers_r2(self):
        # a power law with an exponential cut-off (the "small vocabulary" shape of Figure 1)
        ranks = np.arange(1, 3001)
        straight = 1e6 * ranks ** -1.0
        curved = 1e6 * ranks ** -0.4 * np.exp(-ranks / 300)
        self.assertGreater(zipf_score(straight).r2, zipf_score(curved).r2 + 0.1)

    def test_zipf_reference(self):
        ranks = np.arange(1, 1001)
        self.assertAlmostEqual(zipf_score(1e6 / ranks, reference="zipf").r2, 1.0, places=10)
        steep = zipf_score(1e6 * ranks ** -1.6, reference="zipf")
        self.assertLess(steep.r2, 0.9)
        self.assertEqual(steep.slope, -1.0)

    def test_log_binned(self):
        ranks = np.arange(1, 10001)
        fit = zipf_score(1e8 * ranks ** -1.2, weighting="log_binned", n_bins=50)
        self.assertAlmostEqual(fit.r2, 1.0, places=8)
        self.assertAlmostEqual(fit.slope, -1.2, places=6)

    def test_max_rank(self):
        ranks = np.arange(1, 2001)
        f = np.where(ranks <= 1000, 1e6 / ranks, 1e3 * (ranks / 1000.0) ** -3)
        fit = fit_loglog(ranks, f, max_rank=1000)
        self.assertAlmostEqual(fit.r2, 1.0, places=10)
        self.assertEqual(fit.n_types, 1000)

    def test_needs_two_points(self):
        with self.assertRaises(ValueError):
            zipf_score([5])

    def test_r_squared_constant(self):
        self.assertEqual(r_squared(np.ones(5), np.ones(5)), 1.0)


class TestSegmented(unittest.TestCase):
    def test_recovers_two_regimes(self):
        ranks = np.arange(1, 20001, dtype=float)
        brk = 1500
        logf = np.where(ranks < brk, 7 - 0.9 * np.log10(ranks),
                        7 - 0.9 * np.log10(brk) - 2.0 * (np.log10(ranks) - np.log10(brk)))
        seg = segmented_fit(ranks, 10 ** logf, n_candidates=400)
        self.assertGreater(seg.r2, 0.9999)
        self.assertLess(abs(np.log10(seg.breakpoint_rank) - np.log10(brk)), 0.05)
        self.assertAlmostEqual(seg.slope_head, -0.9, places=2)
        self.assertAlmostEqual(seg.slope_tail, -2.0, places=2)
        single = fit_loglog(ranks, 10 ** logf)
        self.assertGreater(seg.r2, single.r2)


if __name__ == "__main__":
    unittest.main()
