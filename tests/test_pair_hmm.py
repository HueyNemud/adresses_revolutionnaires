import unittest

import numpy as np

from lib import pair_hmm

# Transitions proches de celles estimées sur 1807/1808 ; rapport de
# vraisemblance défavorable à sim = 0,72 (log ≈ −2,8), très défavorable à 0,4.
TRANSITIONS = np.array([[0.75, 0.18, 0.07], [0.6, 0.24, 0.16], [0.84, 0.0, 0.16]])
MODEL = pair_hmm.PairHmm(
    TRANSITIONS,
    pair_hmm.normalize(np.linspace(0.01, 1, pair_hmm.N_BINS) ** 20, 0.0),
    pair_hmm.normalize(np.linspace(1, 0.01, pair_hmm.N_BINS), 0.0),
)


def posterior(similarity) -> np.ndarray:
    return pair_hmm.window_posteriors(np.array(similarity, dtype=float), MODEL).posterior


class WindowPosteriorTests(unittest.TestCase):
    def test_flanked_pair_matched_despite_unfavourable_ratio(self):
        """Cas Pagès : seule entrée non appariée de chaque côté entre deux
        ancres, similarité moyenne : le contexte fait pencher vers la paire."""
        self.assertLess(MODEL.log_odds(np.array([0.72]))[0], 0)
        self.assertGreater(posterior([[0.72]])[0, 0], 0.5)

    def test_same_pair_in_wide_gap_is_less_probable(self):
        wide = np.full((4, 4), 0.3)
        wide[1, 2] = 0.72
        self.assertLess(posterior(wide)[1, 2], posterior([[0.72]])[0, 0])
        self.assertLess(posterior(wide)[1, 2], 0.5)

    def test_low_similarity_not_matched_even_flanked(self):
        self.assertLess(posterior([[0.4]])[0, 0], 0.01)

    def test_posteriors_sum_to_at_most_one(self):
        rng = np.random.default_rng(0)
        post = posterior(rng.uniform(0.3, 1.0, (5, 4)))
        self.assertTrue(np.all(post.sum(axis=0) <= 1 + 1e-9))
        self.assertTrue(np.all(post.sum(axis=1) <= 1 + 1e-9))

    def test_empty_and_one_sided_windows(self):
        for shape in ((0, 0), (2, 0), (0, 3)):
            result = pair_hmm.window_posteriors(np.zeros(shape), MODEL)
            self.assertEqual(result.posterior.shape, shape)
            self.assertTrue(np.isfinite(result.log_likelihood))

    def test_expected_transitions_match_path_length(self):
        """Un chemin de k X puis l Y, ou avec des M : k + l − (#M) + 1
        transitions, fin comprise ; ici 1 × 1 : 2 transitions (M M) ou 3 (X Y)."""
        counts = pair_hmm.window_posteriors(np.array([[0.72]]), MODEL).transitions
        p = posterior([[0.72]])[0, 0]
        self.assertAlmostEqual(counts.sum(), 2 * p + 3 * (1 - p))


class DecodeTests(unittest.TestCase):
    def test_monotone_one_to_one(self):
        rng = np.random.default_rng(1)
        similarity = rng.uniform(0.2, 0.6, (6, 6))
        for i, j in ((0, 0), (2, 1), (3, 3), (5, 4)):
            similarity[i, j] = 0.97
        pairs = pair_hmm.decode(posterior(similarity))
        self.assertEqual(pairs, [(0, 0), (2, 1), (3, 3), (5, 4)])


class FitTests(unittest.TestCase):
    def test_em_increases_likelihood(self):
        rng = np.random.default_rng(2)
        windows = [rng.uniform(0.2, 0.6, (rng.integers(0, 3), rng.integers(0, 3))) for _ in range(40)]
        windows += [np.array([[0.95]]), np.array([[0.9]]), np.array([[0.8]])] * 10
        same = pair_hmm.histogram(np.array([0.95] * 50 + [0.85] * 10 + [0.75] * 3))
        different = pair_hmm.histogram(rng.uniform(0.2, 0.7, 500))
        fitted = pair_hmm.fit(windows, same, different, tolerance=0.0, max_iterations=10)
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(fitted.log_likelihoods, fitted.log_likelihoods[1:])))
        self.assertTrue(np.allclose(fitted.model.transitions.sum(axis=1), 1))
        self.assertEqual(fitted.model.transitions[pair_hmm.Y, pair_hmm.X], 0)

    def test_likelihood_ratio_monotone(self):
        """Une classe vide et basse ne doit pas paraître favorable."""
        different = pair_hmm.normalize(np.r_[np.zeros(2), np.ones(pair_hmm.N_BINS - 2)], 0.0)
        different = pair_hmm.normalize(different, 1e-3)
        same = pair_hmm.estimate_same(pair_hmm.histogram(np.array([0.95] * 20 + [0.8] * 5)), different)
        ratio = same / different
        self.assertTrue(np.all(np.diff(ratio) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
