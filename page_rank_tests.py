import unittest
from pagerank import transition_model


class TestPageRank(unittest.TestCase):
    def test_transition_model(self):
        corpus = {
            "1.html": {"2.html", "3.html"},
            "2.html": {"3.html"},
            "3.html": {"2.html"},
        }
        page = "1.html"
        damping_factor = 0.85
        model = transition_model(corpus, page, damping_factor)
        self.assertEqual(
            model, {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}
        )


if __name__ == "__main__":
    unittest.main()
