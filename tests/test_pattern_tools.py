import unittest
import bento

data = bento.datasets.sample_data()


class TestPatterns(unittest.TestCase):
    def test_intracellular_patterns(self):
        bento.tl.intracellular_patterns(data)
        self.assertTrue(
            all(name in data.layers.keys() for name in bento._utils.PATTERN_NAMES)
        )
