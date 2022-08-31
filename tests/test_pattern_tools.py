import unittest
import bento

data = bento.datasets.sample_data()


class TestPatterns(unittest.TestCase):
    def test_lp(self):
        bento.tl.lp(data)
        self.assertTrue(
            all(name in data.layers.keys() for name in bento._utils.PATTERN_NAMES)
        )
        # Make sure the pattern layers are not empty
        self.assertFalse(all(data.to_df(name).isna().all().all() for name in bento.PATTERN_NAMES))
