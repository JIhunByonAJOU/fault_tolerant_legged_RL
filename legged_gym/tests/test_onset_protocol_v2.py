import unittest
from legged_gym.evaluation.onset_protocol_v2 import make_specs, tracking_summary

class TestOnsetProtocol(unittest.TestCase):
    def test_onsets_are_reproducible_bounded_and_matched_across_severity(self):
        specs = make_specs(123, 48, [0, .5, 1])
        self.assertEqual(specs, make_specs(123, 48, [0, .5, 1]))
        self.assertNotEqual(specs, make_specs(124, 48, [0, .5, 1]))
        self.assertEqual(len(specs), 12*3*48)
        self.assertTrue(all(100 <= s[2] <= 500 for s in specs))
        for i in range(0, len(specs), 3):
            self.assertEqual(len({s[2] for s in specs[i:i+3]}), 1)

    def test_early_death_cannot_improve_tracking_occupancy(self):
        self.assertEqual(tracking_summary([True]*10, [True]*2+[False]*8, 3), (.2, False))
        self.assertEqual(tracking_summary([True]*10, [True]*10, 3), (1., True))

    def test_brief_recovery_and_standing_are_distinguishable(self):
        self.assertEqual(tracking_summary([True]*3+[False]*7, [True]*10, 3), (.3, True))
        self.assertEqual(tracking_summary([False]*10, [True]*10, 3), (0., False))

if __name__ == '__main__':
    unittest.main()
