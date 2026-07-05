import unittest

from study_planner.tools import UnsafeExpressionError, safe_calculate


class ToolsTests(unittest.TestCase):
    def test_safe_calculate_supports_basic_arithmetic(self):
        self.assertEqual(safe_calculate("2 + 3 * (4 - 1)"), "11")

    def test_safe_calculate_rejects_function_calls(self):
        with self.assertRaises(UnsafeExpressionError):
            safe_calculate("__import__('os').system('echo unsafe')")

    def test_safe_calculate_rejects_names(self):
        with self.assertRaises(UnsafeExpressionError):
            safe_calculate("daily_hours * 4")


if __name__ == "__main__":
    unittest.main()
