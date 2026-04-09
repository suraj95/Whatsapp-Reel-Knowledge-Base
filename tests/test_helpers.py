import unittest

from backend.helpers import strip_igsh_parameter


class StripIgshParameterTests(unittest.TestCase):
    def test_returns_original_when_empty(self):
        self.assertEqual(strip_igsh_parameter(""), "")

    def test_strips_query_params(self):
        url = "https://www.instagram.com/reel/abcd/?igsh=user1234"
        expected = "https://www.instagram.com/reel/abcd/"
        self.assertEqual(strip_igsh_parameter(url), expected)

    def test_strips_all_query_params_not_just_igsh(self):
        url = "https://www.instagram.com/reel/abcd/?utm_source=share&igsh=user1234"
        expected = "https://www.instagram.com/reel/abcd/"
        self.assertEqual(strip_igsh_parameter(url), expected)

    def test_strips_all_query_params_in_any_order(self):
        url = "https://www.instagram.com/reel/abcd/?a=1&igsh=user1234&b=2"
        expected = "https://www.instagram.com/reel/abcd/"
        self.assertEqual(strip_igsh_parameter(url), expected)

    def test_preserves_fragment(self):
        url = "https://www.instagram.com/reel/abcd/?igsh=user1234#section"
        expected = "https://www.instagram.com/reel/abcd/#section"
        self.assertEqual(strip_igsh_parameter(url), expected)

    def test_keeps_url_unchanged_when_no_query(self):
        url = "https://www.instagram.com/reel/abcd/"
        self.assertEqual(strip_igsh_parameter(url), url)


if __name__ == "__main__":
    unittest.main()
