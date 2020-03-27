import unittest

import utils


class UtilsTest(unittest.TestCase):
    def test_conditional_proxy(self):
        counter = [0]

        class Increment:
            def call(self, value):
                counter[0] += value

        proxy = utils.ConditionalProxy(Increment(), True)

        proxy.call(10)

        self.assertEqual(counter[0], 10)

        proxy = utils.ConditionalProxy(Increment(), False)

        # This should *not* be forwarded to the increment object
        proxy.call(42)

        self.assertEqual(counter[0], 10)

        with self.assertRaises(AttributeError):
            proxy = utils.ConditionalProxy(Increment(), True)

            proxy.blah(1, 2, 3)

        post_hook_called = [False]

        def post_hook():
            post_hook_called[0] = True

        proxy = utils.ConditionalProxy(Increment(), True, post_hook=post_hook)

        proxy.call(-10)

        self.assertEqual(counter[0], 0)
        self.assertTrue(post_hook_called[0])


if __name__ == '__main__':
    unittest.main()
