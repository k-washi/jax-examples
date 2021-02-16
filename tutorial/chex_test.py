from absl.testing import parameterized, absltest
import chex

def fn(x, y):
    return x + y


class ExampleTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_ex1(self):
        var_fn = self.variant(fn)
        self.assertEqual(fn(1, 2), 3)
        self.assertEqual(var_fn(1, 2), fn(1, 2))
    
    
class ExampleParameterizedTest(parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ('case_positive', 1, 2, 3),
        ('case_negative', -1, -2, -3),
    )
    def test(self, arg_1, arg_2, expected):
        @self.variant
        def var_fn(x, y):
            return x + y

        self.assertEqual(var_fn(arg_1, arg_2), expected)

if __name__ == "__main__":
    absltest.main()