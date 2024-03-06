from src.synthseg_masks import encode_one_hot, decode_one_hot, synthseg_classes
import torch
import torch.testing as tt

import unittest


class TestOneHotEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.device=torch.device("cpu")
        self.example_tensor = torch.tensor([[
                [
                    [28, 52, 46, 42, 41],
                    [4,  3,  10,  2, 11,]
                ]
            ]]).to(self.device)

    def test_encoding(self):
        one_hot = encode_one_hot(self.example_tensor, self.device)
        self.assertEqual(list(one_hot.shape), [1, len(synthseg_classes), 2, 5])
        tt.assert_close(actual=one_hot.sum(dim=1, keepdim=True), expected=torch.ones(1, 1, 2, 5).to(self.device), rtol=0, atol=0)
        self.assertEqual(synthseg_classes.index(52), one_hot[0, :, 0, 1].argmax().item())
    
    def test_decoding_reverses_encoding(self):
        one_hot = encode_one_hot(self.example_tensor, self.device)
        decoded = decode_one_hot(one_hot, self.device)
        tt.assert_close(actual=decoded, expected=self.example_tensor, rtol=0, atol=0)


if __name__== '__main__':
    unittest.main()