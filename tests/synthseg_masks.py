from src.synthseg_masks import encode_one_hot, decode_one_hot, reduce_mask_to_ventricles, ventricles, synthseg_classes
import torch
import torch.testing as tt

import unittest


class TestOneHotEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.device=torch.device("cpu")
        self.example_tensor = torch.tensor([[[
                [
                    [28, 52, 46, 42, 41],
                    [4,  3,  10,  2, 11,]
                ]
            ]]]).to(self.device).int()
        assert len(list(self.example_tensor.shape)) == 5

    def test_encoding(self):
        one_hot = encode_one_hot(self.example_tensor, self.device)
        print(one_hot.shape)
        self.assertEqual(list(one_hot.shape), [1, len(synthseg_classes), 1, 2, 5])
        tt.assert_close(actual=one_hot.sum(dim=1, keepdim=True), expected=torch.ones(1, 1, 1, 2, 5).to(self.device), rtol=0, atol=0)
        self.assertEqual(synthseg_classes.index(52), one_hot[0, :, 0, 0, 1].argmax().item())
    
    def test_decoding_reverses_encoding(self):
        one_hot = encode_one_hot(self.example_tensor, self.device)
        decoded = decode_one_hot(one_hot, self.device)
        tt.assert_close(actual=decoded, expected=self.example_tensor, rtol=0, atol=0)

class TestReduceMaskToVentricles(unittest.TestCase):
    def setUp(self) -> None:
        self.device=torch.device("cpu")
        self.example_mask = torch.tensor([[[
                [
                    [1, 2, 0, 4, 5],
                    [10,  15,  14,  2, 11],
                    [43, 43, 11, 43, 44]
                ]
            ]]]).to(self.device)
        
        self.ventricle_indices = torch.tensor([
            4, 5, 
            14, 15,
            43, 44
        ])
    
    def test_mask_is_boolean(self):
        result = reduce_mask_to_ventricles(self.example_mask, self.device)
        self.assertTrue(torch.logical_or(result == 0,  result == 1).all(), "Ventricle mask should be boolean mask")
        
    def test_all_ventricle_pixels_in_mask(self):
        result = reduce_mask_to_ventricles(self.example_mask, self.device)
        inverse_mask = ~result
        self.assertFalse(torch.isin(self.example_mask * inverse_mask, self.ventricle_indices).any()), "Ensure no ventricle pixel remains in original mask"

    def test_preserves_shape(self):
        result = reduce_mask_to_ventricles(self.example_mask, self.device)
        self.assertEqual(self.example_mask.shape, result.shape)
    
    def test_correct_values(self):
        result = reduce_mask_to_ventricles(self.example_mask, self.device)
        expected = torch.tensor([[[
            [
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 0],
                [1, 1, 0, 1, 1]
            ]
        ]]]).bool()
        tt.assert_allclose(result, expected, 0.0, 0.0, msg="Value should be correct")


if __name__== '__main__':
    unittest.main()