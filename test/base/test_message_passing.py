"""Test the message passing module."""
import pytest
import torch
import numpy as np
import unittest


from topomodelx.base.message_passing import MessagePassing

import toponetx as tnx

<<<<<<< HEAD

class TestMessagePassing(unittest.TestCase):
    """Test the message passing module."""

    def test_init(self):
        """Test the initialization of the message passing module."""
        mp = MessagePassing()
        


    def test_forward(self):
        """Test the weights."""
        a = np.array([[-1.,  0. ],
                      [ 1.,- 1],
                      [ 0.,  1.]])
        
        a_ = np.array([[-1.,  0. ],
                      [ 1.,- 0.5],
                      [ 0.,  1.]])
        mp = MessagePassing()
        x = torch.rand(2,10)
        x_out = mp(x,a)
        expected = np.matmul(a,x.numpy())
        np.testing.assert_array_almost_equal( x_out.numpy() ,expected)

    def test__propagate(self):
        """Test the _propagate function ."""
        a = np.array([[-1.,  0. ],
                      [ 1.,- 1],
                      [ 0.,  1.]])
        
        a_ = np.array([[-1.,  0. ],
                      [ 1.,- 0.5],
                      [ 0.,  1.]])
        mp = MessagePassing()
        x = torch.rand(2,10)
        x_out = mp._propagate(x,a)
        expected = np.matmul(a,x.numpy())
        np.testing.assert_array_almost_equal( x_out.numpy() ,expected)
       
        # test without sign
        x_out = mp._propagate(x,a,aggregate_sign=False)
        expected = np.matmul(abs(a) ,x.numpy())
        np.testing.assert_array_almost_equal( x_out.numpy() ,expected)


        # test without value 
        x_out = mp._propagate(x,a_, aggregate_value= False ) 
        expected = np.matmul(a ,x.numpy()) 
        np.testing.assert_array_almost_equal( x_out.numpy() ,expected) 


        # test without value and without sign
        x_out = mp._propagate(x,a_, aggregate_sign=False, aggregate_value= False ) # vaules of a_ will be ignored
        expected = np.matmul(abs(a) ,x.numpy()) 
        np.testing.assert_array_almost_equal( x_out.numpy() ,expected)

       

    def test_propagate(self):
        """Test the propagate function ."""

        a = np.array([[-1.,  0.],
                      [ 1., -1.],
                      [ 0.,  1.]])

        b = np.array([[1., 1.],
                      [1., 1.],
                      [0., 1.]])

        mp = MessagePassing()
        x = torch.rand(2,10)
        x_out = mp.propagate(x,[a,b])
        
        expected_a = np.matmul(a,x.numpy())
        expected_b = np.matmul(b,x.numpy())
        
        np.testing.assert_array_almost_equal( x_out[0].numpy() ,expected_a)
        np.testing.assert_array_almost_equal( x_out[1].numpy() ,expected_b)
  
        
        pass

    def test_update(self):
        pass

    def test_message(self):
        pass
    
    def test_aggregate(self):
        pass
    def test_get_i(self):
        pass    
    def test_get_j(self):
        pass 
    
if __name__ == "__main__":
    unittest.main()    
=======
class TestMessagePassing:
    """Test the MessagePassing class."""

    def setup_method(self):
        """Make message_passing object."""
        self.x_source = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        self.x_target = torch.tensor([[1, 2], [3, 4]]).float()

        self.neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        ).float()

        self.neighborhood_r_to_s = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(2, 3),
        )

        self.mp = MessagePassing()
        self.mp_with_att = MessagePassing(att=True)
        self.mp_with_att.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        gain = 1.0
        with pytest.raises(RuntimeError):
            self.mp.initialization = "invalid"
            self.mp.reset_parameters(gain=gain)

        # Test xavier_uniform
        self.mp.initialization = "xavier_uniform"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

        # Test xavier_normal
        self.mp.initialization = "xavier_normal"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

        # Test with attention weights & xavier_uniform
        self.mp_with_att.initialization = "xavier_uniform"
        self.mp_with_att.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp_with_att.reset_parameters(gain=gain)
        assert self.mp_with_att.att_weight.shape == (4,)

        # Test with attention weights & xavier_normal
        self.mp_with_att.initialization = "xavier_normal"
        self.mp_with_att.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp_with_att.reset_parameters(gain=gain)
        assert self.mp_with_att.att_weight.shape == (4,)

    def test_attention(self):
        """Test attention."""
        # Test with source & target on the same cells
        neighborhood = self.neighborhood.coalesce()
        (
            self.mp_with_att.target_index_i,
            self.mp_with_att.source_index_j,
        ) = neighborhood.indices()
        n_messages = len(
            self.mp_with_att.target_index_i,
        )
        result = self.mp_with_att.attention(self.x_source)
        assert result.shape == (n_messages,)

        # Test with source & target on different cells
        neighborhood_r_to_s = self.neighborhood_r_to_s.coalesce()
        (
            self.mp_with_att.target_index_i,
            self.mp_with_att.source_index_j,
        ) = neighborhood_r_to_s.indices()
        n_messages = len(
            self.mp_with_att.target_index_i,
        )
        result = self.mp_with_att.attention(self.x_source, self.x_target)
        assert result.shape == (n_messages,)

    def test_aggregate(self):
        """Test aggregate."""
        x_message = torch.tensor([[1, 2], [3, 4], [5, 6], [3, 4], [5, 6], [5, 6]])
        self.mp_with_att.target_index_i = torch.tensor([0, 0, 1, 1, 2, 2])

        result = self.mp_with_att.aggregate(x_message)
        expected = torch.tensor([[4, 6], [8, 10], [10, 12]])
        assert torch.allclose(result, expected)

    def test_forward(self):
        """Test forward."""
        # Test without attention
        result = self.mp.forward(self.x_source, self.neighborhood)
        assert result.shape == (3, 2)

        # Test with attention (source & target on the same cells)
        result = self.mp_with_att.forward(self.x_source, self.neighborhood)
        assert result.shape == (3, 2)

        # Test with attention (source & target on different cells)
        result = self.mp_with_att.forward(
            self.x_source, self.neighborhood_r_to_s, self.x_target
        )
        assert result.shape == (2, 2)
>>>>>>> 03d82bca4fb5c1a611c2a0d97abb7c0ff6cb594d
