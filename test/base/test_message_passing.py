"""Test the message passing module."""

import torch
import numpy as np
import unittest


from topomodelx.base.message_passing import MessagePassing

import toponetx as tnx


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