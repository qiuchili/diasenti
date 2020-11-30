# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexTensorProduct(torch.nn.Module):
    def __init__(self):
        super(ComplexTensorProduct, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
    def kronecker(self, A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

        real_inputs = inputs[0]
        imag_inputs = inputs[1]
       
        
        return [real_part,imag_part]

def test():
    product = ComplexProduct()
    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    c = torch.randn(4, 10)
    d = torch.randn(4, 10)
    product = product([[a,b],[c,d]])
    if product[0].size(1) == a.size(1):
        print('ComplexProduct Test Passed.')
    else:
        print('ComplexProduct Test Failed.')

if __name__ == '__main__':
    test()# -*- coding: utf-8 -*-

