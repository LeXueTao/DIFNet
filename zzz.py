import torch

if __name__ == '__main__':
    x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                      [ 0.1815, -1.0111,  0.9805, -1.5923],
                      [ 0.1062,  1.4581,  0.7759, -1.2344],
                      [-0.1830, -0.0313,  1.1908, -1.4757]])
    b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    print(a // b)






