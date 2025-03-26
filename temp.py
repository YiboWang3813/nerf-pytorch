import torch 

alpha = torch.randn((1024, 64)) 
weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

print(weights.shape) # (1024, 64)

x = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)
y = torch.cumprod(x, -1) 
print(y) 