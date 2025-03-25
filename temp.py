import torch 

H, W = 4, 4 
i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # 要按横轴是u 纵轴是v的顺序 
i = i.t() 
j = j.t() 

print(i) 
print(j) 

u = torch.linspace(0, W-1, W).repeat(H, 1) 
print(u)
v = torch.linspace(0, H-1, H).reshape(H, 1).repeat(1, W) 
print(v) # 默认的张量是行向量 

x = torch.tensor(
    [
        [1, 2], 
        [3, 4]
    ]
) 
y = torch.tensor(
    [
        [5, 6],
        [7, 8]
    ]
)

print(x) 
print(y) 
print(x * y)

z = torch.randn((400, 400, 3)) 
q = torch.randn((4, 4)) 

t = z[..., None, :] * q[:3, :3] 
print(t.shape) 

w = torch.tensor(
    [
        [1, 2, 3], 
        [4, 5, 6],
        [7, 8, 9]
    ]
) / 10.0 
p = torch.tensor([1, 2, 3]) 
print(w * p) 
print((w * p).sum(dim=-1))


H, W = 4, 4 
i, j = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)) 
print(i) 
print(j) 

ij = torch.stack([i, j], -1) 
# print(ij, ij.shape) 

ij = ij.reshape(-1, 2) 

print(ij)

x = torch.tensor([1, 2, 3, 4, 5]).view(-1, 1) 
y = torch.tensor([1, 2, 3, 4]) 

print(x) 
print(y) 

print(x * y) 