# Output of `demo.py` 

If all exercises have been solved correctly, you should see the following output when running `python demo.py`:

```
##### Exercise 1 #####

Tensor: 
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
Number of dimensions: 2
Data type: torch.int64
Shape: torch.Size([4, 4])
Device: cpu


##### Exercise 2 #####

Tensor: 
tensor([[17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32]])
Swapped Quadrants:
tensor([[27, 28, 19, 20],
        [31, 32, 23, 24],
        [25, 26, 17, 18],
        [29, 30, 21, 22]])


##### Exercise 3 #####

Tensor: 
tensor([  7.5000,   2.0000,  -4.0000, -10.0000,   5.0000])
minval = -5.0, maxval = 5.0
Absolute values of tensor normalized to [-1, 1]:
tensor([1.5000, 0.4000, 0.8000, 2.0000, 1.0000])


##### Exercise 4 #####

Tensor: 
tensor([[ 1.2000,  0.7000, -1.5000, -0.5000],
        [ 0.3400,  1.4000, -2.0000,  1.0000]])
Replaced values from [-1, 1] with 0:
tensor([[ 1.2000,  0.0000, -1.5000,  0.0000],
        [ 0.0000,  1.4000, -2.0000,  0.0000]])


##### Exercise 5 #####

Tensor: 
tensor([[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]],

        [[ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15]],

        [[16, 17],
         [18, 19]]])
Indices: [1, 3]

Selected matrices:
tensor([[[ 4,  5],
         [ 6,  7]],

        [[12, 13],
         [14, 15]]])


##### Exercise 6 #####

Numpy array: [  1.5  42.  275. ]
Device: cuda:0
Tensor from numpy array: tensor([  1.5000,  42.0000, 275.0000])
Zero-tensor with the same size: tensor([0., 0., 0.])
4x4 tensor containing integer 42:
tensor([[42, 42, 42, 42],
        [42, 42, 42, 42],
        [42, 42, 42, 42],
        [42, 42, 42, 42]])
Data type of 4x4 tensor: torch.int64
8-bit unsigned int tensor: tensor([ 1, 42, 19], dtype=torch.uint8)
Data type 8-bit unsigned int tensor: torch.uint8
Tensor moved to device: tensor([ 1, 42, 19], device='cuda:0', dtype=torch.uint8)
Device of tensor moved to device: cuda:0
Data type of tensor moved to device: torch.uint8


##### Exercise 7 #####

Vectors:
x: tensor([-2.,  5.])
y: tensor([0., 1.])
z: tensor([ 3.1400, 42.0000])
Matrix with vectors as columns:
tensor([[-2.0000,  0.0000,  3.1400],
        [ 5.0000,  1.0000, 42.0000]])


##### Exercise 8 #####

Tensor: 
tensor([[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]])
Maximum of the sum of each column: tensor([45])
Shape: torch.Size([1])


##### Exercise 9 #####

Vectors:
x: tensor([9, 7, 5])
y: tensor([8, 6, 4])
Interleaved vector: tensor([9, 8, 7, 6, 5, 4])


##### Exercise 10 #####

Tensor:
tensor([[10, 12, 14],
        [16, 18, 20]])
Shape: torch.Size([2, 3])
Transposed tensor:
tensor([[10, 16],
        [12, 18],
        [14, 20]])
Shape: torch.Size([3, 2])

Tensors:
tensor([[[21, 22, 23],
         [24, 25, 26]],

        [[27, 28, 29],
         [30, 31, 32]]])
Shape: torch.Size([2, 2, 3])
Transposed tensors:
tensor([[[21, 24],
         [22, 25],
         [23, 26]],

        [[27, 30],
         [28, 31],
         [29, 32]]])
Shape: torch.Size([2, 3, 2])


##### Exercise 11 #####

### Part (a) ###

Tensors:
tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]]])
tensor([[0, 1, 2],
        [3, 4, 5]])
Elementwise sum of broadcasted tensors:
tensor([[[ 0,  1],
         [ 3,  4],
         [ 6,  7]],

        [[ 9, 10],
         [12, 13],
         [15, 16]]])

Tensor shapes:
torch.Size([2, 3, 4, 5, 1])
torch.Size([2, 4, 1])
Elementwise sum of broadcasted tensors (selected entries):
tensor(0) tensor(35) tensor(126)

Tensor shapes:
torch.Size([3, 1, 2])
torch.Size([3, 2, 7])
Elementwise sum of broadcasted tensors (selected entries):
tensor(0) tensor(28) tensor(46)

### Part (b) ###

Example scalar product:
Vector 1:       tensor([0.0420, 1.0420, 2.0420])
Vector 2:       tensor([0.5420, 1.5420, 2.5420])
Scalar product: tensor(6.8203)

Example scalar product:
Vector 1:       tensor([0.2560, 1.2560, 2.2560])
Vector 2:       tensor([0.7560, 1.7560, 2.7560])
Scalar product: tensor(8.6166)

Example scalar product:
Vector 1:       tensor([0.7500, 1.7500, 2.7500])
Vector 2:       tensor([0.2500, 1.2500, 2.2500])
Scalar product: tensor(8.5625)
```
