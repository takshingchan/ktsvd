Multidimensional Dictionary Learning
------------------------------------

This code reimplements Algorithm 2 of Z. Zhang and S. Aeron, "Denoising and
completion of 3D data via multidimensional dictionary learning," in *Proc.
Int. Joint Conf. Artificial Intelligence*, 2016, pp. 2371-2377.

I have two minor comments on the paper. First, equation (15) minimizes
tr(Q^TX), but Q and X are in the Fourier domain and are therefore not real.
Here we should minimize Re tr(Q^HX) instead (with Wirtinger calculus).
Second, Algorithm 2 computes approximate rank-1 SVDs but this may destroy
the conjugate symmetry in the third dimension. To address this, we can work
on the first half of the spectrum only.

The following functions are available:

| Function | Description                                       |
| -------- | ------------------------------------------------- |
| `ktsvd`  | Multidimensional dictionary learning using K-TSVD |
| `odct`   | Overcomplete DCT dictionary                       |
| `tsc`    | Tensor sparse coding in the Fourier domain        |

My overcomplete DCT dictionary is slightly different from Elad's or
Rubinstein's. It invokes the DCT-III formula to match
kron(idct(eye(sqrt(n))),idct(eye(sqrt(n)))) when m equals n.

Example usage:

```matlab
I = imread('autumn.tif');
Y        = im2col(I(:,:,1),[8 8]);  % Extract 8x8x3 patches.
Y(:,:,2) = im2col(I(:,:,2),[8 8]);
Y(:,:,3) = im2col(I(:,:,3),[8 8]);
Y = Y(:,1:10000,:);                 % Use only the first 10,000 patches.
D0 = odct(size(Y,1),256);           % Get overcomplete DCT dictionary and
D0 = repmat(D0/sqrt(3),[1 1 3]);    % propagate to other frontal slices.
D = ktsvd(Y,D0,0.1,10);
```

Implementation notes. The 1,1,2-norm \[1\] is isomorphic to the polar
n-bicomplex 1-norm \[2\], so I simply copy the polar n-bicomplex `prox.m`
and `tabs.m` from [tsp2016](https://github.com/takshingchan/tsp2016). I
use the IALM variant of ADMM here because it converges faster.

Tak-Shing Chan

24 September 2017

### References

\[1\] Z. Zhang and S. Aeron, "Denoising and completion of 3D data via
multidimensional dictionary learning," in *Proc. Int. Joint Conf.
Artificial Intelligence*, 2016, pp. 2371-2377.

\[2\] T.-S. T. Chan and Y.-H. Yang, "Polar *n*-complex and *n*-bicomplex
singular value decomposition and principal component pursuit," *IEEE Trans.
Signal Process.*, vol. 64, no. 24, pp. 6533-6544, 2016.
