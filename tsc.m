function X = tsc(Y,D,lambda,rho,tau,tol,maxiter)
%TSC Tensor sparse coding in the Fourier domain.
%   X = TSC(Y,D,lambda,rho,tau,tol,maxiter) solves
%                 min_X |Y-DX|_F^2+lambda|X|_1,1,2
%   using the inexact augmented Lagrangian method (IALM).
%
%   Inputs:
%     Y         - Observed tensor data (n1-by-n2-by-n3).
%     D         - Dictionary (n1-by-K-by-n3).
%     lambda    - Sparsity parameter.
%     rho       - Initial penalty parameter (default 1e-3).
%     tau       - Penalty update parameter (default 1.2).
%     tol       - Tolerance (default 1e-6).
%     maxiter   - Maximum number of iterations (default 1000).
%
%   Output:
%     X         - Tensor sparse code (K-by-n2-by-n3).
%
%   See also KTSVD.

%   References:
%     [1] Z. Zhang and S. Aeron, "Denoising and completion of 3D data via
%         multidimensional dictionary learning," in Proc. Int. Joint Conf.
%         Artificial Intelligence, 2016, pp. 2371-2377.

%   Copyright 2017 Tak-Shing Chan

if size(Y,1)~=size(D,1) || size(Y,3)~=size(D,3)
    error('Tensor dimensions must agree.')
end
if nargin<4
    rho = 1e-3;
end
if nargin<5
    tau = 1.2;
end
if nargin<6
    tol = 1e-6;
end
if nargin<7
    maxiter = 1000;
end

K = size(D,2);
[~,n2,n3] = size(Y);

% Initialization.
X = zeros(K,n2,n3);
DD = zeros(K,K,n3);
DY = zeros(K,n2,n3);
for i = 1:n3
    % Precompute 2D'D and 2D'Y to speed things up.
    DD(:,:,i) = 2*D(:,:,i)'*D(:,:,i);
    DY(:,:,i) = 2*D(:,:,i)'*Y(:,:,i);
end
Z = zeros(K,n2,n3);
Q = zeros(K,n2,n3);

% Compute the sparse coefficient tensor using (15)-(17).
for iter = 1:maxiter
    %% Update X.
    for i = 1:n3
        X(:,:,i) = (DD(:,:,i)+rho*eye(K))\(DY(:,:,i)+rho*Z(:,:,i)-Q(:,:,i));
    end

    %% Update Z.
    Z = prox(X+Q/rho,lambda/rho);

    %% Update Q.
    R = X-Z;
    Q = Q+rho*R;
    rho = tau*rho;

    %% Check for convergence.
    if norm(R(:))<tol
        return
    end
end
disp('Maximum iterations exceeded.')
