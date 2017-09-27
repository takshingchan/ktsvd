function D = ktsvd(Y,D0,lambda,iter)
%KTSVD Multidimensional dictionary learning using K-TSVD.
%   D = KTSVD(Y,D0,lambda,iter) returns a trained tensor dictionary D.
%
%   Inputs:
%     Y         - Observed tensor data (n1-by-n2-by-n3).
%     D0        - Initial dictionary (n1-by-K-by-n3).
%     lambda    - Sparsity parameter.
%     iter      - Number of iterations.
%
%   Output:
%     D         - Trained tensor dictionary (n1-by-K-by-n3).
%
%   See also TSC.

%   References:
%     [1] Z. Zhang and S. Aeron, "Denoising and completion of 3D data via
%         multidimensional dictionary learning," in Proc. Int. Joint Conf.
%         Artificial Intelligence, 2016, pp. 2371-2377.

%   Copyright 2017 Tak-Shing Chan

if size(Y,1)~=size(D0,1) || size(Y,3)~=size(D0,3)
    error('Tensor dimensions must agree.')
end

n3 = floor(size(Y,3)/2+1);          % Real-input DFTs are symmetric.
K = size(D0,2);

% Algorithm 2 K-TSVD.
Y = fft(Y,[],3);
D = fft(D0,[],3);
Y = Y(:,:,1:n3);
D = D(:,:,1:n3);
for J = 1:iter
    X = tsc(Y,D,lambda);            % Tensor sparse coding.
    for k = 1:K
        wk = find(any(X(k,:,:),3)); % Let wk = {i|X(k,i,:) ~= 0}.
        D(:,k,:) = 0;               % Compute the overall error without
                                    % using D(:,k,:).
        for i = 1:n3
            % Choosing the tensor columns corresponding to wk to obtain Rk.
            Rk = Y(:,wk,i)-D(:,:,i)*X(:,wk,i);
            [U,S,V] = svds(Rk,1);   % Compute the approximate rank-1 SVD.
            D(:,k,i) = U;
            X(k,wk,i) = S*V';
        end
    end
end
D = cat(3,D,conj(D(:,:,ceil(size(D0,3)/2):-1:2)));
D = ifft(D,[],3);
