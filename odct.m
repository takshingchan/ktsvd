function D = odct(m,n)
%ODCT Overcomplete DCT dictionary.
%  D = ODCT(m,n) returns an overcomplete 2-D DCT dictionary by Kronecker
%  squaring an overcomplete 1-D DCT dictionary.
%
%   Inputs:
%     m - Length of each atom (a square number).
%     n - Number of atoms in D (a square number).
%
%   Output:
%     D - Overcomplete 2-D DCT dictionary (m-by-n).

%   Copyright 2017 Tak-Shing Chan

m = ceil(sqrt(m));
n = ceil(sqrt(n));

% Overcomplete 1-D DCT dictionary.
[x,y] = ndgrid(0:m-1,0:n-1);
D = cos(pi*(x+0.5).*y/n);               % DCT-III.
D = bsxfun(@minus,D,mean(D));           % Remove mean.
D(:,1) = 1;                             % Set the DC.
D = bsxfun(@rdivide,D,sqrt(sum(D.^2))); % Normalize.

% Kronecker squaring.
D = kron(D,D);
