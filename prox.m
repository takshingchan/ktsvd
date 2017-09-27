function X = prox(Z,lambda)
%% prox.m
% Proximity operator for the l1 norm (frequency domain measurements)
%   solve arg min_X 1/2|Z-X|_F^2+lambda|X|_1
% ----------------------------------
% Tak-Shing Chan 7-Apr-2016
% takshingchan@gmail.com
% Copyright: Music and Audio Computing Lab, Academia Sinica, Taiwan
%%

X = bsxfun(@times,max(1-lambda./tabs(Z),0),Z);
