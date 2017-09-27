function Y = tabs(X)
%% tabs.m
% Tubal absolute value (frequency domain measurements)
% ----------------------------------
% Tak-Shing Chan 7-Apr-2016
% takshingchan@gmail.com
% Copyright: Music and Audio Computing Lab, Academia Sinica, Taiwan
%%

% fft is unscaled, so we use root-mean-square to recover the l2-norm
Y = sqrt(mean(real(X).^2+imag(X).^2,3));
