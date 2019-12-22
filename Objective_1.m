%% Simulate a sine wave + noise in MATLAB and see the effect of SNR on both time domain and frequency domain
% role: important.
% status : complete

%% Basic setup
clc;
close all;

%% Initialising variables
f                   = 2000; %the main frequency
Fs                = 25600;  %sampling frequency
Ts                 = 1/Fs;  %sampling interval
N                  = 128;   %number of intervals

SNR             = 1000000;                                     %signal to noise ratio
SNR_weight = 10^(-1*SNR*0.05);          %SNR multiplying factor

t          = (0:N-1)*Ts;                                 %time matrix
new_mat    = zeros(1,N);                          %initialising noise included signal

%% creating the sine wave
y = sin(2*pi*f*t);                                       %signal vector

%% adding the noise
new_mat = y + SNR_weight*rand(1,N);   %noise included signal vector

%% plotting the wave in the time domain
time_axis = linspace(0,(N-1)*Ts,N);         %creating the time axis

figure(1)
plot(time_axis,new_mat,'linewidth',2);     %plotting the signal in time domain
xlabel('time (seconds)','FontSize',32)
ylabel('amplitude','FontSize',32)

%% taking the fourier transform
NFFT = N;                                               %number of frequency samples
fend = (NFFT-1)*Fs/NFFT;                      %finding the end frequency
w_axis = linspace(0,fend/2,NFFT/2);      %creating the spacing

Fourier = fft(new_mat,NFFT);                 %taking the fourier transform

%% plotting the frequency domain
w = 1:NFFT/2;
figure(2)
plot(w_axis,abs(Fourier(1,w)),'linewidth',2);              %plotting the fourier matrix
xlabel('frequency (Hz)','FontSize',32)
ylabel('absolute value','FontSize',32)


