%% SIMULATE BROADBAND BEAMFORMING
% role: important
% status : complete

%% basic setup
clc;
close all;

%% initialising variables
angle   = 120;                                          %input wave angle
f            = 2000;                                       %input wave frequency
Fs         = 12800;                                     %sampling frequency
Ts         = 1/Fs;                                        %sampling interval
c           = 1500;                                       %speed of sound in water
m          = 32;                                           %number of element

SNR      = -2;                                           %signal to noise ratio


N           = 256;                                         %no of original samples
t             = (0:N-1)*Ts;                             %total time of input

lambda  = c/2000;                                     %wavelength
x            = lambda/2;                                 %interspace distance
d            = x*cosd(angle)/c;                      %quantum delay
matrix    = zeros(N,m);                             %delayed pure signals

%% bringing the natural delay
y       = sin(2*pi*f*t);                                %generating the ideal sine wave
tend  = (N-1)*Ts;                                       %finding the time interspace
xaxis = linspace(0,tend,N);                       %constructing the time axis

for i = 1:m                                                 %bringing about the delay
matrix(:,i)=sin(2*pi*f*(t-(i-1)*d));
end

%% adding the noise
SNR_weight = 10^(-1*SNR*0.05);                       %computing the noise weight
new_mat    = matrix + SNR_weight*rand(N,m);  %creating the impure matrix

%% taking the fourier transform

Fourier  = zeros(N,m);                              %initialising the fourier
NFFT     = N;                                            %defining NFFT
fend       = (NFFT-1)*Fs/NFFT;                %end sample index
waxis     = linspace(0,fend,NFFT);           %creating the spacing

Fourier = fft(new_mat,NFFT);                 %taking the fourier transform

figure(3)
plot(waxis,abs(Fourier));                          %plotting the fourier transform

%% establishing the delay thing
delay_column    = zeros(m,1);                  %initialising the delay column
frequency_inter = Fs/N;                            %frequency index
f_mat           = zeros(1,m);                        %initialising the bin row/matrix
angle_matrix    = zeros(N,181);                %initialising the angle matrix
frequency_matrix = zeros(180,9);

for sweep_angle = 1:180
for f = 1000:250:3000
index = 1 + f/(Fs/N);
f_mat(1,:) = Fourier(index,:);

for i = 1:m
delay_column(i,1) = exp(1*1i*(i-1)*2*pi*f*(x/c)*(cosd(sweep_angle)));
end

frequency_matrix(sweep_angle,(f/250)-3) = abs(f_mat*delay_column);

end
end
%% plotting the frequency vs abs
angle_axis = linspace(1,180,180);             %setting up the angle axis
sum_matrix = sum(frequency_matrix,2);  %summing column wise

plot(angle_axis,20*log10(sum_matrix),'linewidth',2);          %plotting the sum_matrix
xlabel('angle','FontSize',32)
ylabel('absolute value','FontSize',32)

