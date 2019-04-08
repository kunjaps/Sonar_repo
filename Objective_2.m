%% Simulate the input received by a 4 element array
% role : important
status : complete

%% basic setup
clc;
close all;

%% Initialising variables
f            = 200; %the main frequency
Fs         = 12800; %sampling frequency
Ts          = 1/Fs; %sampling interval
N           = 128;    %number of intervals

m            = 4;     %number of sensors
angle      = 60;      %incoming angle
c             = 1500;                                     %speed of the sound signal
lambda   = c/f;                                         %wavelength of incoming signal
x             = lambda/2;                              %sensor interspacing
d             = x*cosd(angle)/c;                   %unit delay

SNR        = 10;                                       %signal to noise ratio
SNR_weight = 10^(-1*SNR*0.05);         %SNR noise weight

t          = (0:N-1)*Ts;                               %time matrix
matrix     = zeros(N,m);                          %initialising noise included signal

%% bringing about the natural delay
y = sin(2*pi*f*t);                                      %generating the ideal sine wave

for i = 1:m
matrix(:,i) = sin(2*pi*f*(t-(i-1)*d));
end

%% adding the noise
new_mat = zeros(N,m);                          %initialising the noise matrix
new_mat = matrix + SNR_weight*rand(N,m);   %creating the impure matrix

%% plotting the wave in the time domain
time_axis = linspace(0,(N-1)*Ts,N);        %creating the time axis

figure(1)
plot(time_axis,new_mat,'linewidth',2);    %plotting the signal in time domain
xlabel('Time (seconds)','FontSize',32)
ylabel('Amplitude','FontSize',32)


