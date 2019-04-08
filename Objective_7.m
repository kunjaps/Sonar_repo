%% TO GET THE ORIGINAL SIGNAL FROM THE BEAM FORMED OUTPUT
% role : important.
% status : complete.

%% basic setup
clc;
clear all;
close all;

%% initialising variables
f1          = 1000;                                  %the lower limit
f2          = 2000;                                  %the upper limit
angle    = 60;                                      %angle ranges from 0 to 180
f            = 200;                                    %frequency of the analog wave
Fs         = 800;                                    %sampling frequency
Ts         = 1/Fs;                                   %sampling time
N          = 256;                                    %number of samples
c           = 1500;                                  %speed of sound in water
m          = 32;                                      %number of elements
SNR     = 55;                                      %signal to noise ratio

lambda = c/f;                                      %wavelength
x           = lambda/2;                           %interspace distance
d           = x*cosd(angle)/c;                 %quantum delay

t            = (0:N-1)*Ts ;                        %creating the time
matrix   = zeros(N,m);                        %delayed pure signals

L           = 64;                                     %segment length
N_L      = N/L;                                    %number of such segments

%% bringing about the natural delay
y = sin(2*pi*f*t);                                  %generating the ideal sine wave

for i = 1:m
matrix(:,i) = sin(2*pi*f*(t-(i-1)*d));
end

tend = (N-1)*Ts;                                  %the final value in linspace
xaxis=linspace(0,tend,N);

new_mat    = matrix;

%% Here is where the blocking happens

%initialising the blocks
segment_matrix_1 = zeros(L,m);
segment_matrix_2 = zeros(L,m);

%transferring the data value for the blocks
for i = 1:L
segment_matrix_1(i,:) = new_mat(i,:);
segment_matrix_2(i,:) = new_mat(L+i,:);

end

%% fourier transform

%initialising the fourier
Fourier_1 = zeros(L,m);
Fourier_2 = zeros(L,m);

NFFT  = L;
fend  = (NFFT-1)*Fs/NFFT;
waxis = linspace(0,fend,NFFT);         %creating the spacing

%setting up the fouriers
Fourier_1 = fft(segment_matrix_1,NFFT);
Fourier_2 = fft(segment_matrix_2,NFFT);

%% beamforming
%the angle matrix stores the magnitude of response for each angle

angle_matrix_1 = zeros(1,181);         %initialising the angle matrix
angle_matrix_2 = zeros(1,181);

f_mat_1 =  zeros(1,m);                       %initialising the matrix for the chosen frequency
f_mat_2 =  zeros(1,m);

index        = (f/(Fs/NFFT))+1;              %finding the index value of f
delay_column = zeros(m,1);               %initialising the delay column

f_mat_1(1,:) =  Fourier_1(index,:);      %extracting the values for the frequency
f_mat_2(1,:) =  Fourier_2(index,:);

for test_angle = 0:180
test_d = x*cosd(test_angle)/c;        %quantum delay for test angle

for i = 1:m                                       %setting up the delay column
delay_column(i,1,:) = exp(-1*1i*2*pi*f*(i-1)*test_d);           %steering vector
end

angle_matrix_1(1,test_angle+1,:) = f_mat_1*delay_column; %storing
angle_matrix_2(1,test_angle+1,:) = f_mat_2*delay_column;

end

%% Constructing the artificial fourier

%initialising the artificial fourier
art_fourier_1 = zeros(1,L);
art_fourier_2 = zeros(1,L);

%setting up the artificial fourier
art_fourier_1(1,index) = angle_matrix_1(1,angle+1);
art_fourier_2(1,index) = angle_matrix_2(1,angle+1);

art_fourier_1(1,index + NFFT/2) = conj(angle_matrix_1(1,angle+1));
art_fourier_2(1,index + NFFT/2) = conj(angle_matrix_2(1,angle+1));

%% taking the inverse fourier

inv_fourier_1 = ifft(art_fourier_1);
inv_fourier_2 = ifft(art_fourier_2);

%% plotting the inverse fourier

figure(4)
plot((inv_fourier_1),'linewidth',2)
xlabel('Time (seconds)','FontSize',32)
ylabel('Amplitude','FontSize',32)


figure(5)
plot((inv_fourier_2),'linewidth',2);
xlabel('Time (seconds)','FontSize',32)
ylabel('Amplitude','FontSize',32)

%% concatenating

trunc_sample_num  = round((m-1)*d*Fs);
remain_sample_num = L - trunc_sample_num;

concat_matrix = zeros(1,64);

for i = 1:remain_sample_num
concat_matrix(1,i)    = inv_fourier_1(1,i);
end

for i = 1:remain_sample_num
concat_matrix(1,i + remain_sample_num) = inv_fourier_2(1,i);
end

figure(6)
plot(concat_matrix,'linewidth',2);
xlabel('Time (seconds)','FontSize',32)
ylabel('Amplitude','FontSize',32)


