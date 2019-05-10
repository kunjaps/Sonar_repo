%% Creating the matrix for supervised learning
% AIM : creating a matrix with rows as the values the antennas receive and
% the last row containing the angle from which it is coming
% status : working on it

%% REPORT
% the matrix should be of size (number of antenna +1)*(number of training sets)
% each of those columns corresponds to the values received at different
% noise levels

% the intention is to increase the noise levels as column increases and see
% if the deepnet can still predict the direction in places where the
% vanilla network fails. Oru simple yes/no aanu. 

% 1.create the matrix (for a frequency, for one angle,different noise levels)
% 2.create the matrix (for a frequency, at different angles, same noise level)
% 3.create the matrix (for a frequency, at different angles, at differnt noise level)
% None of the above are beam steered

%% last seen
% 4th May : searching for reusable code
% 10th May : 

% -------------------------------------------------------------


%% basic setup
clc;
close all;

%% initialising the variables
f                  = 4500;                               %the main frequency
Fs                 = 12800;                             %sampling frequency
Ts                 = 1/Fs;                               %sampling interval
N                  = 128;                                  %number of intervals/s

m                  = 32;                                   %number of sensors
angle              = 90;                                   %incoming angle
c                  = 1500;                                %speed of the sound signal
lambda             = c/f;                                   %wavelength of incoming signal
x                  = lambda/2;                         %sensor interspacing
d                    = x*cosd(angle)/c;              %unit delay

t                    = (0:N-1)*Ts;                       %time matrix
matrix           = zeros(N,m);                      %initialising noise included signal

num_decibels = 1000;
f_mat=zeros(num_decibels,m);

for SNR_variable = 1:num_decibels
        SNR              = -SNR_variable/10;                                   %signal to noise ratio
        SNR_weight = 10^(-1*SNR*0.05);          %SNR noise weight
        %% bringing about the natural delay
        y = sin(2*pi*f*t);                                       %generating the ideal sine wave

        for i = 1:m
        matrix(:,i) = sin(2*pi*f*(t-(i-1)*d));
        end

        %% adding the noise
        %new_mat = zeros(N,m);                            %initialising the noise matrix
        new_mat = matrix + SNR_weight*rand(N,m);%creating the impure matrix

        %% taking the fourier transform
        NFFT = N;                                   %number of frequency samples
        fend = (NFFT-1)*Fs/NFFT;                    %finding the end frequency
        w_axis = linspace(0,fend,NFFT);             %creating the spacing

        Fourier = fft(new_mat,NFFT);                  %taking the fourier transform

        %% Choosing the frequency row
        index = f/(Fs/NFFT)+1;
        f_mat(SNR_variable,:)=Fourier(index,:);               % INPUT TO THE DEEP NET
                
end

f_mat = transpose(f_mat);
label_matrix = zeros(1,num_decibels);

for SNR_variable = 1:num_decibels
    label_matrix(1,SNR_variable) = 0;    % the final value indicates whether it is yes/no
end

save('/Users/vrsreeganesh/Desktop/Sonar_repo/training_false.mat','f_mat');

cout = 'done'
size_ = size(f_mat)
size_label = size(label_matrix)

