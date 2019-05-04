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

% -------------------------------------------------------------

%% EFFECT OF SNR ON BEAM PATTERN
% role : important
% status : complete

%% basic setup
clc;
close all;

%% initialising the variables
f                  = 2000;                               %the main frequency
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
        new_mat = zeros(N,m);                            %initialising the noise matrix
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

for SNR_variable = 1:num_decibels
    f_mat(SNR_variable,m+1) = 1;    % the final value indicates whether it is yes/no
end

save('/Users/vrsreeganesh/Desktop/Sonar_repo/deep_input.mat','f_mat');

cout = 'done'
size____ = size(f_mat)







%----------------------------- TRASH BIN ----------------------------------

% %% bringing the delay in frequency region
% angle_matrix= zeros(1,181);
% delay_column = zeros(m,1);
% 
% for test_angle = 0:180
% test_d = x*cosd(test_angle)/c;             %the unit delay for test angle
% 
% for i = 1:m
% delay_column(i,1) = exp(1*1i*2*pi*f*(i-1)*test_d);
% end
% 
% angle_matrix(1,test_angle+1) = abs(f_mat*delay_column);
% end
% 
% %% plotting the beam formed output
% angle_axis = linspace(0,180,181);            %setting up the axis values for displaying angle_matrix
% 
% hold on
% plot(angle_axis,10*log10(angle_matrix),'linewidth',2);  %plotting the angle matrix
% xlabel('angle','FontSize',32)
% ylabel('absolute value (in dB)','FontSize',32)


