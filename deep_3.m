%% Creating the matrix for supervised learning
% AIM : The neural network won't take complex values as arguments. So, we
% need to flatten the whole damn matrix. A matrix ine save cheyyal aanu
% nokkunne. Flattening numpy cheytholum.
% status : working on it

%% REPORT
% the matrix should be of size (number of sample values of one sensor after the other)*(number of training sets)
% it is important to transfer your input range between 0 and 1. Hence I
% think it would be a good idea to add 1 to every value and divide by 2 to
% make sure of the range

% the intention is to increase the noise levels as column increases and see
% if the deepnet can still predict the direction in places where the
% vanilla network fails. Oru simple yes/no aanu. 

% 1.create the matrix (for a frequency, for one angle,different noise levels)
% 2.create the matrix (for a frequency, at different angles, same noise level)
% 3.create the matrix (for a frequency, at different angles, at differnt noise level)
% None of the above are beam steered

%% last seen
% 4th May : searching for reusable code
% 1st June : Matrix reshaping. Matrix normalising
%  

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

num_decibels = 10;
f_mat=zeros(num_decibels,m);


%%

flat_variable = 1;
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
        
        matrix_flat(:,flat_variable) = reshape(matrix',[],1); 
        flat_variable  = flat_variable +1;
end








%%%%%%%%%%%%%%%
f_mat = transpose(f_mat);
label_matrix = zeros(1,num_decibels);

for SNR_variable = 1:num_decibels
    label_matrix(1,SNR_variable) = 0;    % the final value indicates whether it is yes/no
end

save('/Users/vrsreeganesh/Desktop/Sonar_repo/training_false.mat','f_mat');
save('/Users/vrsreeganesh/Desktop/Sonar_repo/label_false.mat','label_matrix');


cout = 'done'
size_ = size(f_mat)
size_label = size(label_matrix)

