%% Simulate beam pattern by shifting theta
% role : important
% status : complete

%% basic setup
clc;
close all;

%% initialising the variables
f               = 2000;                                     %the main frequency

m             = 32;                                         %number of sensors
angle       = 135;                                       %incoming angle
c              = 1500;                                     %speed of the sound signal
lambda    = c/f;                                          %wavelength of incoming signal
x              = lambda/2;                              %sensor interspacing
d              = x*cosd(angle)/c;                    %unit delay

matrix      = zeros(1,m);                           %initialising signal

%% bringing about the natural delay

for i = 1:m
matrix(1,i) = (1/m)*exp(-1*1i*2*pi*f*(i-1)*d);
end

%% bringing the delay in frequency region
delay_column = zeros(m,1);
angle_matrix= zeros(1,181);


for test_angle = 1:180
test_d = x*cosd(test_angle)/c;             %the unit delay for test angle

for i = 1:m
delay_column(i,1) = exp(1*1i*2*pi*f*(i-1)*test_d);
end

angle_matrix(1,test_angle) = abs(matrix*delay_column);
end

%% plotting the response
angle_axis = linspace(0,180,181);           %setting up the axis values for displaying angle_matrix

figure(2)
plot(angle_axis,20*log10(angle_matrix),'linewidth',3); %plotting the angle matrix
xlabel('ANGLE','FontSize',32)
ylabel('Gain in dB','FontSize',32)



