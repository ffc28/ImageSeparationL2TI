clear;clc;close all;
% This script plots the performance of LCIVA as a function of the kernel
% length 

% RT = 130 ms

K = [2 3 4 5 6]*0.0174*1000; % ms

SDR = [2.54 4.21 6.27 7.07 6.35];
SIR = [4.76 7.11 9.50 10.71 9.28];
ISR = [6.32 8.05 9.98 11.11 10.15];
SAR = [18.05 18.52 17.41 16.84 14.82];

% RT = 250 ms
K2 = [3 4 5 6 7]*0.0174*1000; % ms

SDR2 = [1.57 2.19 3.87 3.65 2.87];
SIR2 = [2.87 4.09 6.23 6.55 4.91];
ISR2 = [5.07 6.03 7.61 7.00 6.29];
SAR2 = [15.80 15.07 14.65 12.35 11.87];

%%
figure;
subplot(141);
set(gca,'fontsize',10)
hold on;
plot(K,SDR,'-*b',K2,SDR2,'-sr');
xlabel('Kernel length (ms)');
ylabel('SDR (dB)');
grid on;

subplot(142);
set(gca,'fontsize',10)
hold on;
plot(K,SIR,'-*b',K2,SIR2,'-sr');
xlabel('Kernel length (ms)');
ylabel('SIR (dB)');
grid on;

subplot(143);
set(gca,'fontsize',10)
hold on;
plot(K,ISR,'-*b',K2,ISR2,'-sr');
xlabel('Kernel length (ms)');
ylabel('ISR (dB)');
grid on;

subplot(144);
set(gca,'fontsize',10)
hold on;
plot(K,SAR,'-*b',K2,SAR2,'-sr');
xlabel('Kernel length (ms)');
ylabel('SAR (dB)');
grid on;

legend('RT_{60} = 130 ms','RT_{60} = 250 ms');