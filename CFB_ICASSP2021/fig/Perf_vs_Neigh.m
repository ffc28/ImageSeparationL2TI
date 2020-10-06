clear;clc;close all;
% This script plots the performance of LCIVA as a function of the WGL neigh
% size along with the oracle permutation

neigh = [150 200 300 400 513]/513*100;

SDR = [1.54 2.03 2.68 3.65 2.76];
SIR = [4.16 4.45 3.92 6.55 4.73];
ISR = [5.65 5.90 6.16 7.00 6.36];
SAR = [13.36 13.12 13.39 12.35 13.36];

SDRo = [3.07 3.89 3.88 4.33 3.77];
SIRo = [5.71 6.26 6.79 7.45 6.21];
ISRo = [6.71 7.29 7.93 7.80 7.25];
SARo = [13.77 13.25 14.01 12.43 13.59];

%%
figure;
subplot(141);
set(gca,'fontsize',10)
hold on;
plot(neigh,SDR,'-*b',neigh,SDRo,'--r');
xlabel('Neighborhood size (%)');
ylabel('SDR (dB)');
grid on;

subplot(142);
set(gca,'fontsize',10)
hold on;
plot(neigh,SIR,'-*b',neigh,SIRo,'--r');
xlabel('Neighborhood size (%)');
ylabel('SIR (dB)');
grid on;

subplot(143);
set(gca,'fontsize',10)
hold on;
plot(neigh,ISR,'-*b',neigh,ISRo,'--r');
xlabel('Neighborhood size (%)');
ylabel('ISR (dB)');
grid on;

subplot(144);
set(gca,'fontsize',10)
hold on;
plot(neigh,SAR,'-*b',neigh,SARo,'--r');
xlabel('Neighborhood size (%)');
ylabel('SAR (dB)');
grid on;
ylim([12, 15]);

legend('LCIVA','LCIVA + oracle permutation');