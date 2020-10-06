clear;clc;close all;
% This script plots the performance of LCIVA as a function of the sparsity
% level of the estimated sources

sp = [0.46	6.77	27.33	46.07	61.28	78.37	88.68];

SDR = [3.65	3.75		2.67	2.84	2.45	1.75	1.21];
SIR = [6.55	5.69		5.81	6.25	5.84	5.17	4.84];
ISR = [7.00	6.75		5.30	4.58	4.02	2.64	1.74];
SAR = [12.35	13.06		8.72	6.95	5.47	3.71	2.34];

%%
figure;
subplot(141);
set(gca,'fontsize',10)
hold on;
plot(sp,SDR);
xlabel('Sparsity level (%)');
ylabel('SDR (dB)');
grid on;

subplot(142);
set(gca,'fontsize',10)
hold on;
plot(sp,SIR);
xlabel('Sparsity level (%)');
ylabel('SIR (dB)');
grid on;

subplot(143);
set(gca,'fontsize',10)
hold on;
plot(sp,ISR);
xlabel('Sparsity level (%)');
ylabel('ISR (dB)');
grid on;

subplot(144);
set(gca,'fontsize',10)
hold on;
plot(sp,SAR);
xlabel('Sparsity level (%)');
ylabel('SAR (dB)');
grid on;
