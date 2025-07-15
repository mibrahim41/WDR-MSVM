figure(1)
% sgtitle('4 Classes')
subplot(2,4,1)
load imbalance_experiment_4classes_3features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('3 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 3 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 3 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,2)
load imbalance_experiment_4classes_5features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('5 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 5 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 5 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,3)
load imbalance_experiment_4classes_15features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('15 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 15 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 15 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,4)
load imbalance_experiment_4classes_30features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('30 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 30 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 30 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,5)
load imbalance_experiment_4classes_3features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('3 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 3 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 3 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,6)
load imbalance_experiment_4classes_5features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('5 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 5 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 5 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,7)
load imbalance_experiment_4classes_15features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('15 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 15 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 15 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,8)
load imbalance_experiment_4classes_30features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('30 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('4 Classes, 30 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('4 Classes, 30 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

figure(2)
% sgtitle('8 Classes')
subplot(2,4,1)
load imbalance_experiment_8classes_3features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('3 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 3 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 3 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,2)
load imbalance_experiment_8classes_5features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('5 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 5 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 5 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,3)
load imbalance_experiment_8classes_15features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('15 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 15 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 15 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,4)
load imbalance_experiment_8classes_30features_balanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('30 Features, Balanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 30 Features, Balanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 30 Features, Balanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,5)
load imbalance_experiment_8classes_3features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('3 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 3 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 3 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,6)
load imbalance_experiment_8classes_5features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('5 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 5 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 5 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,7)
load imbalance_experiment_8classes_15features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('15 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 15 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 15 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

subplot(2,4,8)
load imbalance_experiment_8classes_30features_imbalanced.mat
[vec_plot_multi,max_val_multi,kappa_val_multi] = analyze_tensor( ...
    acc_mat_multi,kappa_vec);
[vec_plot_ova,max_val_ova,kappa_val_ova] = analyze_tensor( ...
    acc_mat_ova,kappa_vec);
semilogx(epsilon_vec,vec_plot_multi,'r',LineWidth=1);
hold on
semilogx(epsilon_vec,vec_plot_ova,'k--',LineWidth=1);
title('30 Features, Imbalanced',Interpreter='latex')
xlabel('$\varepsilon$',Interpreter='latex')
ylabel('mCCR',Interpreter='latex')
leg1 = sprintf('MSVM, $\\kappa$=%.2f',kappa_val_multi);
leg2 = sprintf('OVA-SVM, $\\kappa$=%.2f',kappa_val_ova);
legend({leg1,leg2},Location='southoutside',Interpreter='latex')
axis([1e-4 1e0 0 1])
%grid on
fprintf('8 Classes, 30 Features, Imbalanced, MSVM: %d\n', ...
    round(max_val_multi*100,2));
fprintf('8 Classes, 30 Features, Imbalanced, OVA-SVM: %d\n', ...
    round(max_val_ova*100,2));
disp('\n')

function [vec_plot,max_val,kappa_val] = analyze_tensor(arr_in,kappa_vec)
max_vec = max(arr_in,[],1);
[max_val,idx] = max(max_vec);
vec_plot = arr_in(:,idx);
kappa_val = kappa_vec(idx);
end