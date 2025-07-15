load C_data.mat
load P_data.mat
load N_data.mat
C_vec = [4,6,8,10,12];
P_vec = [4,10,16,22,28];
N_vec = [1000,2000,3000,4000,5000];

subplot(2,3,1)
errorbar(C_vec,runtime_ratio_C,std_dev_ratio_C,'xk--','LineWidth',0.7)
axis([3,13,0,0.2])
xlabel('Number of Classes','Interpreter','latex')
ylabel('Runtime Ratio [Barrier:Subgrad]','Interpreter','latex')
grid on

subplot(2,3,2)
errorbar(P_vec,runtime_ratio_P,std_dev_ratio_P,'xk--','LineWidth',0.7)
axis([2,30,0,0.2])
xlabel('Number of Features','Interpreter','latex')
ylabel('Runtime Ratio [Barrier:Subgrad]','Interpreter','latex')
grid on

subplot(2,3,3)
errorbar(N_vec,runtime_ratio_N,std_dev_ratio_N,'xk--','LineWidth',0.7)
axis([900,5100,0,0.2])
xlabel('Number of Training Samples','Interpreter','latex')
ylabel('Runtime Ratio [Barrier:Subgrad]','Interpreter','latex')
grid on

subplot(2,3,4)
errorbar(C_vec,acc_multi_final_C,acc_std_multi_final_C,'or--','LineWidth',0.7)
hold on
errorbar(C_vec,acc_subgrad_final_C,acc_std_subgrad_final_C,'xk--','LineWidth',0.7)
axis([3,13,0.5,1.0])
xlabel('Number of Classes','Interpreter','latex')
ylabel('mCCR','Interpreter','latex')
legend({'Barrier','Subgrad'},'Interpreter','latex','Location','southeast')
grid on

subplot(2,3,5)
errorbar(P_vec,acc_multi_final_P,acc_std_multi_final_P,'or--','LineWidth',0.7)
hold on
errorbar(P_vec,acc_subgrad_final_P,acc_std_subgrad_final_P,'xk--','LineWidth',0.7)
axis([2,30,0.5,1.0])
xlabel('Number of Features','Interpreter','latex')
ylabel('mCCR','Interpreter','latex')
legend({'Barrier','Subgrad'},'Interpreter','latex','Location','southeast')
grid on

subplot(2,3,6)
errorbar(N_vec,acc_multi_final_N,acc_std_multi_final_N,'or--','LineWidth',0.7)
hold on
errorbar(N_vec,acc_subgrad_final_N,acc_std_subgrad_final_N,'xk--','LineWidth',0.7)
axis([900,5100,0.5,1.0])
xlabel('Number of Training Samples','Interpreter','latex')
ylabel('mCCR','Interpreter','latex')
legend({'Barrier','Subgrad'},'Interpreter','latex','Location','southeast')
grid on