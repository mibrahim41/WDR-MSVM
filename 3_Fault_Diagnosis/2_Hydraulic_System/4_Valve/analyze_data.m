num_exp = 50;
Lacc_tensor_multi = zeros([3,3,num_exp]);
Lacc_tensor_multi_ker = zeros([3,3,num_exp]);

Lacc_tensor_ova = zeros([3,3,num_exp]);
Lacc_tensor_ova_ker = zeros([3,3,num_exp]);

Lacc_tensor_Rmulti = zeros([3,num_exp]);
Lacc_tensor_Rmulti_ker = zeros([3,num_exp]);

for i = 1:num_exp
    filename = sprintf('WDR_MSVM_valve_%d.mat',i);
    load(filename)
    Lacc_tensor_multi(:,:,i) = acc_tensor_multi(:,:,1);
    Lacc_tensor_multi_ker(:,:,i) = acc_tensor_multi_ker_rbf(:,:,1);
    
    Lacc_tensor_ova(:,:,i) = acc_tensor_ova(:,:,1);
    Lacc_tensor_ova_ker(:,:,i) = acc_tensor_ova_ker_rbf(:,:,1);
    
    Lacc_tensor_Rmulti(:,i) = acc_tensor_Rmulti2(:,1);
    Lacc_tensor_Rmulti_ker(:,i) = acc_tensor_Rmulti2_ker_rbf(:,1);
end

[acc_multi,std_multi] = comp_metrics(Lacc_tensor_multi);

[acc_multi_ker,std_multi_ker] = comp_metrics(Lacc_tensor_multi_ker);

[acc_ova,std_ova] = comp_metrics(Lacc_tensor_ova);

[acc_ova_ker,std_ova_ker] = comp_metrics(Lacc_tensor_ova_ker);

[acc_Rmulti,std_Rmulti] = comp_metrics2(Lacc_tensor_Rmulti);

[acc_Rmulti_ker,std_Rmulti_ker] = comp_metrics2(Lacc_tensor_Rmulti_ker);

% acc_multi = max(max(mean(Lacc_tensor_multi,3)));
% acc_multi_s = max(max(mean(Lacc_tensor_multi_s,3)));
% 
% acc_multi_ker = max(max(mean(Lacc_tensor_multi_ker,3)));
% acc_multi_ker_s = max(max(mean(Lacc_tensor_multi_ker_s,3)));
% 
% acc_ova = max(max(mean(Lacc_tensor_ova,3)));
% acc_ova_s = max(max(mean(Lacc_tensor_ova_s,3)));
% 
% acc_ova_ker = max(max(mean(Lacc_tensor_ova_ker,3)));
% acc_ova_ker_s = max(max(mean(Lacc_tensor_ova_ker_s,3)));

% acc_Rmulti = max(mean(Lacc_tensor_Rmulti,2));
% acc_Rmulti_s = max(mean(Lacc_tensor_Rmulti_s,2));
% 
% acc_Rmulti_ker = max(mean(Lacc_tensor_Rmulti_ker,2));
% acc_Rmulti_ker_s = max(mean(Lacc_tensor_Rmulti_ker_s,2));

function [ave,st] = comp_metrics(ten)
ave = max(max(mean(ten,3)));
[r,c] = find(mean(ten,3) == ave);
st = std(ten(r,c,:));
end

function [ave,st] = comp_metrics2(arr)
ave = max(mean(arr,2));
r = find(mean(arr,2) == ave,1,'first');
st = std(arr(r,:));
end