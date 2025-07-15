num_exp = 50;
P_vec = [4,10,16,22,28];
runtime_subgrad_final_P = zeros([1,length(P_vec)]);
acc_subgrad_final_P = zeros([1,length(P_vec)]);
acc_std_subgrad_final_P = zeros([1,length(P_vec)]);
runtime_multi_final_P = zeros([1,length(P_vec)]);
acc_multi_final_P = zeros([1,length(P_vec)]);
acc_std_multi_final_P = zeros([1,length(P_vec)]);
runtime_ratio_P = zeros([1,length(P_vec)]);
std_dev_ratio_P = zeros([1,length(P_vec)]);
for c = 1:length(P_vec)
    acc_mat_subgrad = zeros([5,5,num_exp]);
    runtime_mat_subgrad = zeros([5,5,num_exp]);
    acc_vec_multi = zeros([1,num_exp]);
    runtime_vec_multi = zeros([1,num_exp]);
    for i = 1:num_exp
        filename = sprintf('subgrad_acc_P%d_%d.mat',P_vec(c),i);
        load(filename)
        acc_mat_subgrad(:,:,i) = acc_subgrad;
        runtime_mat_subgrad(:,:,i) = runtime_subgrad;
        acc_vec_multi(i) = acc_multi;
        runtime_vec_multi(i) = runtime_multi;
    end
    acc_multi_final_P(c) = mean(acc_vec_multi);
    runtime_multi_final_P(c) = mean(runtime_vec_multi);

    acc_mat_subgrad_mean = mean(acc_mat_subgrad,3);
    runtime_mat_subgrad_mean = mean(runtime_mat_subgrad,3);

    [~,col] = find(round(acc_mat_subgrad_mean,3) == max(max(round(acc_mat_subgrad_mean,3))),...
        1,'first');
    row = 4;
    acc_subgrad_final_P(c) = acc_mat_subgrad_mean(row,col);
    acc_std_subgrad_final_P(c) = std(acc_mat_subgrad(row,col,:),'',3);
    acc_std_multi_final_P(c) = std(acc_vec_multi);
    runtime_subgrad_final_P(c) = runtime_mat_subgrad_mean(row,col);
     runtime_vec_subgrad = zeros(1,length(runtime_vec_multi));
    for i = 1:length(runtime_vec_multi)
        runtime_vec_subgrad = runtime_mat_subgrad(row,col,i);
    end
    runtime_ratio_P(c) = mean(runtime_vec_multi./runtime_vec_subgrad);
    std_dev_ratio_P(c) = std(runtime_vec_multi./runtime_vec_subgrad);
end
