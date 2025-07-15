num_exp = 50;
N_vec = [1000,2000,3000,4000,5000];
runtime_subgrad_final_N = zeros([1,length(N_vec)]);
acc_subgrad_final_N = zeros([1,length(N_vec)]);
acc_std_subgrad_final_N = zeros([1,length(N_vec)]);
runtime_multi_final_N = zeros([1,length(N_vec)]);
acc_multi_final_N = zeros([1,length(N_vec)]);
acc_std_multi_final_N = zeros([1,length(N_vec)]);
runtime_ratio_N = zeros([1,length(N_vec)]);
std_dev_ratio_N = zeros([1,length(N_vec)]);
for c = 1:length(N_vec)
    acc_mat_subgrad = zeros([5,5,num_exp]);
    runtime_mat_subgrad = zeros([5,5,num_exp]);
    acc_vec_multi = zeros([1,num_exp]);
    runtime_vec_multi = zeros([1,num_exp]);
    for i = 1:num_exp
        filename = sprintf('subgrad_acc_N%d_%d.mat',N_vec(c),i);
        load(filename)
        acc_mat_subgrad(:,:,i) = acc_subgrad;
        runtime_mat_subgrad(:,:,i) = runtime_subgrad;
        acc_vec_multi(i) = acc_multi;
        runtime_vec_multi(i) = runtime_multi;
    end
    acc_multi_final_N(c) = mean(acc_vec_multi);
    runtime_multi_final_N(c) = mean(runtime_vec_multi);

    acc_mat_subgrad_mean = mean(acc_mat_subgrad,3);
    runtime_mat_subgrad_mean = mean(runtime_mat_subgrad,3);

    [~,col] = find(round(acc_mat_subgrad_mean,3) == max(max(round(acc_mat_subgrad_mean,3))),...
        1,'first');
    row = 4;
    acc_subgrad_final_N(c) = acc_mat_subgrad_mean(row,col);
    acc_std_subgrad_final_N(c) = std(acc_mat_subgrad(row,col,:),'',3);
    acc_std_multi_final_N(c) = std(acc_vec_multi);
    runtime_subgrad_final_N(c) = runtime_mat_subgrad_mean(row,col);
    runtime_vec_subgrad = zeros(1,length(runtime_vec_multi));
    for i = 1:length(runtime_vec_multi)
        runtime_vec_subgrad = runtime_mat_subgrad(row,col,i);
    end
    runtime_ratio_N(c) = mean(runtime_vec_multi./runtime_vec_subgrad);
    std_dev_ratio_N(c) = std(runtime_vec_multi./runtime_vec_subgrad);
end
