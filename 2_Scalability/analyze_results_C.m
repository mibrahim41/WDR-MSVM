num_exp = 50;
C_vec = [4,6,8,10,12];
runtime_subgrad_final_C = zeros([1,length(C_vec)]);
acc_subgrad_final_C = zeros([1,length(C_vec)]);
runtime_multi_final_C = zeros([1,length(C_vec)]);
acc_multi_final_C = zeros([1,length(C_vec)]);
runtime_ratio_C = zeros([1,length(C_vec)]);
std_dev_ratio_C = zeros([1,length(C_vec)]);
for c = 1:length(C_vec)
    acc_mat_subgrad = zeros([5,5,num_exp]);
    runtime_mat_subgrad = zeros([5,5,num_exp]);
    acc_vec_multi = zeros([1,num_exp]);
    runtime_vec_multi = zeros([1,num_exp]);
    for i = 1:num_exp
        filename = sprintf('subgrad_acc_C%d_%d.mat',C_vec(c),i);
        load(filename)
        acc_mat_subgrad(:,:,i) = acc_subgrad;
        runtime_mat_subgrad(:,:,i) = runtime_subgrad;
        acc_vec_multi(i) = acc_multi;
        runtime_vec_multi(i) = runtime_multi;
    end
    acc_multi_final_C(c) = mean(acc_vec_multi);
    runtime_multi_final_C(c) = mean(runtime_vec_multi);

    acc_mat_subgrad_mean = mean(acc_mat_subgrad,3);
    runtime_mat_subgrad_mean = mean(runtime_mat_subgrad,3);

    [~,col] = find(round(acc_mat_subgrad_mean,3) == max(max(round(acc_mat_subgrad_mean,3))),...
        1,'first');
    row = 4;
    acc_subgrad_final_C(c) = acc_mat_subgrad_mean(row,col);
    runtime_subgrad_final_C(c) = runtime_mat_subgrad_mean(row,col);
    runtime_vec_subgrad = zeros(1,length(runtime_vec_multi));
    for i = 1:length(runtime_vec_multi)
        runtime_vec_subgrad = runtime_mat_subgrad(row,col,i);
    end
    runtime_ratio_C(c) = mean(runtime_vec_multi./runtime_vec_subgrad);
    std_dev_ratio_C(c) = std(runtime_vec_multi./runtime_vec_subgrad);
end


