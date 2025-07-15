# Importing Packages
import gurobipy as grb
import numpy as np
import scipy.io
import sklearn
from sklearn.datasets import make_classification
import timeit

# Distributionally Robust Multiclass SVM
class DR_MSVM:
    """Distributionally robust multiclass SVM"""
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        
        
    def train(self,train_data):
        """train_data: Dictionary with 2 keys:
            'x': N*P array of x data (N samples and P features)
            'y': N*C array of labels (N samples and C classes)"""
        
        x_train = train_data['x']
        y_train = train_data['y']

        row_x,col_x = x_train.shape
        row_y,col_y = y_train.shape
        self.num_classes = col_y
        optimal = {}

        # Creating Model
        model = grb.Model('DRMSVM')
        model.setParam('OutputFlag',False)
        model.setParam('Method',2)
        model.setParam('BarConvTol',1e-2)

        # Defining Decision Variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_M = {}
        slack_var = {}
        for n in range(row_x):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        for p in range(col_x):
            for c in range(col_y):
                var_M[c,p] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

        if self.pnorm == 1:
            for p in range(col_x):
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    #         for c1 in range(col_y):
    #             for c2 in range(col_y):
    #                 slack_var[col_y*c1 + c2] = model.addVar(vtype=grb.GRB.CONTINUOUS)
    
        model.update()

        # Define Constraints
        for n in range(row_x):
            for c1 in range(col_y):
                if y_train[n,c1] == 0:
                    temp = 0
                else:
                    temp = 1
                correct_class = np.nonzero(y_train[n])[0][0]
                model.addConstr(
                    grb.quicksum(var_M[c1,p]*x_train[n,p] for p in range(col_x)) - temp + 1 -  
                    grb.quicksum(var_M[correct_class,p]*x_train[n,p] for p in range(col_x)) <= var_s[n]
                )

                for c2 in range(col_y):
                    if c2 != correct_class:
                        if c2 == c1:
                            temp = 1
                        else:
                            temp = 0
                        model.addConstr(
                            grb.quicksum(var_M[c1,p]*x_train[n,p] for p in range(col_x)) - temp + 1 -  
                            grb.quicksum(var_M[c2,p]*x_train[n,p] for p in range(col_x)) - self.kappa*var_lambda <= var_s[n]
                        )

        for c1 in range(col_y):
            for c2 in range(col_y):
                var_M_vec = {}
                for p in range(col_x):
                    var_M_vec[p] = var_M[c1,p] - var_M[c2,p]

                if self.pnorm == 1:
                    for p in range(col_x):
                        model.addConstr(var_M_vec[p] <= slack_var[p])
                        model.addConstr(-var_M_vec[p] <= slack_var[p])
                    model.addConstr(grb.quicksum(slack_var[p]
                                                 for p in range(col_x)) <= var_lambda)
                elif self.pnorm == 2:
                    model.addQConstr(
                        grb.quicksum(var_M_vec[p] * var_M_vec[p]
                                     for p in range(col_x)) <= var_lambda * var_lambda)

                elif self.pnorm == float('Inf'):
                    for p in range(col_x):
                        model.addConstr(var_M_vec[p] <= var_lambda)
                        model.addConstr(-var_M_vec[p] <= var_lambda)

        # Define Objective Function
        sum_var_s = grb.quicksum(var_s[n] for n in range(row_x))
        obj = var_lambda*self.epsilon + (1/row_x)*sum_var_s
        model.setObjective(obj,grb.GRB.MINIMIZE)

        # Solve the Problem
        model.optimize()

        # Store Results
        M_opt = np.ones([col_y,col_x])
        for p in range(col_x):
            for c in range(col_y):
                M_opt[c,p] = var_M[c,p].x
        self.M_opt = M_opt
        results_dict = {
            'M': M_opt,
            'objective_value': model.ObjVal,
            'diagnosis': model.status
        }
        optimal.update(results_dict)
        runtime = model.Runtime
        iter_count = model.BarIterCount

        return optimal, runtime, iter_count, var_lambda.x
    
    
    def test(self,test_data):
        """test_data: N*P array of x data (N samples and P features)"""
        
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x,self.num_classes])
        
        for n in range(row_x):
            similarity_scores = np.matmul(self.M_opt,x_test[n])
            prediction = np.argmax(similarity_scores)
            y_pred[n,prediction] = 1
            
        return y_pred
    
    
    def evaluate_accuracy(self,y_true,y_pred):
        """y_true: N*C array of true labels (N samples and C classes)
           y_pred: N*C array of predicted labels (N samples and C classes)"""
        
        row_y,col_y = y_true.shape
        incorrect_count = 0
        
        for n in range(row_y):
            true_class = np.nonzero(y_true[n])[0][0]
            predicted_class = np.nonzero(y_pred[n])[0][0]
            if true_class != predicted_class:
                incorrect_count = incorrect_count+1
                
        acc = 1 - (incorrect_count/row_y)
        return acc
    
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*C array of true labels (N samples and C classes)
           y_pred: N*C array of predicted labels (N samples and C classes)"""
        
        row_y,col_y = y_true.shape
        true_classes = np.zeros(row_y)
        pred_classes = np.zeros(row_y)
        
        for n in range(row_y):
            true_classes[n] = np.nonzero(y_true[n])[0][0]
            pred_classes[n] = np.nonzero(y_pred[n])[0][0]
            
        conf_mat = sklearn.metrics.confusion_matrix(true_classes,pred_classes)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat

def comp_grad_sample(x_sample,y_sample,kap,M,lam):
    P = len(x_sample)
    C = len(y_sample)
    
    curr_max = 0
    curr_g_M = np.zeros(M.shape)
    curr_g_lam = 0
    
    for c1 in range(C):
        for c2 in range(C):
            v_c = np.zeros([C,1])
            v_c[c1] = 1
            y_c = np.zeros([C,1])
            y_c[c2] = 1
            if list(y_c) == list(y_sample):
                temp = (M[c1]@x_sample - y_sample[c1]) + 1 - y_sample.T@M@x_sample
                if temp > curr_max:
                    curr_max = temp
                    curr_g_M = v_c@x_sample.T - y_sample@x_sample.T
                    curr_g_lam = 0
            else:
                temp = (M[c1]@x_sample - y_c[c1]) + 1 - M[c2]@x_sample - lam*kap
                if temp > curr_max:
                    cyrr_max = temp
                    curr_g_M = v_c@x_sample.T - y_c@x_sample.T
                    curr_g_lam = -kap
                    
    grad_lam = curr_g_lam
    grad_M = curr_g_M
    
    return grad_M, grad_lam

def comp_grad(x_train,y_train,kap,M,lam,eps):
    grad_M = np.zeros(M.shape)
    grad_lam = 0
    
    N,P = x_train.shape
    _,C = y_train.shape
    
    for n in range(len(x_train)):
        x_sample = np.zeros([P,1])
        x_sample[:,0] = x_train[n]
        y_sample = np.zeros([C,1])
        y_sample[:,0] = y_train[n]
        sample_grad_M, sample_grad_lam = comp_grad_sample(x_sample,y_sample,kap,M,lam)
        grad_M += sample_grad_M
        grad_lam += sample_grad_lam
        
    grad_lam += eps
    
    return grad_M, grad_lam

def comp_obj(x_train,y_train,kap,M,lam,eps):
    curr_obj = eps*lam
    
    N,P = x_train.shape
    _,C = y_train.shape
    
    for n in range(N):
        curr_max = 0
        x_sample = np.zeros([P,1])
        x_sample[:,0] = x_train[n]
        y_sample = np.zeros([C,1])
        y_sample[:,0] = y_train[n]
        for c1 in range(C):
            for c2 in range(C):
                v_c = np.zeros([C,1])
                v_c[c1] = 1
                y_c = np.zeros([C,1])
                y_c[c2] = 1
                if list(y_c) == list(y_sample):
                    temp = v_c.T@(M@x_sample - y_sample) + 1 - y_sample.T@M@x_sample
                    if temp > curr_max:
                        curr_max = temp
                        
                else:
                    temp = v_c.T@(M@x_sample - y_c) + 1 - y_c.T@M@x_sample - lam*kap
                    if temp > curr_max:
                        cyrr_max = temp
        curr_obj += curr_max
        
    return curr_obj

def lam_feasibility_check(lam,M,C,pnorm):
    for c1 in range(C):
        for c2 in range(C):
            curr_vec = M[c1] - M[c2]
            max_val = np.max(np.abs(curr_vec))
            if lam >= max_val:
                return True
            else:
                return False
            
def projection_prob(M,lam):
    C,P = M.shape
    
    mod = grb.Model('Planning_module')
    
    # Add decision variables
    var_lam = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb = 0)
    M_i = {}
    M_j = {}
    for p in range(P):
        M_j[p] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
        for c in range(C):
            M_i[c,p] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

            
    # Add constraints
    mod.addConstrs(M_i[c,p]>= M_j[p]-(var_lam/2) for p in range(P) for c in range(C))
    mod.addConstrs(M_i[c,p]<= M_j[p]+(var_lam/2) for p in range(P) for c in range(C))
    obj = (var_lam - lam)**2 + grb.quicksum((M_i[c,p]-M[c,p])**2 for p in range(P) for c in range(C))
    mod.Params.LogToConsole = 0
    mod.setObjective(obj, grb.GRB.MINIMIZE)
    mod.update()
    mod.optimize()
    mod.update()
    
    M_opt = np.ones([C,P])
    for p in range(P):
        for c in range(C):
            M_opt[c,p] = M_i[c,p].x
            
    lam_opt = var_lam.x
    
    return M_opt,lam_opt
                
    
# Distributionally Robust Multiclass SVM
class DR_MSVM_subgrad:
    """Distributionally robust multiclass SVM"""
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.max_iter = param['max_iter']
        self.gamma = param['gamma']
        
        
    def train(self,train_data):
        """train_data: Dictionary with 2 keys:
            'x': N*P array of x data (N samples and P features)
            'y': N*C array of labels (N samples and C classes)"""
        
        x_train = train_data['x']
        y_train = train_data['y']

        row_x,col_x = x_train.shape
        row_y,col_y = y_train.shape
        self.num_classes = col_y
        
        self.M_opt = np.zeros([col_y,col_x])
        lam = 0
        self.obj = []
        min_val = 1e20
        for t in range(self.max_iter):
            sg_M, sg_lam = comp_grad(x_train,y_train,self.kappa,self.M_opt,lam,self.epsilon)
            self.M_opt = self.M_opt - (self.gamma/(t+1))*sg_M
            lam = lam - (self.gamma/(t+1))*sg_lam
            check_feasibility = lam_feasibility_check(lam,self.M_opt,self.num_classes,self.pnorm)
            if check_feasibility == False:
                self.M_opt,lam = projection_prob(self.M_opt,lam)
            obj_val = comp_obj(x_train,y_train,self.kappa,self.M_opt,lam,self.epsilon)
            min_val = min(obj_val[0][0],min_val)
            self.obj.append(min_val)
#             print(t)
            
        
        return self.M_opt, self.obj
    
    def test(self,test_data):
        """test_data: N*P array of x data (N samples and P features)"""
        
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x,self.num_classes])
        
        for n in range(row_x):
            similarity_scores = np.matmul(self.M_opt,x_test[n])
            prediction = np.argmax(similarity_scores)
            y_pred[n,prediction] = 1
            
        return y_pred
    
    
    def evaluate_accuracy(self,y_true,y_pred):
        """y_true: N*C array of true labels (N samples and C classes)
           y_pred: N*C array of predicted labels (N samples and C classes)"""
        
        row_y,col_y = y_true.shape
        incorrect_count = 0
        
        for n in range(row_y):
            true_class = np.nonzero(y_true[n])[0][0]
            predicted_class = np.nonzero(y_pred[n])[0][0]
            if true_class != predicted_class:
                incorrect_count = incorrect_count+1
                
        acc = 1 - (incorrect_count/row_y)
        return acc
    
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*C array of true labels (N samples and C classes)
           y_pred: N*C array of predicted labels (N samples and C classes)"""
        
        row_y,col_y = y_true.shape
        true_classes = np.zeros(row_y)
        pred_classes = np.zeros(row_y)
        
        for n in range(row_y):
            true_classes[n] = np.nonzero(y_true[n])[0][0]
            pred_classes[n] = np.nonzero(y_pred[n])[0][0]
            
        conf_mat = sklearn.metrics.confusion_matrix(true_classes,pred_classes)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
    
# Setting Up Data Generation
def generate_n_sample_vec(weights, total_samples, n_classes):
    n_samples_vec = np.zeros(len(weights))
    
    for j in range(len(n_samples_vec)):
        n_samples_vec[j] = weights[j]*total_samples
        
    n_samples_vec = n_samples_vec.astype(int)
    
    count_add = 0
    count_subt = 0
    while sum(n_samples_vec) < total_samples:
        idx_add = count_add%n_classes
        n_samples_vec[idx_add] += 1
        count_add += 1
        
    while sum(n_samples_vec) > total_samples:
        idx_subt = count_subt%n_classes
        n_samples_vec[idx_subt] -= 1
        count_subt += 1
        
    return n_samples_vec


def change_labels(perc_wrong_y,total_samples,n_classes,y):
    n_wrong = round(perc_wrong_y*total_samples)
    idx_wrong = np.random.randint(0, high=total_samples, size=n_wrong)
    y[idx_wrong] = np.random.randint(0, high=n_classes, size=n_wrong)
    
    return y


def reformulate_labels(y_in):
    num_classes = max(y_in)
    num_samples = len(y_in)
    y_out_multi = np.zeros([num_samples,num_classes+1])
    y_out_ova = np.ones([num_classes+1,num_samples])*-1
    
    for i in range(num_samples):
        y_out_multi[i,y_in[i]] = 1
        y_out_ova[y_in[i],i] = 1

    return y_out_multi, y_out_ova


def gen_data(n_train, n_test, n_features, n_informative, n_redundant, 
             n_classes, weights_train, weights_test, perc_wrong_y=0, class_sep=2.0, n_clusters=1):
    n_samples = (n_train + n_test)*10
    x_all, y_all = make_classification(n_samples=n_samples,
                                       n_features=n_features,
                                       n_informative=n_informative, 
                                       n_redundant=n_redundant, 
                                       n_repeated=0, 
                                       n_classes=n_classes, 
                                       n_clusters_per_class=n_clusters, 
                                       weights=None, 
                                       flip_y=0, 
                                       class_sep=class_sep, 
                                       hypercube=True, 
                                       shift=0.0, 
                                       scale=1.0, 
                                       shuffle=False, 
                                       random_state=None)
    x_train = np.zeros([n_train,n_features])
    y_train = np.zeros(n_train)
    
    x_test = np.zeros([n_test,n_features])
    y_test = np.zeros(n_test)
    
    n_samples_vec_train = generate_n_sample_vec(weights_train, n_train, n_classes)
    n_samples_vec_test = generate_n_sample_vec(weights_test, n_test, n_classes)
    
    for c in range(n_classes):
        x_temp = x_all[y_all == c,:]
        y_temp = y_all[y_all == c]
        
        x_train[sum(n_samples_vec_train[0:c]):
                sum(n_samples_vec_train[0:c])+n_samples_vec_train[c],:] = x_temp[:n_samples_vec_train[c]]
        
        y_train[sum(n_samples_vec_train[0:c]):
                sum(n_samples_vec_train[0:c])+n_samples_vec_train[c]] = y_temp[:n_samples_vec_train[c]]
        
        x_test[sum(n_samples_vec_test[0:c]):
                sum(n_samples_vec_test[0:c])+n_samples_vec_test[c],:] = x_temp[n_samples_vec_train[c]:
                                                                               n_samples_vec_test[c]+n_samples_vec_train[c]]
        
        y_test[sum(n_samples_vec_test[0:c]):
                sum(n_samples_vec_test[0:c])+n_samples_vec_test[c]] = y_temp[n_samples_vec_train[c]:
                                                                             n_samples_vec_test[c]+n_samples_vec_train[c]]
        
    if perc_wrong_y > 0:
        y_train = change_labels(perc_wrong_y,n_train,n_classes,y_train)
        
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    y_train_multi, y_train_ova = reformulate_labels(y_train)
    y_test_multi, y_test_ova = reformulate_labels(y_test)
    
    return x_train, y_train, y_train_multi, y_train_ova, x_test, y_test, y_test_multi, y_test_ova


C = 10
max_iter_vec = [10,60,100,140,180]
stepsize_vec = [1e-2,1e-1,1e0,1e1,1e2]
num_exp = 1

acc_mat_subgrad_C = np.zeros([len(max_iter_vec),len(stepsize_vec),num_exp])
runtime_mat_subgrad_C = np.zeros([len(max_iter_vec),len(stepsize_vec),num_exp])


for j in range(len(stepsize_vec)):
    for i in range(len(max_iter_vec)):
        for k in range(num_exp):

            x_train, y_train, y_train_multi, y_train_ova, \
            x_test, y_test, y_test_multi, y_test_ova = gen_data(n_train=1000, 
                                                                n_test=2000, 
                                                                n_features=4, 
                                                                n_informative=4, 
                                                                n_redundant=0, 
                                                                n_classes=C, 
                                                                weights_train=np.ones(C)*(1/C), 
                                                                weights_test=np.ones(C)*(1/C), 
                                                                perc_wrong_y=0, 
                                                                class_sep=1.5, 
                                                                n_clusters=1)

            train_data_multi = {'x': x_train, 'y': y_train_multi}

            param_multi = {'epsilon': 1e-4, 'kappa': 0.5, 'pnorm':float('Inf'), 'gamma':stepsize_vec[j],
                            'max_iter':max_iter_vec[i]}
            classifier_multi = DR_MSVM_subgrad(param_multi)
            start_subgrad = timeit.default_timer()
            param_multi, obj_val_multi = classifier_multi.train(train_data_multi)
            stop_subgrad = timeit.default_timer()
            y_pred_multi = classifier_multi.test(x_test)
            acc_multi = classifier_multi.evaluate_accuracy(y_test_multi,y_pred_multi)
            acc_mat_subgrad_C[i,j,k] = acc_multi
            runtime_mat_subgrad_C[i,j,k] = stop_subgrad - start_subgrad
            
param_multi = {'epsilon': 1e-4, 'kappa': 0.5, 'pnorm':float('Inf')}
classifier_multi_orig = DR_MSVM(param_multi)
start_multi = timeit.default_timer()
optimal_multi, runtime_multi, iter_count_multi, opt_lam = classifier_multi_orig.train(train_data_multi)
stop_multi = timeit.default_timer()
y_pred_multi = classifier_multi_orig.test(x_test)
acc_multi = classifier_multi_orig.evaluate_accuracy(y_test_multi,y_pred_multi)

runtime_multi_C = stop_multi - start_multi
acc_multi_C = acc_multi
            
dict_to_save = {'acc_subgrad': acc_mat_subgrad_C,'runtime_subgrad':runtime_mat_subgrad_C,'acc_multi':acc_multi_C,
                'runtime_multi':runtime_multi_C}
filename = 'subgrad_acc_C10_'+str(int(timeit.default_timer()))+'.mat'
scipy.io.savemat(filename,dict_to_save)
                


