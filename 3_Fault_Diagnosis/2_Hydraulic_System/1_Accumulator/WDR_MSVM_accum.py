# Importing Packages
import gurobipy as grb
import numpy as np
import scipy.io
import sklearn
import timeit
from sklearn import svm

def gen_data(perc_train,perc_wrong_y=0):
    def create_n_samples_vecs(n_samples,perc_train,n_classes,x,y):
        tot_train = round(perc_train*n_samples)
        n_samples_vec_train = np.ones(n_classes)*-1
        n_samples_vec_test = np.ones(n_classes)*-1

        for c in range(n_classes):
            x_c = x[y==c,:]
            n_samples_vec_train[c] = round(perc_train*len(x_c))

        count_add = 0
        count_subt = 0
        while sum(n_samples_vec_train) < tot_train:
            idx_add = count_add%n_classes
            n_samples_vec_train[idx_add] += 1
            count_add += 1

        while sum(n_samples_vec_train) > tot_train:
            idx_subt = count_subt%n_classes
            n_samples_vec_train[idx_subt] -= 1
            count_subt += 1

        for c in range(n_classes):
            x_c = x[y==c,:]
            row,col = x_c.shape
            n_samples_vec_test[c] = row - n_samples_vec_train[c]

        n_samples_vec_train = n_samples_vec_train.astype(int)
        n_samples_vec_test = n_samples_vec_test.astype(int)
        
        return n_samples_vec_train, n_samples_vec_test


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

    all_data = scipy.io.loadmat('hydraulic_sys_data.mat')
    x = all_data['x_all']
    y = all_data['y_accum'][0]
    
    n_samples,n_features = x.shape
    n_classes = len(np.unique(y))

    n_samples_train_vec, n_samples_test_vec = create_n_samples_vecs(n_samples,perc_train,n_classes,x,y)

    x_train = np.zeros([round(perc_train*n_samples),n_features])
    y_train = np.zeros(round(perc_train*n_samples))

    x_test = np.zeros([n_samples - round(perc_train*n_samples),n_features])
    y_test = np.zeros(n_samples - round(perc_train*n_samples))

    for c in range(n_classes):
        x_c = x[y==c,:]
        y_c = y[y==c]

        idx_vec = np.arange(len(x_c))
        idx_train_c = np.random.choice(idx_vec, size=n_samples_train_vec[c], replace=False)

        x_train[sum(n_samples_train_vec[0:c]):sum(n_samples_train_vec[0:c])+n_samples_train_vec[c],:] = x_c[idx_train_c,:]
        y_train[sum(n_samples_train_vec[0:c]):sum(n_samples_train_vec[0:c])+n_samples_train_vec[c]] = y_c[idx_train_c]

        x_temp_c = np.delete(x_c,idx_train_c,axis=0)
        y_temp_c = np.delete(y_c,idx_train_c,axis=0)

        x_test[sum(n_samples_test_vec[0:c]):sum(n_samples_test_vec[0:c])+n_samples_test_vec[c],:] = x_temp_c
        y_test[sum(n_samples_test_vec[0:c]):sum(n_samples_test_vec[0:c])+n_samples_test_vec[c]] = y_temp_c

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    if perc_wrong_y > 0:
            y_train = change_labels(perc_wrong_y,round(perc_train*n_samples),n_classes,y_train)

    y_train_multi, y_train_ova = reformulate_labels(y_train)
    y_test_multi, y_test_ova = reformulate_labels(y_test)
    
    return x_train, y_train, y_train_multi, y_train_ova, x_test, y_test, y_test_multi, y_test_ova

        
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
#         model.setParam('FeasibilityTol',1e-2)
#         model.setParam('OptimalityTol',1e-2)
        

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

        return optimal
    
    
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
    
    

# Distributionally Robust One-vs-All
class DR_OVA:
    """ One-Vs-All distributionally robust binary SVM"""
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        
    def train(self,train_data):
        """train_data: Dictionary with 2 keys:
            'x': N*P array of x data (N samples and P features)
            'y': C*N array of labels (N samples and C classes)"""
        
        x_train = train_data['x']
        y_train = train_data['y']

        row_x,col_x = x_train.shape
        row_y,col_y = y_train.shape
        self.num_classes = row_y
        
        optimal_all_classes = {}
        self.w_opt_all_classes = np.zeros([row_y,col_x])
        for c in range(self.num_classes):
            y_c = y_train[c]
            data_c = {'x':x_train, 'y':y_c}
            opt_dr_svm_c = self.dist_rob_svm_without_support(data_c)
            optimal_all_classes["optimal_" + str(c)] = opt_dr_svm_c
            self.w_opt_all_classes[c,:] = opt_dr_svm_c["w"]
            
        return optimal_all_classes    
        
    def dist_rob_svm_without_support(self,data):
        """ distributionally robust SVM without support information """
        
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}

        # Step 0: create model
        model = grb.Model('DRSVM_without_support')
        model.setParam('OutputFlag', False)
#         model.setParam('FeasibilityTol',1e-2)
#         model.setParam('OptimalityTol',1e-2)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for i in range(row):
            var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS,)
        for j in range(col):
            var_w[j] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[j] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for i in range(row):
            model.addConstr(
                1 - y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                              for j in range(col)) <= var_s[i])
            model.addConstr(
                1 + y_train[i] * grb.quicksum(var_w[j] * x_train[i, j]
                                              for j in range(col)) -
                self.kappa * var_lambda <= var_s[i])

        if self.pnorm == 1:
            for j in range(col):
                model.addConstr(var_w[j] <= slack_var[j])
                model.addConstr(-var_w[j] <= slack_var[j])
            model.addConstr(grb.quicksum(slack_var[j]
                                         for j in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[j] * var_w[j]
                             for j in range(col)) <= var_lambda*var_lambda)

        elif self.pnorm == float('Inf'):
            for j in range(col):
                model.addConstr(var_w[j] <= var_lambda)
                model.addConstr(-var_w[j] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[i] for i in range(row))
        obj = var_lambda*self.epsilon + (1/row)*sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[i].x for i in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)

        return optimal
    
    def test(self,test_data):
        """test_data: N*P array of x data (N samples and P features)"""
        
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x])
        
        for n in range(row_x):
            scores = np.ones([self.num_classes])*-1e10
            for c in range(self.num_classes):
                w_c = self.w_opt_all_classes[c]
                test_sample = x_test[n]
                pred_c = np.sum(w_c*test_sample)
                scores[c] = pred_c
            y_pred[n] = np.argmax(scores)
        
        return y_pred
    
    def evaluate_accuracy(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
        
        acc = 1-np.sum(y_pred != y_true)/len(y_true)
        
        return acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
    
    
# Kernel Distributionally Robust Multiclass SVM
class kDR_MSVM:
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.kernel = param['kernel']
        if self.kernel == "rbf" or self.kernel == "laplacian":
            self.gamma = param["gamma"]
        elif self.kernel == "poly":
            self.gamma = param["gamma"]
            self.d = param["d"]
        
        
        
    def compute_kernel_entry(self,x1,x2):
        if self.kernel == "rbf":
            return np.exp(-self.gamma*(np.linalg.norm(x1-x2, ord=2)**2))
        elif self.kernel == "laplacian":
            return np.exp(-self.gamma*np.linalg.norm(x1-x2, ord=1))
        elif self.kernel == "poly":
            return (self.gamma*np.sum(x1*x2) + 1)**self.d
        
        
    def train(self,train_data):
        
        x_train = train_data['x']
        y_train = train_data['y']
        
        row_x,col_x = x_train.shape
        row_y,col_y = y_train.shape
        self.num_classes = col_y
        
        if self.gamma == 'Auto':
            self.gamma = 1/col_x
        
        k_train = np.zeros([row_x,row_x])
        for n1 in range(row_x):
            for n2 in range(row_x):
                k_train[n1,n2] = self.compute_kernel_entry(x_train[n1,:],x_train[n2,:])
                
        optimal = {}
        
        # Creating Model
        model = grb.Model('kDRMSVM')
        model.setParam('OutputFlag',False)
#         model.setParam('FeasibilityTol',1e-2)
#         model.setParam('OptimalityTol',1e-2)
#         model.setParam('NonConvex', 2)
        
        # Defining Decision Variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_A = {}
        for n in range(row_x):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            for c in range(col_y):
                var_A[c,n] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
                
                
        model.update()
        
        # Defining Constraints
        for n in range(row_x):
            for c1 in range(col_y):
                if y_train[n,c1] == 0:
                    temp = 0
                else:
                    temp = 1
                correct_class = np.nonzero(y_train[n])[0][0]
                model.addConstr(
                    grb.quicksum(var_A[c1,j]*k_train[n,j] for j in range(row_x)) - temp + 1 -
                    grb.quicksum(var_A[correct_class,j]*k_train[n,j] for j in range(row_x)) <= var_s[n]
                )
                
                for c2 in range(col_y):
                    if c2 != correct_class:
                        if c2 == c1:
                            temp = 1
                        else:
                            temp = 0
                        model.addConstr(
                            grb.quicksum(var_A[c1,j]*k_train[n,j] for j in range(row_x)) - temp + 1 -
                            grb.quicksum(var_A[c2,j]*k_train[n,j] for j in range(row_x)) - self.kappa*var_lambda <= var_s[n]
                        )
                        
        for c1 in range(col_y):
            for c2 in range(col_y):
                if c2 > c1:
                    model.addQConstr(
                        grb.quicksum(var_A[c1,n1]*k_train[n1,n2]*var_A[c1,n2]
                                    for n1 in range(row_x) 
                                    for n2 in range(row_x)) +
                        grb.quicksum(var_A[c2,n1]*k_train[n1,n2]*var_A[c2,n2]
                                    for n1 in range(row_x)
                                    for n2 in range(row_x)) <= var_lambda*var_lambda)
                
        # Define Objective Function
        sum_var_s = grb.quicksum(var_s[n] for n in range(row_x))
        obj = var_lambda*self.epsilon + (1/row_x)*sum_var_s
        model.setObjective(obj,grb.GRB.MINIMIZE)
        
        # Solve the Problem
        model.optimize()
        
        # Store Results
        A_opt = np.ones([col_y,row_x])
        for n in range(row_x):
            for c in range(col_y):
                A_opt[c,n] = var_A[c,n].x
        
        self.A_opt = A_opt
        results_dict = {
            'A': A_opt,
            'objective_value': model.ObjVal,
            'diagnosis': model.status
        }
        optimal.update(results_dict)
        
        return optimal
    
    def test(self,test_data,train_data):
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x,self.num_classes])
        
        x_train = train_data['x']
        y_train = train_data['y']
        row_x_train,col_x_train = x_train.shape
        row_y,col_y = y_train.shape
        
        for n1 in range(row_x):
            similarity_scores = np.zeros([col_y])
            for c in range(col_y):
                k_vec = np.zeros([row_x_train])
                for n2 in range(row_x_train):
                    k_vec[n2] = self.compute_kernel_entry(x_test[n1,:],x_train[n2,:])
                similarity_scores[c] = np.sum(self.A_opt[c,:]*k_vec)
                
            prediction = np.argmax(similarity_scores)
            y_pred[n1,prediction] = 1
            
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
    
    

# Kernel Distributionally Robust One-vs-All SVM
class kDR_OVA:
    """ One-Vs-All distributionally robust binary SVM"""
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.kernel = param['kernel']
        if self.kernel == "rbf" or self.kernel == "laplacian":
            self.gamma = param["gamma"]
        elif self.kernel == "poly":
            self.gamma = param["gamma"]
            self.d = param["d"]

            
    def compute_kernel_entry(self,x1,x2):
        if self.kernel == "rbf":
            return np.exp(-self.gamma*(np.linalg.norm(x1-x2, ord=2)**2))
        elif self.kernel == "laplacian":
            return np.exp(-self.gamma*np.linalg.norm(x1-x2, ord=1))
        elif self.kernel == "poly":
            return (self.gamma*np.sum(x1*x2) + 1)**self.d
        
    def train(self,train_data):
        """train_data: Dictionary with 2 keys:
            'x': N*P array of x data (N samples and P features)
            'y': C*N array of labels (N samples and C classes)"""
        
        x_train = train_data['x']
        y_train = train_data['y']

        row_x,col_x = x_train.shape
        row_y,col_y = y_train.shape
        self.num_classes = row_y
        
        if self.gamma == 'Auto':
            self.gamma = 1/col_x
        
        k_train = np.zeros([row_x,row_x])
        for n1 in range(row_x):
            for n2 in range(row_x):
                k_train[n1,n2] = self.compute_kernel_entry(x_train[n1,:],x_train[n2,:])
                
        optimal_all_classes = {}
        self.alpha_opt_all_classes = np.zeros([row_y,row_x])
        
        for c in range(self.num_classes):
            y_c = y_train[c]
            data_c = {'k':k_train, 'y':y_c}
            opt_dr_svm_c = self.kernel_dist_rob_svm_without_support(data_c)
            optimal_all_classes["optimal_" + str(c)] = opt_dr_svm_c
            self.alpha_opt_all_classes[c,:] = opt_dr_svm_c["alpha"]
            
        return optimal_all_classes
    
    def kernel_dist_rob_svm_without_support(self, data):
        """ kernelized distributionally robust SVM """
        k_train = data['k']
        y_train = data['y'].flatten()

        row = k_train.shape[0]
        optimal = {}

        # Step 0: create model
        model = grb.Model('Ker_DRSVM')
        model.setParam('OutputFlag', False)
#         model.setParam('FeasibilityTol',1e-2)
#         model.setParam('OptimalityTol',1e-2)
#         model.setParam('NonConvex', 2)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_alpha = {}
        for i in range(row):
            var_s[i] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            var_alpha[i] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for i in range(row):
            model.addConstr(
                1 - y_train[i] * grb.quicksum(var_alpha[k] * k_train[k, i]
                                              for k in range(row)) <= var_s[i])
            model.addConstr(
                1 + y_train[i] * grb.quicksum(var_alpha[k] * k_train[k, i]
                                              for k in range(row)) -
                self.kappa * var_lambda <= var_s[i])
        model.addQConstr(
            grb.quicksum(var_alpha[k1] * k_train[k1, k2] * var_alpha[k2]
                         for k1 in range(row)
                         for k2 in range(row)) <= var_lambda * var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[i] for i in range(row))
        obj = var_lambda*self.epsilon + (1/row)*sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        alpha_opt = np.array([var_alpha[i].x for i in range(row)])
        tmp = {'alpha': alpha_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)

        return optimal
    
    def test(self,test_data,train_data):
        """test_data: N*P array of x data (N samples and P features)"""
        
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x])
        
        x_train = train_data['x']
        y_train = train_data['y']
        row_x_train,col_x_train = x_train.shape
        
        for n1 in range(row_x):
            scores = np.ones([self.num_classes])*-1e10
            for c in range(self.num_classes):
                alpha_c = self.alpha_opt_all_classes[c]
                test_sample = x_test[n1]
                k_vec = np.zeros([row_x_train])
                for n2 in range(row_x_train):
                    k_vec[n2] = self.compute_kernel_entry(x_test[n1,:],x_train[n2,:])
                
                pred_c = np.sum(alpha_c*k_vec)
                scores[c] = pred_c
            y_pred[n1] = np.argmax(scores)
        
        return y_pred
    
    def evaluate_accuracy(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
        
        acc = 1-np.sum(y_pred != y_true)/len(y_true)
        
        return acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
    
epsilon_vec = [1e-5,1e-3,1e-1]
kappa_vec = [0.2,0.4,0.6]
C_vec = [1e-3,1e-1,1e1]

acc_tensor_multi = np.zeros([len(epsilon_vec),len(kappa_vec)])
acc_tensor_multi_ker_rbf = np.zeros([len(epsilon_vec),len(kappa_vec)])

acc_tensor_ova = np.zeros([len(epsilon_vec),len(kappa_vec)])
acc_tensor_ova_ker_rbf = np.zeros([len(epsilon_vec),len(kappa_vec)])
C_vec = [1e-3,1e-1,1e1]
acc_tensor_Rmulti2 = np.zeros([len(epsilon_vec)])
acc_tensor_Rmulti2_ker_rbf = np.zeros([len(epsilon_vec)])

x_train, y_train, y_train_multi, y_train_ova, x_test, y_test, y_test_multi, y_test_ova = gen_data(0.3,0)

train_data_multi = {'x': x_train, 'y': y_train_multi}
train_data_ova = {'x': x_train, 'y': y_train_ova}

for i in range(len(epsilon_vec)):
    
    
    lin_clf = svm.SVC(kernel='linear',max_iter=int(1e6),C=C_vec[i])
    lin_clf.fit(x_train, y_train)
    result = lin_clf.predict(x_test)
    acc_reg = 1-np.sum(result != y_test)/len(y_test)
    acc_tensor_Rmulti2[i] = acc_reg
    
    lin_clf = svm.SVC(kernel='rbf',max_iter=int(1e6),gamma='auto',C=C_vec[i])
    lin_clf.fit(x_train, y_train)
    result = lin_clf.predict(x_test)
    acc_reg = 1-np.sum(result != y_test)/len(y_test)
    acc_tensor_Rmulti2_ker_rbf[i] = acc_reg

    
    
    for j in range(len(kappa_vec)):
        param_multi = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf')}
        param_multi_ker_rbf = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf'),
                                  'kernel': 'rbf', 'gamma': 'Auto'}
        
        param_ova = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf')}
        param_ova_ker_rbf = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf'),
                                'kernel': 'rbf', 'gamma': 'Auto'}
        
        classifier_multi = DR_MSVM(param_multi)
        optimal_multi = classifier_multi.train(train_data_multi)
        y_pred_multi = classifier_multi.test(x_test)
        acc_multi = classifier_multi.evaluate_accuracy(y_test_multi,y_pred_multi)
        acc_tensor_multi[i,j] = acc_multi
        
        classifier_ova = DR_OVA(param_ova)
        optimal_ova = classifier_ova.train(train_data_ova)
        y_pred_ova = classifier_ova.test(x_test)
        acc_ova = classifier_ova.evaluate_accuracy(y_test,y_pred_ova)
        acc_tensor_ova[i,j] = acc_ova
        
        if kappa_vec[j] == 0.4 and epsilon_vec[i] == 1e-3:
            classifier_multi_ker_rbf = kDR_MSVM(param_multi_ker_rbf)
            optimal_multi_ker_rbf = classifier_multi_ker_rbf.train(train_data_multi)
            y_pred_multi_ker_rbf = classifier_multi_ker_rbf.test(x_test,train_data_multi)
            acc_multi_ker_rbf = classifier_multi_ker_rbf.evaluate_accuracy(y_test_multi,y_pred_multi_ker_rbf)
            acc_tensor_multi_ker_rbf[i,j] = acc_multi_ker_rbf
            
            classifier_ova_ker_rbf = kDR_OVA(param_ova_ker_rbf)
            optimal_ova_ker_rbf = classifier_ova_ker_rbf.train(train_data_ova)
            y_pred_ova_ker_rbf = classifier_ova_ker_rbf.test(x_test,train_data_ova)
            acc_ova_ker_rbf = classifier_ova_ker_rbf.evaluate_accuracy(y_test,y_pred_ova_ker_rbf)
            acc_tensor_ova_ker_rbf[i,j] = acc_ova_ker_rbf
        
        
exp_dict = {'acc_tensor_multi':acc_tensor_multi,
            'acc_tensor_multi_ker_rbf':acc_tensor_multi_ker_rbf,
            'acc_tensor_ova':acc_tensor_ova,
            'acc_tensor_ova_ker_rbf':acc_tensor_ova_ker_rbf,
            'acc_tensor_Rmulti2':acc_tensor_Rmulti2,
            'acc_tensor_Rmulti2_ker_rbf':acc_tensor_Rmulti2_ker_rbf,
            'epsilon_vec':epsilon_vec,
            'kappa_vec':kappa_vec}

filename = 'WDR_MSVM_accum_'+str(int(timeit.default_timer()))+str(np.random.randint(0,high=1000))+'.mat'
scipy.io.savemat(filename,exp_dict)
        