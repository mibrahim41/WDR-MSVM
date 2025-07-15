# Importing Packages
import gurobipy as grb
import numpy as np
import scipy.io
import sklearn
from sklearn.datasets import make_classification

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

        # Defining Objective Function
        sum_var_s = grb.quicksum(var_s[n] for n in range(row_x))
        obj = var_lambda*self.epsilon + (1/row_x)*sum_var_s
        model.setObjective(obj,grb.GRB.MINIMIZE)

        # Solving the Problem
        model.optimize()

        # Storing Results
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
        """test_data: N_test*P array of x data (N_test samples and P features)"""
        
        x_test = test_data
        row_x,col_x = x_test.shape
        y_pred = np.zeros([row_x,self.num_classes])
        
        for n in range(row_x):
            similarity_scores = np.matmul(self.M_opt,x_test[n])
            prediction = np.argmax(similarity_scores)
            y_pred[n,prediction] = 1
            
        return y_pred
    
    
    def evaluate_accuracy(self,y_true,y_pred):
        """y_true: N_test*C array of true labels (N_test samples and C classes)
           y_pred: N_test*C array of predicted labels (N_test samples and C classes)"""
        
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
        """y_true: N_test*C array of true labels (N_test samples and C classes)
           y_pred: N_test*C array of predicted labels (N_test samples and C classes)"""
        
        row_y,col_y = y_true.shape
        true_classes = np.zeros(row_y)
        pred_classes = np.zeros(row_y)
        
        for n in range(row_y):
            true_classes[n] = np.nonzero(y_true[n])[0][0]
            pred_classes[n] = np.nonzero(y_pred[n])[0][0]
            
        conf_mat = sklearn.metrics.confusion_matrix(true_classes,pred_classes)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    

# Distributionally Robust One-vs-All SVM
class DR_OVA:
    """One-Vs-All distributionally robust binary SVM"""
    
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
        """distributionally robust SVM without support information """
        
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}

        # Creating Model
        model = grb.Model('DRSVM_without_support')
        model.setParam('OutputFlag', False)

        # Defining Decision Variables
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

        # Integrating Variables
        model.update()

        # Defining Constraints
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

        # Defining Objective function
        sum_var_s = grb.quicksum(var_s[i] for i in range(row))
        obj = var_lambda*self.epsilon + (1/row)*sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Solving the Problem
        model.optimize()

        # Storing the Results
        w_opt = np.array([var_w[i].x for i in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)

        return optimal
    
    def test(self,test_data):
        """test_data: N_test*P array of x data (N_test samples and P features)"""
        
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
        """y_true: N_test*1 array of true labels
           y_pred: N_test*1 array of predicted labels"""
        
        acc = 1-np.sum(y_pred != y_true)/len(y_true)
        
        return acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N_test*1 array of true labels
           y_pred: N_test*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
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
             n_classes, weights_train, weights_test, class_sep=2.0, n_clusters=1):
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
        
        
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    y_train_multi, y_train_ova = reformulate_labels(y_train)
    y_test_multi, y_test_ova = reformulate_labels(y_test)
    
    return x_train, y_train, y_train_multi, y_train_ova, x_test, y_test, y_test_multi, y_test_ova
        
# Running Experiments (Parameters can be changed for each experiment)       
num_experiments = 50
epsilon_vec = np.logspace(-6,1,num=25)
kappa_vec = np.linspace(0.05,1,num=20)
acc_tensor_multi = np.zeros([len(epsilon_vec),len(kappa_vec),num_experiments])
acc_tensor_ova = np.zeros([len(epsilon_vec),len(kappa_vec),num_experiments])

for k in range(num_experiments):
    
    # Data Generation
    x_train, y_train, y_train_multi, y_train_ova, \
    x_test, y_test, y_test_multi, y_test_ova = gen_data(n_train=200, 
                                                        n_test=2000, 
                                                        n_features=15, 
                                                        n_informative=15, 
                                                        n_redundant=0, 
                                                        n_classes=8, 
                                                        weights_train=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], 
                                                        weights_test=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], 
                                                        class_sep=1.5, 
                                                        n_clusters=1)
    
    train_data_multi = {'x': x_train, 'y': y_train_multi}
    train_data_ova = {'x': x_train, 'y': y_train_ova}
    
    for i in range(len(epsilon_vec)):
        for j in range(len(kappa_vec)):
            param_multi = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf')}
            classifier_multi = DR_MSVM(param_multi)
            optimal_multi = classifier_multi.train(train_data_multi)

            y_pred_multi = classifier_multi.test(x_test)
            acc_multi = classifier_multi.evaluate_accuracy(y_test_multi,y_pred_multi)
            acc_tensor_multi[i,j,k] = acc_multi
            
            
            
            param_ova = {'epsilon': epsilon_vec[i], 'kappa': kappa_vec[j], 'pnorm':float('Inf')}
            classifier_ova = DR_OVA(param_ova)
            optimal_ova = classifier_ova.train(train_data_ova)

            y_pred_ova = classifier_ova.test(x_test)
            acc_ova = classifier_ova.evaluate_accuracy(y_test,y_pred_ova)
            acc_tensor_ova[i,j,k] = acc_ova
            
            
acc_mat_multi = np.mean(acc_tensor_multi,axis=2)
acc_mat_ova = np.mean(acc_tensor_ova,axis=2)

# Storing Experiment Data
EPS,KAP = np.meshgrid(epsilon_vec, kappa_vec)
exp_info = {'n_train': 200,
            'n_test': 2000,
            'n_features': 15,
            'n_informative': 15,
            'n_redundant': 0,
            'n_classes': 8,
            'weights_train': [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
            'weights_test': [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
            'perc_wrong_y': 0,
            'class_sep': 1.5,
            'n_clusters': 1,
            'epsilon_vec': epsilon_vec,
            'kappa_vec': kappa_vec,
            'epsilon_mesh': EPS,
            'kappa_mesh': KAP,
            'n_runs': 50
           }
new_experiment = {}
new_experiment['kappa_vec'] = kappa_vec
new_experiment['epsilon_vec'] = epsilon_vec
new_experiment['kappa_mesh'] = KAP
new_experiment['epsilon_mesh'] = EPS

new_experiment['acc_mat_multi'] = acc_mat_multi
new_experiment['acc_tensor_multi'] = acc_tensor_multi
new_experiment['acc_mat_ova'] = acc_mat_ova
new_experiment['acc_tensor_ova'] = acc_tensor_ova
new_experiment['exp_info'] = exp_info

scipy.io.savemat('imbalance_experiment_8classes_15features_balanced.mat',new_experiment)