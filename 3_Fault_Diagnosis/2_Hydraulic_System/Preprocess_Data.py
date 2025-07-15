import numpy as np
import scipy

def extract_feats(sensor_name):
    whole_arr = np.loadtxt(sensor_name + '.txt', delimiter='\t')
    feat_arr = np.zeros([2205,2])
    
    for i in range(2205):
        feat_arr[i,0] = np.mean(whole_arr[i,:])
        feat_arr[i,1] = np.std(whole_arr[i,:])
        
    return feat_arr

sensor_names = ['PS1','PS2','PS3','PS4','PS5','PS6',
                'EPS1',
                'FS1','FS2',
                'TS1','TS2','TS3','TS4',
                'VS1','SE','CE','CP']

x_all = extract_feats(sensor_names[0])

for i in range(1,len(sensor_names)):
    x_new = extract_feats(sensor_names[i])
    x_all = np.concatenate([x_all,x_new],axis=1)
    
y_all = np.loadtxt('profile.txt', delimiter='\t')

_, y_cooler = np.unique(y_all[:,0],return_inverse=True)
_, y_valve = np.unique(y_all[:,1],return_inverse=True)
_, y_pump = np.unique(y_all[:,2],return_inverse=True)
_, y_accum = np.unique(y_all[:,3],return_inverse=True)

y_cooler = y_cooler.astype(int)
y_valve = y_valve.astype(int)
y_pump = y_pump.astype(int)
y_accum = y_accum.astype(int)


dict_to_save = {'x_all': x_all,
                'y_cooler': y_cooler,
                'y_pump': y_pump,
                'y_accum': y_accum,
                'y_valve': y_valve
               }

scipy.io.savemat('hydraulic_sys_data.mat',dict_to_save)