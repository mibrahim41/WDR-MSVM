import glob
import os

all_files = glob.glob('WDR_MSVM_pump_4_*.mat')
count = 1
for file in all_files:
    new_name = 'WDR_MSVM_pump_4_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1