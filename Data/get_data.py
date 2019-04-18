import os
import csv

f = open('ancillary_c11_n.filtres').read().splitlines();
fa = f[68:];
fa.insert(0,f[6]);

with open('data_naze.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(fa)
    
file.close()