import os

f_s_list = [2, 1.5, 1, 0.5, 0.2]

g_end_list = [500, 1000, 2000, 5000, 10000, 20000, 60000]

h_i_list = [(10000 * i, 10000 * (i + 1)) for i in range(6)]

i_seed_list = [1, 12, 123, 1234, 12345, 123456]

for s in f_s_list:
    os.system("python SKs_model.py " + str(s) + " 0 10000 0 relu")

for train_end in g_end_list:
    os.system("python SKs_model.py 0.2 0 " + str(train_end) + " 0 relu")

for train_begin, train_end in h_i_list:
    os.system("python SKs_model.py 0.2 " + str(train_begin) + " " + str(train_end) + " 0 relu")

for seed in i_seed_list:
    os.system("python SKs_model.py 0.2 0 10000 " + str(seed) + " relu")

for activation in ["sigmoid", "relu"]:
    os.system("python SKs_model.py 0.2 1 10000 0 " + activation)