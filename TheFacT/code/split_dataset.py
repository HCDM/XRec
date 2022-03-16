import random

fin = open('yelp.txt', encoding='UTF-8')
fout1 = open('yelp_train.txt', 'w')
fout2 = open('yelp_test.txt', 'w')
lines = fin.readlines()
train_ratio = 0.9
for line in lines:
    if random.random() < train_ratio:
        fout1.writelines(line)
    else:
        fout2.writelines(line)











