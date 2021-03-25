import os

for data_type in ['A', 'C', 'E', 'N', 'O']:
	os.chdir('CNN-LSTM_{}'.format(data_type))
	os.system('python Train.py')
	os.chdir('..')