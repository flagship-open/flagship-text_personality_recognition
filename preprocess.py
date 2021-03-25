import os
import sys
import json

make_test_input = False
for data_type in ['A', 'C', 'E', 'N', 'O']:
	dir_name = 'CNN-LSTM_{}'.format(data_type)
	for data_name in ['train', 'dev', 'test']:
		with open(os.path.join(dir_name, 'Data/{}.json'.format(data_name))) as f:
			data = json.load(f)

		if data_name == 'test':
			data_path = os.path.join('Eval_Gold', '{}_{}.json'.format(data_name, data_type))
		else:
			data_path = os.path.join(dir_name, 'Data/{}.bin.json'.format(data_name))

		if data_name == 'test' and make_test_input == False:
			make_test_input = True
			with open('input.json', 'w') as f:
				bin_data = []
				for instances in data:
					bin_instances = []
					for instance in instances:
						bin_instance = {}
						bin_instance['utterance'] = instance['utterance']
						bin_instance['id'] = instance['id']
						bin_instances.append(bin_instance)
					bin_data.append(bin_instances)
				json.dump(bin_data, f, indent='\t', ensure_ascii=False)

		with open(data_path, 'w') as f:
			bin_data = []
			for instances in data:
				bin_instances = []
				for instance in instances:
					bin_instance = {}
					bin_instance['utterance'] = instance['utterance']
					bin_instance['id'] = instance['id']
					bin_instance['input'] = instance['input']
					bin_instance['output'] = 1 if instance['output'] == 0 else 0
					bin_instances.append(bin_instance)
				bin_data.append(bin_instances)
			json.dump(bin_data, f, indent='\t', ensure_ascii=False)
