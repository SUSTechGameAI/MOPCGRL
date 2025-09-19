import json
import os
import numpy as np
import matplotlib.pyplot as plt

exp_info = {
    12: {'start_id': 1, 'seed': 23456},
    13: {'start_id': 3, 'seed': 45678},
    14: {'start_id': 6, 'seed': 76543},
    16: {'start_id': 2, 'seed': 12345},
    17: {'start_id': 4, 'seed': 34567},
    18: {'start_id': 5, 'seed': 54321},
    19: {'start_id': 7, 'seed': 87654},
    20: {'start_id': 8, 'seed': 98765},
    21: {'start_id': 9, 'seed': 89123},
    22: {'start_id': 10, 'seed': 20255}
}

exp_info_29 = {
    4: {'start_id': 1, 'end_id': 11, 'seed': 23456},
    31: {'start_id': 11, 'end_id': 21, 'seed': 45678},
    32: {'start_id': 21, 'end_id': 31, 'seed': 76543},
    6: {'start_id': 31, 'end_id': 41, 'seed': 12345},
    9: {'start_id': 41, 'end_id': 51, 'seed': 34567}
}

exp_info_25 = {
    25: {'start_id': 1, 'end_id': 11, 'seed': 23456},
    26: {'start_id': 11, 'end_id': 21, 'seed': 45678},
    27: {'start_id': 21, 'end_id': 31, 'seed': 76543},
    7: {'start_id': 31, 'end_id': 41, 'seed': 12345},
    8: {'start_id': 41, 'end_id': 51, 'seed': 34567}
}

exp_info_20 = {
    3: {'start_id': 1, 'end_id': 11, 'seed': 23456},
    24: {'start_id': 11, 'end_id': 21, 'seed': 45678},
    0: {'start_id': 1, 'end_id': 11, 'seed': 12345},
    1: {'start_id': 11, 'end_id': 21, 'seed': 23456},
    2: {'start_id': 21, 'end_id': 31, 'seed': 34567}
}

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        aver_json = json.load(f)
        # print(aver_json)
        return aver_json

# output initial models
def output_egsac_to_txt(path, start_id, end_id, start_step ,end_step, exp_num):
    # with open(f'pretrain_evaluation/individual/model_r{run_id}_i{iter_id}_g{g}.json', 'r', encoding='utf-8') as f1:
        # ind_json = json.load(f1)
    txt_path = os.path.join(path, f'training_exp/experiment{exp_num}/initial_test.txt')
    all_lines = []
    for run_id in range(start_id, end_id):
        iter_id = start_step
        while iter_id <= end_step:
            json_path = os.path.join(path, f'model4pop/test_20w/model_aver_r{run_id}_i{iter_id}_g1.json')
            with open(json_path, 'r', encoding='utf-8') as f_json:
                aver_json = json.load(f_json)
                content_avg = 1 - aver_json[0].get("Fun Content Average", 0)
                behaviour_avg = 1 - aver_json[0].get("Fun Behaviour Average", 0)
                playability_avg = aver_json[0].get("Playability Average", 0)
                line = f"{content_avg} {behaviour_avg} {playability_avg}\n"
                all_lines.append(line)

            iter_id += 10000

    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.writelines(all_lines)  


# load from interation to end_iteration
def output_json_to_txt(path, iteration, end_iteration, exp_num): 
    # json_path = os.path.join(path, f'src/morl/analyze/exp_data/exp{exp_num}_model_aver{iteration}-{end_iteration}.json')
    json_path = os.path.join(out_path, f'exp{exp_num}_model_aver{iteration}-{end_iteration}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    # txt_path = os.path.join(path, f'src/morl/analyze/exp_data/exp{exp_num}_txt_aver{iteration}-{end_iteration}.txt')
    txt_path = os.path.join(out_path, f'exp{exp_num}_txt_aver{iteration}-{end_iteration}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        for item in json_data:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                content_avg = 1 - item[0].get("Fun Content Average", "")
                behaviour_avg = 1 - item[0].get("Fun Behaviour Average", "")
                playability_avg = item[0].get("Playability Average", "")
                ind = f"{content_avg} {behaviour_avg} {playability_avg}\n"
                f_txt.write(ind)
    print(f"Combined TXT saved to: {txt_path}")


# combine from interation to end_iteration
def output_json(path, iteration, end_iteration, exp_num, pop_size): 
    combined_data = []
    for iter in range(iteration, end_iteration, 10):
        for i in range(pop_size):
            # file_path_ca = os.path.join(path, f'training_exp/experiment{exp_num}/testing_ca/model_aver_iter{iter}_ind{i}.json')
            file_path_da = os.path.join(path, f'experiment{exp_num}/testing_da/model_aver_iter{iter}_ind{i}.json')
            # json_ca = load_json(file_path_ca)
            json_da = load_json(file_path_da)
            # if json_ca[0]["Playability Average"] > 24:
            #     combined_data.append(json_ca)
            if json_da[0]["Playability Average"] > 24:
                combined_data.append(json_da)
    output_path = os.path.join(out_path, f'exp{exp_num}_model_aver{iteration}-{end_iteration}.json')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(combined_data, out_f, ensure_ascii=False, indent=4)
    print(f"Combined JSON saved to: {output_path}")

# load training data
def output_txt(path, iteration, end_iteration, exp_num, seed_num):
    combined_data = []
    iter = iteration
    while iter <= end_iteration:
        for archive in ['ca', 'da']:
            file_path = os.path.join(path, f'training_exp/experiment{exp_num}/iteration{iter}/iter_pop/{archive}/gen{iter}_test{seed_num}.txt')
            if os.path.exists(file_path):
                data = np.loadtxt(file_path)
                if data.ndim == 1:
                    combined_data.append(data)
                else:
                    combined_data.extend(data)
            else:
                print(f"Warning: {file_path} not found.")
        iter += 10
    output_path = os.path.join(path1+out_path, f'/exp{exp_num}_train_aver{iteration}-{end_iteration}.txt')
    np.savetxt(output_path, combined_data)
    print(f"Combined TXT saved to: {output_path}")

def merge_txt_files(file_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    print(f"All files merged into: {output_file}")

def merge_json_files(exp_nums, file_list, merge_iter, output_file):
    algo_data = []  
    for exp_num, file_path in zip(exp_nums, file_list):
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)  
        for model in data:
            model = model[0]
            if model["Model ID"][0] == merge_iter:
                model["Experiment Number"] = exp_num
                # print(model)  
                algo_data.append(model)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(algo_data, outfile, indent=4)
        

def generate_file_list(base_path, iteration, end_iteration, exp_nums):
    file_list = []
    for exp_num in exp_nums:
        file_path = f'{base_path}/src/morl/analyze/exp_data_revision/exp{exp_num}_txt_aver{iteration}-{end_iteration}.txt'
        # file_path = f'{base_path}/src/morl/analyze/exp_data/exp{exp_num}_model_aver{iteration}-{end_iteration}.json'
        file_list.append(file_path)
    return file_list


# get data for specific generations
def read_gen(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[-20:]

def merge_exp(file_list, output_file):
    merged_lines = []
    for file_path in file_list:
        try:
            last_lines = read_gen(file_path)
            merged_lines.extend(last_lines)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    with open(output_file, 'w', encoding='utf-8') as out:
        out.writelines(merged_lines)

pop_size = 5
path1 = '../MOPCGRL'
path2 = '../training_exp'
out_path = '../MOPCGRL/src/morl/analyze/exp_data_revision'
# exp_nums_29w = [4, 31, 32, 6, 9]
# exp_nums_25w = [25, 26, 27, 7, 8]
# exp_nums_20w = [3, 24, 0, 1, 2]
# exp_nums = [12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 4, 31, 32, 6, 9, 25, 26, 27, 7, 8, 3, 24, 0, 1, 2]

iteration = 9
end_iteration = 200
seed_num = 12345
partial_iter = 5000

# output initial population testing results
exp_num = 12
seed = [23456, 45678, 76543, 12345, 34567, 54321, 87654, 98765, 89123, 20255]
end_id = 2
start_step = 200000
end_step = 200000

# for exp_num, info in exp_info_29.items():
#     # start_id = info['start_id']
#     # end_id = info['end_id']
#     # output_egsac_to_txt(path1, start_id, end_id, start_step, end_step, exp_num)
#     file_path =  [f'training_exp/experiment{exp_num}/initial_test.txt',
#                   f'{path1}/src/morl/analyze/exp_data/exp{exp_num}_txt_aver{iteration}-{end_iteration}.txt']
#     merge_txt_files(file_path, f'{path1}/src/morl/analyze/exp_data/exp{exp_num}_txt_aver0-{end_iteration}.txt')

exp_nums = [0, 1, 2, 3, 4]
for exp_num in exp_nums:
    output_json(path2, iteration, end_iteration, exp_num, pop_size)
    output_json_to_txt(path2, iteration, end_iteration, exp_num)
# output_txt(path1, iteration, end_iteration, exp_num, seed_num)
merge_iter = 9
# file_list = generate_file_list(path1, iteration, end_iteration, exp_nums)
# merge all output txt
# merge_txt_files(file_list, f'{path1}/src/morl/analyze/exp_data/merged_test_output_all.txt')
# merge specific generations
# merge_json_files(exp_nums, file_list, merge_iter, f'{path1}/src/morl/analyze/exp_data/merged_test_output_gen100.json')
# merge_exp(file_list, f'{path1}/src/morl/analyze/exp_data/merged_test_output_gen100_mix.txt')