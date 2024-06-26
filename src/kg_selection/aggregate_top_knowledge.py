from config import parameters as conf
import pickle as p
import json


# update with the location of the result file (i.e. res_dialog_[train/valid/test].pkl)
results_path = "[result_file_dir]"
data_type = "test"
snippet_scores = p.load(open(results_path + "res_dialog_" + data_type + ".pkl", "rb"))
data_input = json.load(open('../../data/ketod/'+ data_type + '.json'))

count = 0
total_count = 0
for data in data_input:
    dialogue_id = data['dialogue_id']
    for idx, turn in enumerate(data['turns']):
        if turn['speaker'] == 'SYSTEM':
            total_count += 1
            this_turn_id = idx // 2
            this_ind = str(dialogue_id) + "_" + str(this_turn_id)
            if this_ind in snippet_scores:
                this_turn_snippet_score = snippet_scores[this_ind]
                sorted_snippet_score = sorted(this_turn_snippet_score, key=lambda x: x['score'], reverse=True)
                top_5_ids = [x['snippet'] for x in sorted_snippet_score[:5]]
                snippet_pool = data['entity_passages_sents']
                all_snippets = {}
                for each_query in snippet_pool:
                    for each_passage in snippet_pool[each_query]:
                        passage_title = each_passage[0]
                        for each_snippet in each_passage[1:]:
                            all_snippets[int(each_snippet[0])] = passage_title + " " + each_snippet[1]
                turn['retrieved_snippets'] = [all_snippets[x] for x in top_5_ids]
            else:
                print("this_turn_id not in snippet_scores: ", dialogue_id, this_ind)
                count += 1

json.dump(data_input, open(results_path + data_type + "_data_with_snippets.json", "w"))