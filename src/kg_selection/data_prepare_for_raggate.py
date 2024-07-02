import json
import pandas as pd

def process_and_structure(data):
    utterances = []
    labels = []
    contexts = []
    full_contexts = []
    system_response_context_only = []
    system_response_context_response = []
    system_response_labels = []
    snippet_for_augmentation = []
    for data in data:
        turns = data['turns']
        context = "" # context includes the converation history without current response
        full_context = ""  # full context includes the conversation history with current system response
        for idx, turn in enumerate(turns):
            utterance = turn['utterance']
            label = turn['enrich']
            
            # collecting previous turns of conversations
            if idx > 0:
                context += turn['speaker'] + ": " + turns[idx-1]['utterance'] + " "
                contexts.append(context)
            else:
                contexts.append("beginning of conversation")

            if turn['speaker'] == "SYSTEM":
                system_response_context_only.append(full_context)
                
                # you can vary the use of knowledge snippet from top 1 to top 3 or other settings 
                snippet_for_augmentation.append(' '.join(turn['retrieved_snippets'][:3]))
            
            # collecting full context for system responses
            full_context += turn['speaker'] + ": " + utterance + " "
            if turn['speaker'] == "SYSTEM":
                system_response_context_response.append(full_context)
                system_response_labels.append(label)

            full_contexts.append(full_context)
            
            utterances.append(utterance)
            labels.append(label)
    
    # convert data to dictionary
    data_dict = {
        "utterances": utterances,
        "labels": labels,
        "contexts": contexts,
        "full_contexts": full_contexts,
        "system_response_context_only": system_response_context_only,
        "system_response_context_response": system_response_context_response,
        "system_response_labels": system_response_labels,
        "snippet_for_augmentation": snippet_for_augmentation
    }
    return data_dict
   
# set the root directory that saves the data with aggregated relevant knowledge snippets
root_dir = "[root directory of the data]"
train_data = json.load(open(root_dir + 'train_data_with_snippets.json'))
test_data = json.load(open(root_dir + 'test_data_with_snippets.json'))
dev_data = json.load(open(root_dir + 'valid_data_with_snippets.json'))

formatted_train_data = process_and_structure(train_data)
formatted_test_data = process_and_structure(test_data)
formatted_dev_data = process_and_structure(dev_data)


data_options = ['ctx-only', 'ctx-(syn-resp)', 'ctx-(syn-resp)-ner', 'ctx-(syn-resp)-ner-source', 'ctx-(syn-resp)-ner-know']

instructions = {
    "ctx-only":"Analyse the conversational context so far. Estimate if augmenting the response with external knowledge is helpful with an output of 'True' or 'False' only.",
    "ctx-resp": "Analyse the conversation. Estimate if augmenting the latest utterance with external knowledge is helpful with an output of 'True' or 'False' only.",
    "ctx-(syn-resp)":"Analyse the conversational context so far. Generate an appropriate response. Estimate if augmenting the response with external knowledge is helpful with an output of 'True' or 'False' only.",
    "ctx-(syn-resp)-ner":"Analyse the conversational context so far. Generate an appropriate response. Consider the invovled entites. Estimate if augmenting the response with external knowledge is helpful with an output of 'True' or 'False' only.",
    "ctx-(syn-resp)-ner-source": "Analyse the conversational context so far. Generate an appropriate response. Consider the invovled entites. Estimate if augmenting the response with external knowledge sourced from the WikiHow website is helpful with an output of 'True' or 'False' only.",
    "ctx-(syn-resp)-ner-know": "Analyse the conversational context so far. Generate an appropriate system response. Consider the invovled entites and retrieved knowledge. Estimate if augmenting the response with retrieved knowledge is helpful with an output of 'True' or 'False' only."
}

# process train, dev and test data

data_type = "train"
if data_type == 'train':
    data = formatted_train_data
elif data_type == 'dev':
    data = formatted_dev_data
elif data_type == 'test':
    data = formatted_test_data

for data_option in data_options:
    fine_tune_data = pd.DataFrame(
        {'input': data['system_response_context_response'] if data_option == 'ctx-resp' else data['system_response_context_only'],
         'output': data['system_response_labels'],
         'knowledge': data['snippet_for_augmentation'] if data_option == 'ctx-(syn-resp)-ner-know' else None
        })
    fine_tune_data['instruction'] = instructions[data_option]

    output_df = fine_tune_data[['instruction', 'input', 'knowledge', 'output']] if data_option == 'ctx-(syn-resp)-ner-know' else fine_tune_data[['instruction', 'input', 'output']]

    context_json = output_df.to_json(orient = 'records', lines = True).splitlines()
    with open(f"../../data/lm_finetune_data/" + data_option + "_" + data_type + ".json", "w") as f:
        for line in context_json:
            f.write(f"{line}\n")
    output_df.to_csv(f"../../data/lm_finetune_data/" + data_option + "_" + data_type + ".csv", index=False)