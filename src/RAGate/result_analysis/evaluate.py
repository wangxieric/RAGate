import gem_metrics

pred_dir = 'reformatted_result_output.txt'
ref_dir = 'test_system_utterances.txt' # path to reference file

with open(pred_dir, 'r') as f:
    list_of_predictions = f.readlines()

with open(ref_dir, 'r') as f:
    list_of_references = f.readlines()

preds = gem_metrics.texts.Predictions(list_of_predictions)
refs = gem_metrics.texts.References(list_of_references)

result = gem_metrics.compute(preds, refs, metrics_list=['bleu', 'rouge', 'bertscore'])  # add list of desired metrics here
print(result)