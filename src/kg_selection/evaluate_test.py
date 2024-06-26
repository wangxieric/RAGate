import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_repo", default='../data/outputs/inference_only_20231117130304_kg_select_bert_base_model/results/test/', type=str, required=False, help="result file repository")
    parser.add_argument("--prediction_file", default="predictions.json", type=str, required=False, help="result file name")

    args = parser.parse_args()
    results = json.load(open(args.result_repo + args.prediction_file, "r"))
    print(len(results))
    # calculate recall@3
    recall_1 = []
    recall_3 = []
    for result in results:
        for turn in result["turns"]:
            if turn['enrich'] == True:
                ground_truth = turn['kg_snippets']
                retrieved = turn['retrieved']
                recall_1.append(sum([1 for i in retrieved[:1] if i in ground_truth]) / len(ground_truth))
                recall_3.append(sum([1 for i in retrieved if i in ground_truth]) / len(ground_truth))
    
    print("Recall@1: ", sum(recall_1) / len(recall_1))
    print("Recall@3: ", sum(recall_3) / len(recall_3))



if __name__ == '__main__':
    main()
