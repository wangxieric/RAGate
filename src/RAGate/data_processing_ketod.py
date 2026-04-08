'''

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
'''

import json


class ketod_data_processing:
    def __init__(self, dataset_dir):
        """
        Load a KETOD-format JSON file that has been enriched with retrieved knowledge
        snippets (e.g. produced by aggregate_top_knowledge.py).

        :param dataset_dir: Path to the JSON dataset file
                            (e.g. test_data_with_snippets.json)
        """
        self.dataset_dir = dataset_dir
        with open(dataset_dir, 'r', encoding='utf8') as f:
            self.data = json.load(f)

    def process_data(self):
        """
        Iterate over every SYSTEM turn in the dataset and extract the information
        needed for RAGate inference.

        The returned lists are aligned: index i always refers to the same SYSTEM turn.

        Returns
        -------
        dialogue_ids : list[str]
            Dialogue ID string for each SYSTEM turn.
        turn_idx : list[str]
            Zero-based SYSTEM-turn counter within its dialogue (e.g. "0", "1", …).
            Combined with dialogue_id as ``dialogue_id + "_" + turn_idx`` this forms
            the canonical key used by aggregate_top_knowledge.py.
        contexts : list[str]
            Conversation history *before* the current system response, formatted as
            ``"SPEAKER: utterance "`` sequences (matching data_prepare_for_raggate.py).
        contexts_and_system_responses : list[str]
            Same as ``contexts`` but with the current system utterance appended.
        retrieved_knowledge : list[str]
            Top-3 retrieved knowledge snippets joined by a space for each SYSTEM turn.
            Empty string when no snippets are available.
        labels : list[bool]
            Ground-truth ``enrich`` label for each SYSTEM turn.
        """
        dialogue_ids = []
        turn_idx = []
        contexts = []
        contexts_and_system_responses = []
        retrieved_knowledge = []
        labels = []

        for dialogue in self.data:
            dialogue_id = str(dialogue['dialogue_id'])
            turns = dialogue['turns']
            full_context = ""  # accumulates "SPEAKER: utterance " for all preceding turns

            for idx, turn in enumerate(turns):
                utterance = turn['utterance']
                speaker = turn['speaker']

                if speaker == 'SYSTEM':
                    # turn_num follows the convention in aggregate_top_knowledge.py:
                    #   this_turn_id = idx // 2  (integer division over the full turn list)
                    turn_num = idx // 2

                    dialogue_ids.append(dialogue_id)
                    turn_idx.append(str(turn_num))

                    # Context = everything before the current system response
                    contexts.append(full_context.strip())

                    # Retrieved knowledge snippets (top 3, space-joined)
                    if 'retrieved_snippets' in turn and turn['retrieved_snippets']:
                        retrieved_knowledge.append(' '.join(turn['retrieved_snippets'][:3]))
                    else:
                        retrieved_knowledge.append('')

                    # Ground-truth augmentation label
                    labels.append(turn.get('enrich', False))

                    # Extend the running context with this system utterance
                    full_context += speaker + ": " + utterance + " "

                    # Context including the current system response
                    contexts_and_system_responses.append(full_context.strip())

                else:
                    # USER turn — simply extend the running context
                    full_context += speaker + ": " + utterance + " "

        return dialogue_ids, turn_idx, contexts, contexts_and_system_responses, retrieved_knowledge, labels

    def add_prediction(self, predictions):
        """
        Write model predictions back into the in-memory dataset so they are
        preserved when save_data() is called.

        :param predictions: dict mapping ``dialogue_id + "_" + turn_num`` → int (0 or 1)
                            where 1 means "augment" and 0 means "do not augment".
        """
        for dialogue in self.data:
            dialogue_id = str(dialogue['dialogue_id'])
            turns = dialogue['turns']

            for idx, turn in enumerate(turns):
                if turn['speaker'] == 'SYSTEM':
                    turn_num = idx // 2
                    key = dialogue_id + '_' + str(turn_num)
                    if key in predictions:
                        turn['enrich_pred'] = bool(predictions[key])

    def save_data(self, output_path):
        """
        Persist the dataset (including any added predictions) to disk.

        :param output_path: Destination file path for the JSON output.
        """
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(self.data, f, indent=4)
