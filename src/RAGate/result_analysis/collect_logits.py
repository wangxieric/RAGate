import json
import numpy as np

generated_outputs_dir = 'result_output.json'
generated_outputs = json.load(open(generated_outputs_dir, 'r'))

logits = []

for gen in generated_outputs:
    logits.append(gen.split(' ')[-1])

print(np.mean([float(logit) for logit in logits]))

# save logits
with open('knowledge_logits.txt', 'w') as f:
    json.dump(logits, f)
