import json

# load schemas from different folders
schema_train_dir = '../../data/SGD/train/schema.json'
schema_dev_dir = '../../data/SGD/dev/schema.json'
schema_test_dir = '../../data/SGD/test/schema.json'

schema_train = json.load(open(schema_train_dir, 'r'))
schema_dev = json.load(open(schema_dev_dir, 'r'))
schema_test = json.load(open(schema_test_dir, 'r'))

# present the statistics of the schemas
print('train: ', len(schema_train))
print('dev: ', len(schema_dev))
print('test: ', len(schema_test))

schema_all = schema_train

for schema in schema_dev:
    if schema not in schema_all:
        schema_all.append(schema)

for schema in schema_test:
    if schema not in schema_all:
        schema_all.append(schema)

print('all: ', len(schema_all))
json.dump(schema_all, open('../../data/SimpleTOD/entity_schemas/schema_all.json', 'w'), indent=4)