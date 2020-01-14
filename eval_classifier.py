from fairseq import checkpoint_utils, data, options, tasks
import os
from sklearn.metrics import f1_score
# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='simple_classification')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)

# Load model
print('| loading model from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
model = models[0]

### open file and read data
testdata = []
labels = []
predictions = []
with open(os.path.join('names','test.input')) as infile:  # using `with open()` you do not need to do `infile.close()`, that is done for you.
    for line in infile:  # I
        testdata.append(line.strip())

with open(os.path.join('names','test.label')) as infile:  # using `with open()` you do not need to do `infile.close()`, that is done for you.
    for line in infile:  # I
        labels.append(line.strip())

for d,l in zip(testdata,labels):

    # Tokenize into characters
    tokens = task.source_dictionary.encode_line(
        d, add_if_not_exist=False,
    )

    # Build mini-batch to feed to the model
    batch = data.language_pair_dataset.collate(
        samples=[{'id': -1, 'source': tokens}],  # bsz = 1
        pad_idx=task.source_dictionary.pad(),
        eos_idx=task.source_dictionary.eos(),
        left_pad_source=False,
        input_feeding=False,
    )

    # Feed batch to the model and get predictions
    preds = model(**batch['net_input'])
    top_scores, top_labels = preds[0].topk(k=2)
    label_name = task.target_dictionary.string([top_labels[0]])
    predictions.append(label_name)    

# calculate the metric 
f1 = f1_score(labels, predictions, average='weighted')
print("F1 score: {}".format(f1))