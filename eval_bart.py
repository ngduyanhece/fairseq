from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))