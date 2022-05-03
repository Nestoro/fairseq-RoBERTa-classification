from fairseq.models.roberta import RobertaModel
import argparse

args_parser = argparse.ArgumentParser(description='Run RoBERTa classification Task on file')

args_parser.add_argument('--input', type=str, help='input file', required=True)
args_parser.add_argument('--output', type=str, help='output file', required=True)
args_parser.add_argument('--data_name', type=str, help='data name', required=True)
args_parser.add_argument('--checkpoint_path', type=str, help='checkpoint path', required=True)
args_parser.add_argument('--checkpoint_file', type=str, help='checkpoint file', required=True)
args_parser.add_argument('--classification_head_name', type=str, help='whatever you set --classification-head-name to during training', required=True)


args = args_parser.parse_args()

checkpoint_file = args.checkpoint_file
checkpoint_path = args.checkpoint_path
data_name_or_path = args.data_name
output_path = args.output
input_path = args.input
classification_head_name = args.classification_head_name

roberta = RobertaModel.from_pretrained(
    checkpoint_path,
    checkpoint_file=checkpoint_file,
    data_name_or_path=data_name_or_path
)

roberta.eval()

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

inputFile = open(input_path, 'r')
OutputFile = open(output_path, 'w')

OutputFile.flush()

Lines = inputFile.readlines()

for line in Lines:
    line = line.strip()
    tokens = roberta.encode(line)
    pred = label_fn(roberta.predict(classification_head_name, tokens).argmax().item())
    OutputFile.write(line + ';' +pred + '\n')

inputFile.close()
OutputFile.close()


