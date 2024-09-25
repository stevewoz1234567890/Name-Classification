from pathlib import Path
import torch
from torch.autograd import Variable
from train import n_hidden, n_categories, n_letters, all_categories, lineToTensor, RNN


# Just return an output given a line
def evaluate(line_tensor, weights_file: Path):
    rnn = RNN(n_letters, n_hidden, n_categories)
    rnn.load_state_dict(torch.load(weights_file))
    rnn.eval()
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output  


def predict(name, n_predictions: int, weights_file: Path):
    print(f"{n_letters=}, {n_hidden=}, {n_categories=}")
    output = evaluate(Variable(lineToTensor(name)), weights_file=weights_file)  

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print("(%.2f) %s" % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-w", type=Path, default=Path("./weights.pt"), help="path to model weights")
    parser.add_argument("-n", type=int, default=3, help="return top n mosty likely classes")
    parser.add_argument("name", type=str, help="name to classify")
    args = parser.parse_args()

    predict(name=args.name, n_predictions=args.n, weights_file=args.w)
