import os
import argparse
from utils.preprocess_func import Preprocesser, Read_Corpus

parser = argparse.ArgumentParser()

parser.add_argument(
    '-train',
    type=str,
    nargs='+',
    default=None,
    required=True,
    help='source train data path')

parser.add_argument(
    '-V_min_freq',
    nargs='+',
    type=int,
    help='minimum frequency of vocabulary words')

parser.add_argument(
    '-V',
    nargs='+',
    type=int,
    help='vocabulary size')

parser.add_argument(
    '-V_files',
    nargs='+',
    type=str,
    help='vocabulary file')

parser.add_argument(
    '-save_name',
    default="default",
    type=str,
    help='data name')

parser.add_argument(
    '-save_dir',
    default="",
    type=str,
    required=True,
    help='data directory')

parser.add_argument(
    '-output_vocab',
    action='store_true',
    help='output vocabulary txt file'
)
parser.add_argument(
    '-comparable',
    action='store_true',
    help='use comparable corpora'
)

opt = parser.parse_args()

if __name__ == '__main__':

    train_corpus = Read_Corpus(opt.train)
    if (not os.path.isdir(opt.save_dir)):
        os.mkdir(opt.save_dir)

    with open(opt.save_dir + opt.save_name + "_inputs.txt", "w") as f:
        f.write("save_name" + ": " + opt.save_dir + opt.save_name + "\n")
        for lang in range(len(opt.train)):
            f.write("train_files" + str(lang) + ": " + opt.train[lang] + "\n")

    Preprocesser = Preprocesser()
    Preprocesser.load_vocab(opt, train_corpus, len(opt.train))
    Preprocesser.dataset.lines_id_input, Preprocesser.dataset.lines_id_output, Preprocesser.dataset.lengths \
        = Preprocesser.load_corpus(train_corpus)
    Preprocesser.oversampling()
    Preprocesser.shuffle_train_data(opt.comparable)
    Preprocesser.save_files(opt.save_name, opt.save_dir, opt.output_vocab)
