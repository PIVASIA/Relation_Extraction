# coding=utf-8
"""
RoBERTa-EM + Entity Start Model
"""
import argparse

from eval.data_loader import load_and_cache_examples
from eval.eval_es import Trainer
from eval.utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)
    train_dataset = []
    test_dataset = load_and_cache_examples(args, args.eval_data_file, tokenizer,for_test=True)
    
    trainer = Trainer(args, tokenizer, train_dataset=train_dataset, test_dataset=test_dataset)
    
    trainer.load_model()
    
    return trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_data_file", required=True,
        type=str,
        help="Path to validation data (a text file).",
    )
    parser.add_argument(
        "--save_output", required=True,
        type=str,
        help="Path to save data (a text file).",
    )
    parser.add_argument(
        "--id2label", type=str, required=True,
        help="Path to id2label file")
    parser.add_argument("--model_type", type=str, choices=["es", "all"],
                        required=True, help="Model type (BertEntityStarts or BertConcatAll)")
    parser.add_argument("--model_dir", type=str, default="./models", help="Path to save model checkpoints")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="vinai/phobert-base",
        help="Model name or path to BERT model",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Whether to lower case texts")
    parser.add_argument("--seed", type=int, default=77, help="random seed for initialization")

    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add </s> token at the end of the sentence",
    )
    args = parser.parse_args()
    
    outputs = main(args)