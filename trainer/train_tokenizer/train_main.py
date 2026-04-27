from .train_tokenizer import train_tokenizer
from .eval_tokenizer import eval_tokenizer

if __name__ == '__main__':
    DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
    TOKENIZER_DIR = '../model_learn_tokenizer/'
    VOCAB_SIZE = 6400
    SPECIAL_TOKENS_NUM = 36
    
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE,SPECIAL_TOKENS_NUM)
    eval_tokenizer(TOKENIZER_DIR)