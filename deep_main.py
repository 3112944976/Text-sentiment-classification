from functools import partial
from torch.utils.data import DataLoader
from tool import DataLoad, collate_fn_dl, load_word2id, play_vocab
from DLModels.deep import DeepModel


VOCAB_SIZE = 28663
SENT_MAX_LEN = 128


def main():
    word2id = load_word2id(length=VOCAB_SIZE)
    train_loader_dl = DataLoader(
        dataset=DataLoad("train"),
        batch_size=64,
        collate_fn=partial(collate_fn_dl, word2id)
    )
    test_loader_dl = DataLoader(
        dataset=DataLoad("test"),
        batch_size=64,
        collate_fn=partial(collate_fn_dl, word2id)
    )
    vocab_size = len(word2id)
    print("CNN模型训练与评估...")
    cnn_model = DeepModel(vocab_size, None, method="cnn")
    cnn_model.train_and_eval(train_loader_dl, test_loader_dl)


if __name__ == "__main__":
    # play_vocab("./datasets/vocab.csv")
    main()
