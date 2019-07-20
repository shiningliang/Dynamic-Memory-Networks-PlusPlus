from babi_loader import BabiDataset
import pickle as pkl
import os


def process_babi(path):
    dset_dict, vocab_dict = {}, {}
    for task_id in range(1, 21):
        dset_dict[task_id] = BabiDataset(task_id)
        vocab_dict[task_id] = len(dset_dict[task_id].QA.VOCAB)

    dset_path = os.path.join(path, 'dset.pkl')
    with open(dset_path, 'wb') as f:
        pkl.dump(dset_dict, f)
    f.close()
    vocab_path = os.path.join(path, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pkl.dump(vocab_dict, f)
    f.close()