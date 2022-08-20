from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor


def get_optimizer(model, mode, learning_rate, opti_name):

    update_parts = None
    if mode == 'model':
        update_parts = model.parameters()
    if mode == 'encoder':
        update_parts = model.encoder.parameters()
    if mode == 'decoder':
        # update_parts = model.decoder.parameters()
        update_parts = list(model.decoder.parameters()) + list(model.lm_head.parameters())

    if opti_name == 'adafactor':
        optimizer = Adafactor(
            update_parts,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=learning_rate
        )
        return optimizer


def learning_scheduler(train_loops, curr_ite, max_alpha=None):

    max_lr = max_alpha
    if not max_alpha:
        max_lr = 0.01
    p = 0
    cut_frac = 0.1
    ratio = 200
    cut_point = int(train_loops * cut_frac)

    if curr_ite < cut_point:
        p = curr_ite / cut_point
    if curr_ite >= cut_point:
        p = (train_loops - curr_ite) / (train_loops - cut_point)

    learning_rate = max_lr * ((p * (ratio - 1) + 1) / ratio)

    return learning_rate


def get_train_data(domain):
    _path = './json_data/aggs/{}.json'.format(domain)
    _data = load_dataset('json', data_files=_path)['train']
    return _data


def compute_bleu(metric_name, y_pred, y_true):
    metric = load_metric(metric_name)
    metric.add_batch(predictions=y_pred, references=y_true)
    report = metric.compute()
    if metric_name == 'bleu':
        return report['bleu'] * 100
    if metric_name == 'sacrebleu':
        return report['score']


def write_log(device, input_list):
    for item in input_list:
        with open('./logs/epi-NMT_res_{}.txt'.format(device), 'a') as file:
            file.write('{} '.format(item))
    with open('./logs/epi-NMT_res_{}.txt'.format(device), 'a') as file:
        file.write('\n')


def get_loader(split, batchsz, domain=None):
    if split == 'ds':
        path = './json_data/aggs/{}.json'.format(domain)
        temp_data = load_dataset('json', data_files=path)['train']
        temp_loader = DataLoader(temp_data, shuffle=True, batch_size=batchsz)
        return temp_loader

    if split == 'FT_target':
        path = './json_data/test_support/{}.json'.format(domain)
        temp_data = load_dataset('json', data_files=path)['train']
        temp_loader = DataLoader(temp_data, shuffle=True, batch_size=batchsz)
        return temp_loader

    if split == 'evaluation':
        path = './json_data/test_query/{}.json'.format(domain)
        temp_data = load_dataset('json', data_files=path)['train']
        temp_loader = DataLoader(temp_data, shuffle=False, batch_size=batchsz)
        return temp_loader

