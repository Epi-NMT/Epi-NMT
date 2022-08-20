from transformers import T5Tokenizer, T5ForConditionalGeneration
from epi_utils import *
import torch
import argparse
from datetime import datetime
import random
import logging
import warnings
import copy
warnings.filterwarnings("ignore")
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


class EpisodicNMT:
    def __init__(self):

        self.train_data = {}
        self.eva_loader = {}
        self.init_datasets()

        self.scaler = torch.cuda.amp.GradScaler()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

        self.domain_weights = {}
        self.init_model_states()

    # Initialize the domain data and the domain
    def init_datasets(self):
        self.train_data['agg'] = get_train_data('train')
        for domain in seen_domains:
            self.train_data[domain] = get_train_data(domain)
        for domain in domains:
            self.eva_loader[domain] = {}
            self.eva_loader[domain]['FT'] = get_loader('FT_target', args().batchsz, domain)
            self.eva_loader[domain]['test'] = get_loader('evaluation', args().batchsz, domain)
        print(
            '****************************',
            'Data Initialization Done !',
            '****************************',
        )

    # Initialize and store the weights of each domain model
    def init_model_states(self):
        self.domain_weights['agg'] = {}
        for _domain in seen_domains:
            self.domain_weights[_domain] = {}
        for domain in self.domain_weights.keys():
            self.domain_weights[domain]['encoder'] = copy.deepcopy(self.model.encoder).state_dict()
            self.domain_weights[domain]['decoder'] = copy.deepcopy(self.model.decoder).state_dict()
            self.domain_weights[domain]['lm_head'] = copy.deepcopy(self.model.lm_head).state_dict()
        print(
            '****************************',
            'Model State Initialization Done !',
            '****************************',
        )

    # Load the saved model weights
    def load_model_state(self, domain):
        self.model.encoder.load_state_dict(self.domain_weights[domain]['encoder'])
        self.model.decoder.load_state_dict(self.domain_weights[domain]['decoder'])
        self.model.lm_head.load_state_dict(self.domain_weights[domain]['lm_head'])

    # Save the trained model weight to the dictionary
    def save_model_state(self, domain):
        self.domain_weights[domain]['encoder'] = copy.deepcopy(self.model.encoder).state_dict()
        self.domain_weights[domain]['decoder'] = copy.deepcopy(self.model.decoder).state_dict()
        self.domain_weights[domain]['lm_head'] = copy.deepcopy(self.model.lm_head).state_dict()

    # Load the saved model encoder / decoder + lm_head
    def load_encoder_decoder(self, is_encoder, domain):
        self.load_model_state('agg')
        if is_encoder:
            self.model.encoder.load_state_dict(self.domain_weights[domain]['encoder'])
        if not is_encoder:
            self.model.decoder.load_state_dict(self.domain_weights[domain]['decoder'])
            # self.model.lm_head.load_state_dict(self.domain_weights[domain]['lm_head'])

    # Tokenize the batch corpus
    def encode_corpus(self, is_src, batch):
        # Prepare and tokenize source and target sentences
        _prefix = "Translate English to German: "
        _encoded_sentences = self.tokenizer(
            [_prefix + line for line in batch['en']] if is_src
            else [line for line in batch['de']],
            max_length=175,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True).input_ids.to(device)
        return _encoded_sentences

    # Randomly sample the training sentences
    def sample_train_batch(self, domain):
        _curr_data = self.train_data[domain]
        _size = len(_curr_data)
        sample_list = random.sample(range(_size), args().ds_batchsz)
        _batch = _curr_data[sample_list]
        return _batch

    # Test the SacreBleu score on test_query
    def evaluation(self, domain):

        self.model.eval()
        y_true = []
        y_pred = []
        eva_loader = self.eva_loader[domain]['test']

        for i, batch in enumerate(eva_loader):

            # Prepare and tokenize the source sentences
            encoded_src = self.encode_corpus(is_src=True, batch=batch)

            # Translate and decode the inputs
            outputs = self.model.generate(encoded_src, max_length=175, num_beams=5)
            batch_pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Concatenate the translated and reference sentences
            for sentence in batch['de']:
                # if args().metric == 'bleu':
                #     sentence = self.tokenizer.tokenize(sentence)
                y_true.append([sentence])
            for sentence in batch_pred:
                # if args().metric == 'bleu':
                #     sentence = self.tokenizer.tokenize(sentence)
                y_pred.append(sentence)

        bleu = compute_bleu(args().metric, y_pred, y_true)
        print(domain, bleu)

        return bleu

    # Simple fine-tune on test_support set
    def fine_tune_target(self, domain):

        self.model.train()
        ft_loader = self.eva_loader[domain]['FT']
        optimizer = get_optimizer(self.model, 'model', args().tgt_ft_lr, args().opti_name)
        best_bleu = 0

        for epoch in range(args().tgt_ft_epochs):

            loss = 0

            for i, batch in enumerate(ft_loader):
                # Prepare the source and target data corpus
                encoded_src = self.encode_corpus(is_src=True, batch=batch)
                encoded_tgt = self.encode_corpus(is_src=False, batch=batch)
                prediction = self.model(input_ids=encoded_src, labels=encoded_tgt)

                optimizer.zero_grad()

                # Compute the loss
                train_loss = prediction.loss
                loss += train_loss.item()

                # Update model parameters
                self.scaler.scale(train_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            bleu = self.evaluation(domain)
            best_bleu = max(best_bleu, bleu)
        now = datetime.now()
        print(
            'Time: {}:{},'.format(now.hour, now.minute),
            'Domain {} best Bleu: {:.2f}'.format(domain, best_bleu)
        )
        return best_bleu

    # Domain specific training branch
    def ds_train(self, ds_domain, ds_lr):

        self.load_model_state(ds_domain)
        self.model.train()

        ds_optimizer = get_optimizer(self.model, 'model', ds_lr, args().opti_name)

        # Prepare the source and target data corpus
        batch = self.sample_train_batch(ds_domain)
        encoded_src = self.encode_corpus(is_src=True, batch=batch)
        encoded_tgt = self.encode_corpus(is_src=False, batch=batch)

        ds_optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            ds_prediction = self.model(input_ids=encoded_src, labels=encoded_tgt)
            ds_loss = ds_prediction.loss

        self.scaler.scale(ds_loss).backward()
        self.scaler.step(ds_optimizer)
        self.scaler.update()

        self.save_model_state(domain=ds_domain)

    # Episodic training by aggregating all the loss and update once
    def epis_agg_train(self, batch, expose_domain, agg_lr):

        self.model.train()
        optimizer = get_optimizer(self.model, 'model', agg_lr, args().opti_name)

        # Prepare the source and target data corpus for the selected domain
        encoded_src = self.encode_corpus(is_src=True, batch=batch)
        encoded_tgt = self.encode_corpus(is_src=False, batch=batch)

        batch_copy = copy.deepcopy(batch)
        delete_list = []

        # If the data domain is not the expose domain, delete the corpus
        for i in range(len(batch['en'])):
            if batch['domain_name'][i] == expose_domain:
                delete_list.append(i)
                batch_copy['en'].remove(batch['en'][i])
                batch_copy['de'].remove(batch['de'][i])
                batch_copy['domain_name'].remove(batch['domain_name'][i])

        epi_encoded_src = self.encode_corpus(is_src=True, batch=batch_copy)
        epi_encoded_tgt = self.encode_corpus(is_src=False, batch=batch_copy)

        optimizer.zero_grad()

        # Episodic encoder training
        self.load_encoder_decoder(True, expose_domain)
        with torch.cuda.amp.autocast():
            spec_encoder_prediction = self.model(input_ids=epi_encoded_src, labels=epi_encoded_tgt)
            spec_en_loss = spec_encoder_prediction.loss

        optimizer.zero_grad()

        # Episodic decoder training
        self.load_encoder_decoder(False, expose_domain)
        with torch.cuda.amp.autocast():
            spec_decoder_prediction = self.model(input_ids=epi_encoded_src, labels=epi_encoded_tgt)
            spec_de_loss = spec_decoder_prediction.loss

        # print(
        #     'Spec Encoder Loss: {:.5f} |'.format(spec_en_loss),
        #     'Spec Decoder Loss: {:.5f}'.format(spec_de_loss)
        # )

        epi_loss = spec_en_loss.item() * args().alpha + spec_de_loss.item() * (1 - args().alpha)

        optimizer.zero_grad()

        self.load_model_state('agg')
        with torch.cuda.amp.autocast():
            agg_prediction = self.model(input_ids=encoded_src, labels=encoded_tgt)
            agg_loss = agg_prediction.loss

        batch_loss = agg_loss + epi_loss

        # print(
        #     'Agg Loss: {:.5f} |'.format(agg_loss.item()),
        #     'Batch Loss: {:.5f}'.format(batch_loss.item())
        # )

        self.scaler.scale(batch_loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        self.save_model_state('agg')

        return batch_loss

    def warm_up(self):

        now = datetime.now()
        print(
            '****************************',
            'Start Warm Up: {}:{}:{}'.format(now.hour, now.minute, now.second),
            '****************************'
        )

        # warm up the domain specific branches
        for domain in seen_domains:
            self.load_model_state(domain)
            for ite in range(args().warmup_iterations):
                unit_lr = args().ds_lr / args().warmup_iterations
                self.ds_train(domain, (ite + 1) * unit_lr)
            print(
                'Domain {} Warm up Done !'.format(domain)
            )

        # warm up the agg branch
        self.load_model_state('agg')
        for ite in range(args().warmup_iterations):
            unit_lr = args().epi_lr / args().warmup_iterations
            self.ds_train('agg', (ite + 1) * unit_lr)

        print(
            '****************************',
            'Done Warm Up!',
            '****************************',
            '\n'
        )

    # Training work flow of our episodic-NMT
    def episodic_workflow(self):

        now = datetime.now()
        print(
            '****************************',
            'Start Training: {}:{}'.format(now.hour, now.minute),
            '****************************',
            '\n'
        )
        # Warm up
        self.warm_up()

        # best_res = 0 # Only use when you want to save the model
        for epoch in range(args().train_epochs):
            # Domain Aggregation model data loader
            train_loader = get_loader('ds', batchsz=args().batchsz, domain='train')
            epoch_loss = 0
            # self.train_data['agg'] = self.train_data['agg'].shuffle(seed=epoch)
            for ite, batch in enumerate(train_loader):
                # For each batch, random sample 2 seen domains.
                # Select the expose domain as the specific branch.
                # When the input domain is trigger domain, we apply episodic training.
                candidates = list(seen_domains)
                trigger_domain = random.sample(candidates, 1)[0]
                candidates.remove(trigger_domain)
                exposed_domain = random.sample(candidates, 1)[0]

                # Domain Specific Training
                ds_lr = args().ds_lr
                for ds_domain in seen_domains:
                    self.ds_train(ds_domain, ds_lr)

                # Episodic Training
                agg_lr = args().epi_lr
                batch_loss = self.epis_agg_train(batch, exposed_domain, agg_lr)
                epoch_loss += batch_loss

            now = datetime.now()
            print(
                '****************************',
                'Time: {}:{},'.format(now.hour, now.minute),
                'Epoch: {}/{},'.format(epoch + 1, args().train_epochs),
                'Epoch Average Loss: {:.5f}'.format(epoch_loss / len(train_loader)),
                '****************************',
            )

            # We evaluate the performance every k epochs
            if (epoch + 1) % args().test_every == 0:

                bfr_res = []
                afr_res = []
                for test_domain in domains:
                    self.load_model_state('agg')
                    bleu = self.evaluation(test_domain)
                    bfr_res.append(bleu)
                    print(
                        'Epoch: {}/{}'.format(epoch + 1, args().train_epochs),
                        'Domain: {}'.format(test_domain),
                        'Bleu W/O Fine-Tune: {}'.format(bleu)
                    )
                    bleu = self.fine_tune_target(test_domain)
                    afr_res.append(bleu)
#                 write_log(args().gpu, bfr_res)
#                 write_log(args().gpu, afr_res)

        # Save Model
        #         if sum(afr_res) / len(afr_res) > best_res:
        #             self.load_model_state('agg')
        #             best_res = sum(afr_res) / len(afr_res)
        #             torch.save(self.model.state_dict(), './models/epi-NMT_{}.pt'.format(args().gpu))
        #
        # for ds_domain in seen_domains:
        #     self.load_model_state(ds_domain)
        #     torch.save(self.model.state_dict(), './models/{}_{}.pt'.format(ds_domain, args().gpu))


# Arguments
def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--batchsz", type=int, default=8,
                                  help="batch size")
    train_arg_parser.add_argument("--ds_batchsz", type=int, default=8,
                                  help="batch size")
    train_arg_parser.add_argument("--opti_name", type=str, default='adafactor',
                                  help="Adam / adafactor")
    train_arg_parser.add_argument("--metric", type=str, default='sacrebleu',
                                  help="sacrebleu / bleu")
    train_arg_parser.add_argument("--tgt_ft_epochs", type=int, default=15,
                                  help="sacrebleu / bleu")
    train_arg_parser.add_argument("--tgt_ft_lr", type=float, default=2e-5,
                                  help=" initial learning rate")
    train_arg_parser.add_argument("--ds_lr", type=float, default=5e-5,
                                  help=" Domain specific learning rate ")
    train_arg_parser.add_argument("--alpha", type=float, default=0.5,
                                  help=" Domain specific learning rate ")
    train_arg_parser.add_argument("--epi_lr", type=float, default=3e-5,
                                  help=" episodic learning rate when aggregate all losses ")
    train_arg_parser.add_argument("--train_epochs", type=int, default=12,
                                  help=" Total Epochs ")
    train_arg_parser.add_argument("--warmup_iterations", type=int, default=5,
                                  help=" warm up iterations ")
    train_arg_parser.add_argument("--test_every", type=int, default=3,
                                  help=" Test final performance for every {X} epochs ")
    return train_arg_parser.parse_args()


if __name__ == '__main__':
    device = torch.device(
        'cuda:{}'.format(args().gpu) if torch.cuda.is_available() else 'cpu'
    )
    print(
        args(),
        '\n'
        '****************************',
        'Using Device: {}'.format(device),
        '****************************',
    )

    domains = ['covid', 'bible', 'books', 'ECB', 'TED2013',
               'EMEA', 'Tanzil', 'KDE4', 'OpenSub16', 'JRC-Acquis']

#     write_log(args().gpu, [args().ds_lr, args().epi_lr, args().tgt_ft_lr])
#     write_log(args().gpu, domains)

    seen_domains = domains[5:]
    EpisodicNMT().warm_up()
    EpisodicNMT().episodic_workflow()
