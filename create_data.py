import os
import pandas as pd
from transformers import T5TokenizerFast


class SplitData:

    def text2json(self, domain):
        def reset_df_index(dataframe):
            dataframe = dataframe.reset_index(drop=True)
            return dataframe

        source = path + '{}_{}.txt'.format(domain, src_language)
        target = path + '{}_{}.txt'.format(domain, tgt_language)

        src_text = open(source, encoding='utf8').read().split('\n')
        tgt_text = open(target, encoding='utf8').read().split('\n')

        raw_data = {src_language: [line for line in src_text],
                    tgt_language: [line for line in tgt_text],
                    'domain_name': [domain] * len(src_text)}

        df = pd.DataFrame(raw_data, columns=[src_language, tgt_language, 'domain_name'])
        drop_index = []

        for i in range(len(df)):
            en = tokenizer(df.loc[i]['en'])
            de = tokenizer(df.loc[i]['de'])
            en_size = len(en['input_ids'])
            de_size = len(de['input_ids'])
            ratio = en_size / de_size
            if en_size < 5 or en_size > 175 or de_size < 5 or de_size > 175 or ratio < 0.66 or ratio > 1.5:
                drop_index.append(i)
        df = df.drop(drop_index)
        df = reset_df_index(df)
        print('{} Done Transfer to JSON.'.format(domain))
        return df

    def split_by_tokens(self, df, mode, domain):
        tokens = 0
        for i in range(len(df)):
            sentence1 = tokenizer(df.loc[i]['en'])
            tokens += len(sentence1['input_ids'])
            if tokens > toks['{}'.format(mode)]:
                print('number of sentences in {}: {}, tokens: {}'.format(mode, i + 1, tokens))
                df.loc[:i].to_json('./json_data/{}/{}.json'.format(mode, domain), orient='records', force_ascii=False,
                                   lines=True)
                df = df.loc[i + 1:].reset_index(drop=True)
                break
        return df

    def split_train_test(self, df, is_seen, domain):
        df = df.sample(frac=1).reset_index(drop=True)
        if is_seen:
            df = self.split_by_tokens(df, 'train_support', domain)
            df = df.sample(frac=1).reset_index(drop=True)
            df = self.split_by_tokens(df, 'train_query', domain)
            df = df.sample(frac=1).reset_index(drop=True)
            df = self.split_by_tokens(df, 'test_support', domain)
            df = df.sample(frac=1).reset_index(drop=True)
            self.split_by_tokens(df, 'test_query', domain)
            print('\n')
        else:
            df = self.split_by_tokens(df, 'test_support', domain)
            df = df.sample(frac=1).reset_index(drop=True)
            self.split_by_tokens(df, 'test_query', domain)
        print('\n')

    def workflow(self):
        # create seen domain data
        for domain in seen:
            data = self.text2json(domain)
            print('Seen Domain: {}, Total Number of Sentences: {}'.format(domain, len(data)), '\n')
            self.split_train_test(data, True, domain)

        # create unseen domain data
        # for domain in unseen:
        #     data = self.text2json(domain)
        #     print('Unseen Domain: {}, Total Number of Sentences: {}'.format(domain, len(data)), '\n')
        #     self.split_train_test(data, False, domain)

        # create domain aggregation data
        df = pd.DataFrame([])
        for domain in seen:
            domain_df = pd.DataFrame([])
            for mode in ['train_support', 'train_query']:
                path = './json_data/{}/{}.json'.format(mode, domain)
                curr_df = pd.read_json(path, orient='records', encoding='utf-8', lines=True)
                df = df.append(curr_df)
                domain_df = domain_df.append(curr_df)
            print(domain, len(domain_df))
            domain_df.to_json('./json_data/aggs/{}.json'.format(domain), orient='records', force_ascii=False, lines=True)
        print(len(df))
        df.to_json('./json_data/aggs/train.json', orient='records', force_ascii=False, lines=True)


if __name__ == '__main__':
    src_language = 'en'
    tgt_language = 'de'
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    unseen = ['covid', 'bible', 'books',  'ECB', 'TED2013']
    seen = ['EMEA', 'GlobalVoices', 'KDE4', 'WMT-News', 'JRC-Acquis']

    toks = {'train_support': 320000,
            'train_query': 640000,
            'test_support': 10000,
            'test_query': 20000}

    path = './txt_data/'
    json_path = './json_data/'
    modes = ['aggs', 'train_support', 'train_query', 'test_support', 'test_query']
    if not os.path.exists(json_path):
        os.mkdir(json_path)
    for mode in modes:
        if not os.path.exists(json_path + mode):
            os.mkdir(json_path + mode)
    SplitData().workflow()

