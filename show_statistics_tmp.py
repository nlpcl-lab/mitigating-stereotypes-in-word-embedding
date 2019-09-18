#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import codecs, re


def load_analogy_pair(fname):
    ap_dict = defaultdict(list)
    with codecs.open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(re.split('[\r\n]+', f.read())):
            if len(line.strip()) > 0:
                tokens = re.split(r'\t', line.strip())
                for token in re.split(r' ', tokens[1]):
                    if tokens[0] == 'pairs':
                        ap_dict[tokens[0]].append(tuple(re.split(r'/', token)))
                    else:
                        ap_dict[tokens[0]].append(token)

    return ap_dict['pairs'], ap_dict['neutral_words']

