#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import lxml.etree as ET
import os
import re as regex
import time

import config

if __name__ == "__main__":
    line_tokens = config.WikiCorpus()
    for tokens in line_tokens:
        print(token)
