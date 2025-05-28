# Copyright 2025 The MSEB Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate IDF table from Wikipedia articles."""

# Original code:
#   https://github.com/marcocor/wikipedia-idf
#
# Modified as follows:
#  - Use spacy instead of nltk for word tokenization.
#  - Remove stemming support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import bz2
import collections
import contextlib
import json
import logging
import math
import multiprocessing
import re
import sys
import spacy
import unicodecsv as csv


DROP_TOKEN_RE = re.compile("^\\W*$")
model = None


def filter_tokens(tokens):
  for t in tokens:
    if not DROP_TOKEN_RE.match(t):
      yield t.lower()


def get_file_reader(filename):
  if filename.endswith(".bz2"):
    return bz2.BZ2File(filename)
  else:
    return open(filename)


def get_lines(input_files):
  for filename in input_files:
    with get_file_reader(filename) as f:
      for line in f:
        yield line


def process_line(line):
  article_json = json.loads(line)
  return set(filter_tokens(tokenize_line(article_json["text"])))


def tokenize_line(line):
  tokens = []
  sublines = line.split("\n")
  with model.memory_zone():
    for doc in model.pipe(sublines, disable=model.pipe_names):
      for token in doc:
        tokens.append(token.text)
  return tokens


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-i",
      "--input",
      required=True,
      nargs="+",
      action="store",
      help="Input JSON files",
  )
  parser.add_argument(
      "-s",
      "--spacy_lang",
      required=True,
      metavar="LANG",
      help="IETF language tag, such as ‘en’, of the language.",
  )
  parser.add_argument(
      "-o",
      "--output",
      metavar="OUT_BASE",
      required=True,
      help="Output CSV files base",
  )
  parser.add_argument(
      "-l",
      "--limit",
      metavar="LIMIT",
      type=int,
      help="Stop after reading LIMIT articles.",
  )
  parser.add_argument(
      "-c", "--cpus", default=1, type=int, help="Number of CPUs to employ."
  )
  args = parser.parse_args()

  global model
  model = spacy.blank(args.spacy_lang)
  tokens_c = collections.Counter()
  articles = 0
  total_tokens = 0

  with contextlib.closing(multiprocessing.Pool(processes=args.cpus)) as pool:
    for tokens in pool.imap_unordered(process_line, get_lines(args.input)):
      if not tokens:
        continue
      tokens_c.update(tokens)
      total_tokens += len(tokens)
      articles += 1
      if not (articles % 100):
        logging.info(
            "Done %d articles; %d total tokens; %d unique tokens.",
            articles,
            total_tokens,
            len(tokens_c),
        )
      if articles == args.limit:
        break

  with open("{}_{}".format(args.output, "terms.csv"), "wb") as o:
    w = csv.writer(o, encoding="utf-8")
    w.writerow(("token", "frequency", "total", "idf"))
    for token, freq in tokens_c.most_common():
      w.writerow([token, freq, articles, math.log(float(articles) / freq)])


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  sys.exit(main())
