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

# Copyright 2025 Google LLC
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

"""Configuration of Simple Voice Questions (SVQ) tasks for MSEB."""

import types

config_by_task = types.MappingProxyType({
    'document_retrieval_in_lang': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'en',
            'ar',
            'bn',
            'fi',
            'id',
            'ko',
            'ru',
            'sw',
            'te',
        ),
    }),
    'document_retrieval_cross_lang': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'ar',
            'bn',
            'fi',
            'gu',
            'hi',
            'ja',
            'kn',
            'ko',
            'ml',
            'mr',
            'ru',
            'ta',
            'te',
            'ur',
        ),
    }),
    'document_retrieval_in_lang_small': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'en',
            'ar',
            'bn',
            'fi',
            'id',
            'ko',
            'ru',
            'sw',
            'te',
        ),
    }),
    'document_retrieval_cross_lang_small': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'ar',
            'bn',
            'fi',
            'gu',
            'hi',
            'ja',
            'kn',
            'ko',
            'ml',
            'mr',
            'ru',
            'ta',
            'te',
            'ur',
        ),
    }),
    'passage_retrieval_in_lang': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'en',
            'ar',
            'bn',
            'fi',
            'id',
            'ko',
            'ru',
            'sw',
            'te',
        ),
    }),
    'passage_retrieval_cross_lang': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM'),
        'languages': (
            'ar',
            'bn',
            'fi',
            'gu',
            'hi',
            'ja',
            'kn',
            'ko',
            'ml',
            'mr',
            'ru',
            'ta',
            'te',
            'ur',
        ),
    }),
    'span_retrieval_in_lang': types.MappingProxyType({
        'evaluation_metric': 'F1',
        'languages': (
            'en',
            'ar',
            'bn',
            'fi',
            'id',
            'ko',
            'ru',
            'sw',
            'te',
        ),
    }),
    'span_retrieval_cross_lang': types.MappingProxyType({
        'evaluation_metric': 'F1',
        'languages': (
            'ar',
            'bn',
            'fi',
            'ja',
            'kn',
            'ko',
            'ru',
            'te',
            'hi',
            'gu',
            'ml',
            'mr',
            'ta',
            'ur',
        ),
    }),
    'span_reasoning_in_lang': types.MappingProxyType({
        'evaluation_metric': 'F1',
        'languages': (
            'en',
            'ar',
            'bn',
            'fi',
            'id',
            'ko',
            'ru',
            'sw',
            'te',
        ),
    }),
    'span_reasoning_cross_lang': types.MappingProxyType({
        'evaluation_metric': 'F1',
        'languages': (
            'ar',
            'bn',
            'fi',
            'ja',
            'kn',
            'ko',
            'ru',
            'te',
            'hi',
            'gu',
            'ml',
            'mr',
            'ta',
            'ur',
        ),
    }),
    'query_reranking': types.MappingProxyType({
        'evaluation_metric': ('MRR', 'MRR_STD', 'EM', 'WER', 'QER'),
        'languages': (
            'ar',
            'bn',
            'en',
            'fi',
            'gu',
            'hi',
            'ja',
            'id',
            'kn',
            'ko',
            'ml',
            'mr',
            'ru',
            'sw',
            'ta',
            'te',
            'ur',
        ),
    }),
})
