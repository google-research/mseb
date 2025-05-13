# Copyright 2024 The MSEB Authors.
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

"""Configuration of Simple QA tasks for MSEB."""

from flax import core

config_by_task = core.FrozenDict({
    'document_retrieval_in_lang': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'document_retrieval_cross_lang': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'document_retrieval_in_lang_small': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'document_retrieval_cross_lang_small': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'passage_retrieval_in_lang': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'passage_retrieval_cross_lang': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
    'span_retrieval_in_lang': {
        'evaluation_metric': 'F1',
        'languages': (
            'star',
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
    },
    'span_retrieval_cross_lang': {
        'evaluation_metric': 'F1',
        'languages': (
            'star',
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
    },
    'span_reasoning_in_lang': {
        'evaluation_metric': 'F1',
        'languages': (
            'star',
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
    },
    'span_reasoning_cross_lang': {
        'evaluation_metric': 'F1',
        'languages': (
            'star',
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
    },
    'query_reranking': {
        'evaluation_metric': ('MRR', 'EM'),
        'languages': (
            'star',
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
    },
})
