# Copyright 2026 The MSEB Authors.
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

"""Constants for MSEB tracks."""

SVQ_CROSS_LANG_LOCALES = tuple(
    sorted(
        set([
            'ArEg',
            'ArXGulf',
            'ArXLevant',
            'ArXMaghrebi',
            'BnBd',
            'BnIn',
            'FiFi',
            'GuIn',
            'HiIn',
            'JaJp',
            'KnIn',
            'KoKr',
            'MlIn',
            'MrIn',
            'RuRu',
            'TaIn',
            'TeIn',
            'UrIn',
            'UrPk',
        ])
    )
)

SVQ_IN_LANG_LOCALES = tuple(
    sorted(
        set([
            'ArEg',
            'ArXGulf',
            'ArXLevant',
            'ArXMaghrebi',
            'BnBd',
            'BnIn',
            'EnAu',
            'EnGb',
            'EnIn',
            'EnPh',
            'EnUs',
            'FiFi',
            'IdId',
            'KoKr',
            'RuRu',
            'Sw',
            'TeIn',
        ])
    )
)

SVQ_LOCALES = tuple(sorted(set(SVQ_CROSS_LANG_LOCALES + SVQ_IN_LANG_LOCALES)))

SPEECH_MASSIVE_LOCALES = tuple(
    sorted(
        set([
            'ArSa',
            'DeDe',
            'EsEs',
            'FrFr',
            'HuHu',
            'KoKr',
            'NlNl',
            'PlPl',
            'PtPt',
            'RuRu',
            'TrTr',
            'ViVn',
        ])
    )
)
