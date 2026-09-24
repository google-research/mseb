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

"""Retrieval track."""

import types

from mseb import track
from mseb.tracks import constants

_DEBUG_RETRIEVAL = tuple(
    sorted(
        set([
            'SVQEnUsPassageInLangRetrievalDebug',
            'SVQFiFiPassageInLangRetrievalDebug',
            'SpokenCocoEnImageRetrievalDebug',
            'SVQEnUsSpanInLangReasoningDebug',
            'SVQFiFiSpanInLangReasoningDebug',
            'SVQEnUsQueryRerankingDebug',
            'SVQFiFiQueryRerankingDebug',
            'SVQEnUsSalientTermRerankingDebug',
            'SVQFiFiSalientTermRerankingDebug',
            'SVQClusteringEnUsDebug',
            'SVQClusteringFiFiDebug',
            'FSD50KTestClusteringDebug',
            'SpeechMassiveDeDeIntentClassificationDebug',
            'SpeechMassiveFrFrIntentClassificationDebug',
            'FSD50KTestClassificationDebug',
        ])
    )
)

_COMPACT_RETRIEVAL = tuple(
    sorted(
        set(
            [
                f'SVQ{locale}PassageCrossLangRetrievalCompact'
                for locale in constants.SVQ_CROSS_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}PassageInLangRetrievalCompact'
                for locale in constants.SVQ_IN_LANG_LOCALES
            ]
            + [
                'SpokenCocoEnImageRetrieval',
                'SpokenCocoEnSpeechRetrieval',
            ]
            + [
                f'SVQ{locale}SpanCrossLangReasoningCompact'
                for locale in constants.SVQ_CROSS_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}SpanInLangReasoningCompact'
                for locale in constants.SVQ_IN_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}QueryRerankingCompact'
                for locale in constants.SVQ_LOCALES
            ]
            + [
                f'SVQ{locale}SalientTermRerankingCompact'
                for locale in constants.SVQ_LOCALES
            ]
            + [
                f'SVQClustering{locale}Compact'
                for locale in constants.SVQ_LOCALES
            ]
            + [
                'FSD50KTestClustering',
                'BirdsetClusteringHSN',
                'BirdsetClusteringNBP',
                'BirdsetClusteringPOW',
            ]
            + [
                f'SpeechMassive{locale}IntentClassificationCompact'
                for locale in constants.SPEECH_MASSIVE_LOCALES
            ]
            + [
                'FSD50KTestClassification',
                'BirdsetHSNClassification',
                'BirdsetNBPClassification',
                'BirdsetPOWClassification',
            ]
        )
    )
)

_FULL_RETRIEVAL = tuple(
    sorted(
        set(
            [
                f'SVQ{locale}PassageCrossLangRetrieval'
                for locale in constants.SVQ_CROSS_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}PassageInLangRetrieval'
                for locale in constants.SVQ_IN_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}DocumentCrossLangRetrievalSmallIndex'
                for locale in constants.SVQ_CROSS_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}DocumentInLangRetrievalSmallIndex'
                for locale in constants.SVQ_IN_LANG_LOCALES
            ]
            + [
                'SpokenCocoEnImageRetrieval',
                'SpokenCocoEnSpeechRetrieval',
            ]
            + [
                f'SVQ{locale}SpanCrossLangReasoning'
                for locale in constants.SVQ_CROSS_LANG_LOCALES
            ]
            + [
                f'SVQ{locale}SpanInLangReasoning'
                for locale in constants.SVQ_IN_LANG_LOCALES
            ]
            + [f'SVQ{locale}QueryReranking' for locale in constants.SVQ_LOCALES]
            + [
                f'SVQ{locale}SalientTermReranking'
                for locale in constants.SVQ_LOCALES
            ]
            + [f'SVQClustering{locale}' for locale in constants.SVQ_LOCALES]
            + [
                'FSD50KTestClustering',
                'BirdsetClusteringHSN',
                'BirdsetClusteringNBP',
                'BirdsetClusteringPOW',
            ]
            + [
                f'SVQ{locale}SpeakerGenderClassification'
                for locale in constants.SVQ_LOCALES
            ]
            + [
                f'SpeechMassive{locale}IntentClassification'
                for locale in constants.SPEECH_MASSIVE_LOCALES
            ]
            + [
                f'SpeechMassive{locale}SpeakerGenderClassification'
                for locale in constants.SPEECH_MASSIVE_LOCALES
            ]
            + [
                'FSD50KTestClassification',
                'BirdsetHSNClassification',
                'BirdsetNBPClassification',
                'BirdsetPOWClassification',
            ]
        )
    )
)

RETRIEVAL = track.Track(
    name='retrieval',
    description=(
        'Retrieval tasks (embedding models): classification, clustering,'
        ' reasoning, reranking, retrieval.'
    ),
    tasks_by_size=types.MappingProxyType({
        track.Size.DEBUG: _DEBUG_RETRIEVAL,
        track.Size.COMPACT: _COMPACT_RETRIEVAL,
        track.Size.FULL: _FULL_RETRIEVAL,
    }),
)
