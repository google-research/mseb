"""HuthLab Encoding Model Scaling Laws dataset.

Loads fMRI responses and stimulus metadata from the HuthLab
encoding-model-scaling-laws dataset:

  Antonello, Turek, Vo, and Huth (2024). "Scaling in Speech and Language Models:
  A Path to Human-Level Performance?" arXiv:2401.10150.

  Data: https://utexas.box.com/v/EncodingModelScalingLaws

The data consists of:
  - fMRI BOLD responses from subjects listening to narrative stories.
  - Precomputed model features (downsampled to TR resolution).
  - Audio stimuli as WAV files.

Directory structure expected under `base_path`:
  base_path/
    responses/
      {subject}_test_stories_avg_resp.jbl   # (T_test, V) test fMRI responses
    stimuli/
    <-- story_audio/
      {story_name}.wav                  # Audio stimuli
"""

import io
import os
from typing import Sequence

from absl import flags
from absl import logging
from etils import epath
import joblib
import numpy as np

_HUTHLAB_DATA_PATH = flags.DEFINE_string(
    'huthlab_data_path',
    None,
    'Path to the HuthLab encoding-model-scaling-laws data directory.',
)

# Default parameters from the paper and tutorial notebook.
DEFAULT_TR_DURATION = 2.0  # seconds
DEFAULT_DELAYS = (1, 2, 3, 4)  # FIR delays in TRs (2s, 4s, 6s, 8s)

# Subjects available in the dataset.
SUBJECTS = (
    'S01',
    'S02',
    'S03',
)

TRAIN_STORIES = {
    'S01': (
        'itsabox',
        'odetostepfather',
        'inamoment',
        'hangtime',
        'ifthishaircouldtalk',
        'goingthelibertyway',
        'golfclubbing',
        'thetriangleshirtwaistconnection',
        'igrewupinthewestborobaptistchurch',
        'tetris',
        'becomingindian',
        'canplanetearthfeedtenbillionpeoplepart1',
        'thetiniestbouquet',
        'swimmingwithastronauts',
        'lifereimagined',
        'forgettingfear',
        'stumblinginthedark',
        'backsideofthestorm',
        'food',
        'theclosetthatateeverything',
        'notontheusualtour',
        'exorcism',
        'adventuresinsayingyes',
        'thefreedomridersandme',
        'cocoonoflove',
        'waitingtogo',
        'thepostmanalwayscalls',
        'googlingstrangersandkentuckybluegrass',
        'mayorofthefreaks',
        'learninghumanityfromdogs',
        'shoppinginchina',
        'souls',
        'cautioneating',
        'comingofageondeathrow',
        'breakingupintheageofgoogle',
        'gpsformylostidentity',
        'eyespy',
        'treasureisland',
        'thesurprisingthingilearnedsailingsoloaroundtheworld',
        'theadvancedbeginner',
        'goldiethegoldfish',
        'life',
        'thumbsup',
        'seedpotatoesofleningrad',
        'theshower',
        'adollshouse',
        'canplanetearthfeedtenbillionpeoplepart2',
        'sloth',
        'howtodraw',
        'quietfire',
        'metsmagic',
        'penpal',
        'thecurse',
        'canadageeseandddp',
        'thatthingonmyarm',
        'buck',
        'wildwomenanddancingqueens',
        'againstthewind',
        'indianapolis',
        'alternateithicatom',
        'bluehope',
        'kiksuya',
        'afatherscover',
        'haveyoumethimyet',
        'firetestforlove',
        'catfishingstrangerstofindmyself',
        'christmas1940',
        'tildeath',
        'lifeanddeathontheoregontrail',
        'vixenandtheussr',
        'undertheinfluence',
        'beneaththemushroomcloud',
        'jugglingandjesus',
        'superheroesjustforeachother',
        'sweetaspie',
        'naked',
        'singlewomanseekingmanwich',
        'avatar',
        'whenmothersbullyback',
        'myfathershands',
        'reachingoutbetweenthebars',
        'theinterview',
        'stagefright',
        'legacy',
        'canplanetearthfeedtenbillionpeoplepart3',
        'listo',
        'gangstersandcookies',
        'birthofanation',
        'mybackseatviewofagreatromance',
        'lawsthatchokecreativity',
        'threemonths',
        'whyimustspeakoutaboutclimatechange',
        'leavingbaghdad',
    ),
    'S02': (
        'itsabox',
        'odetostepfather',
        'inamoment',
        'afearstrippedbare',
        'findingmyownrescuer',
        'hangtime',
        'ifthishaircouldtalk',
        'goingthelibertyway',
        'golfclubbing',
        'thetriangleshirtwaistconnection',
        'igrewupinthewestborobaptistchurch',
        'tetris',
        'becomingindian',
        'canplanetearthfeedtenbillionpeoplepart1',
        'thetiniestbouquet',
        'swimmingwithastronauts',
        'lifereimagined',
        'forgettingfear',
        'stumblinginthedark',
        'backsideofthestorm',
        'food',
        'theclosetthatateeverything',
        'escapingfromadirediagnosis',
        'notontheusualtour',
        'exorcism',
        'adventuresinsayingyes',
        'thefreedomridersandme',
        'cocoonoflove',
        'waitingtogo',
        'thepostmanalwayscalls',
        'googlingstrangersandkentuckybluegrass',
        'mayorofthefreaks',
        'learninghumanityfromdogs',
        'shoppinginchina',
        'souls',
        'cautioneating',
        'comingofageondeathrow',
        'breakingupintheageofgoogle',
        'gpsformylostidentity',
        'marryamanwholoveshismother',
        'eyespy',
        'treasureisland',
        'thesurprisingthingilearnedsailingsoloaroundtheworld',
        'theadvancedbeginner',
        'goldiethegoldfish',
        'life',
        'thumbsup',
        'seedpotatoesofleningrad',
        'theshower',
        'adollshouse',
        'canplanetearthfeedtenbillionpeoplepart2',
        'sloth',
        'howtodraw',
        'quietfire',
        'metsmagic',
        'penpal',
        'thecurse',
        'canadageeseandddp',
        'thatthingonmyarm',
        'buck',
        'thesecrettomarriage',
        'wildwomenanddancingqueens',
        'againstthewind',
        'indianapolis',
        'alternateithicatom',
        'bluehope',
        'kiksuya',
        'afatherscover',
        'haveyoumethimyet',
        'firetestforlove',
        'catfishingstrangerstofindmyself',
        'christmas1940',
        'tildeath',
        'lifeanddeathontheoregontrail',
        'vixenandtheussr',
        'undertheinfluence',
        'beneaththemushroomcloud',
        'jugglingandjesus',
        'superheroesjustforeachother',
        'sweetaspie',
        'naked',
        'singlewomanseekingmanwich',
        'avatar',
        'whenmothersbullyback',
        'myfathershands',
        'reachingoutbetweenthebars',
        'theinterview',
        'stagefright',
        'legacy',
        'canplanetearthfeedtenbillionpeoplepart3',
        'listo',
        'gangstersandcookies',
        'birthofanation',
        'mybackseatviewofagreatromance',
        'lawsthatchokecreativity',
        'threemonths',
        'whyimustspeakoutaboutclimatechange',
        'leavingbaghdad',
    ),
}
TRAIN_STORIES['S03'] = TRAIN_STORIES['S02']

TEST_STORIES = {
    'S01': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
    'S02': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
    'S03': ('wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood'),
}


class HuthLabDataset:
  """Loads HuthLab encoding model scaling laws data.

  This class handles loading fMRI responses, audio stimuli, and
  story-level metadata from the HuthLab dataset.
  """

  def __init__(
      self,
      base_path: str | None = None,
      subject: str = 'S03',
      tr_duration: float = DEFAULT_TR_DURATION,
  ):
    """Initializes the HuthLabDataset.

    Args:
      base_path: Root path to the data. If None, uses the --huthlab_data_path
        flag.
      subject: Subject identifier (e.g., 'S01', 'S02', 'S03').
      tr_duration: Duration of one fMRI TR in seconds.
    """
    if base_path is None:
      base_path = _HUTHLAB_DATA_PATH.value
    if base_path is None:
      raise ValueError(
          'Must provide base_path or set --huthlab_data_path flag.'
      )
    self.base_path = base_path
    self.subject = subject
    self.tr_duration = tr_duration
    self._resp = None

  def _path(self, *args: str) -> str:
    return os.path.join(self.base_path, *args)

  def load_responses(self) -> dict[str, np.ndarray]:
    """Loads fMRI responses (averaged across sessions).

    Returns:
      Dict of story name to fMRI responses, each of shape (T, V) where T is the
      total number of TRs (after trimming) across all stories for that split,
      and V is the number of voxels.
    """
    if self._resp is None:
      path = self._path('responses', f'UT{self.subject}_responses.jbl')
      logging.info('Loading responses from %s', path)
      self._resp = joblib.load(io.BytesIO(epath.Path(path).read_bytes()))
      logging.info(
          'Loaded responses for %d stories, first story dims: %s',
          len(self._resp),
          next(iter(self._resp.values())).shape if self._resp else 'N/A',
      )
    return self._resp

  def load_train_responses(self) -> np.ndarray:
    """Loads training fMRI responses.

    Returns:
      Array of shape (T_train, V) where T_train is the total number of
      train TRs (after trimming) across all train stories, and V is the
      number of voxels.
    """
    resp = self.load_responses()
    return np.vstack([resp[story] for story in self.get_train_stories()])

  def load_test_responses(self) -> np.ndarray:
    """Loads test fMRI responses (averaged across sessions).

    Returns:
      Array of shape (T_test, V) where T_test is the total number of
      test TRs (after trimming) across all test stories, and V is the
      number of voxels.
    """
    resp = self.load_responses()
    return np.vstack([resp[story][40:] for story in self.get_test_stories()])

  def get_train_stories(self) -> Sequence[str]:
    """Returns the list of training story names."""
    return TRAIN_STORIES[self.subject]

  def get_test_stories(self) -> Sequence[str]:
    """Returns the list of test story names."""
    return TEST_STORIES[self.subject]

  def get_audio_path(self, story: str) -> str:
    """Returns the path to the audio file for a story."""
    return self._path('stimuli', f'{story}.wav')
