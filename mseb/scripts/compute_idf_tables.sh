#!/bin/bash

DUMP_DATE="20251001"
output_dir="${HOME}/tmp/idf_tables"
python_bindir="${HOME}/python/bin"
scripts_dir=`pwd`

languages="ar bn en fi gu hi id ja kn ko ml mr ru sw ta te ur"
cpus=30

if [ ! -f "${python_bindir}/python" ]; then
  echo "Python virtual environment not found. Creating it now at ~/python..."
  python3 -m venv "${HOME}/python"
  echo "Environment created successfully."
fi

# We use the spacy multi-lamguage tokenizer for these languages:
#  - 'sw': no spacy support,
#  - 'ko': encountered out-of-bound errors when using 'ko' tokenizer.
function set_spacy_language() {
  case $language in
    "ko"|"sw")
      spacy_language="xx"
      ;;
    *)
      spacy_language=${language}
  esac
}

# Some REs in wikiextractor do not follow the lastest format.
# Apply fix. This should be safe.
function patch_wikiextractor() {
  local location=`~/python/bin/pip show -f wikiextractor | grep "Location: "`
  local file=${location#"Location: "}/wikiextractor/extract.py
  local backup=${file}.bak
  if [ ! -s ${backup} ]; then
    cp -a ${file} ${backup}
  fi
  echo "Patching ${file}..."
  sed -e 's/\\\[(((?i)/(?i)\\\[((/' ${backup} |
  sed -e 's#r"""\^(http://|https://)#r"""(?i)^(http://|https://)#' |
  sed -e 's#((?i)gif|png|jpg|jpeg)#(gif|png|jpg|jpeg)#' > ${file}
  diff ${backup} ${file}
}

# Warning: there is a memory leak in the spacy Japanese tokenizer.
# See https://github.com/explosion/spaCy/issues/13684
# A workaround is to comment out line 89 of the following file in your
# local spacy installation:
#  spacy/lang/ja/__init__.py
# The following function apply the workaround but it might break some features
# spacy Japanese models so we'll revert that change at the end of the script.
function patch_spacy() {
  local location=`~/python/bin/pip show -f spacy | grep "Location: "`
  local file=${location#"Location: "}/spacy/lang/ja/__init__.py
  local backup=${file}.bak
  if [ ! -s ${backup} ]; then
    cp -a ${file} ${backup}
  fi
  echo "Patching ${file}..."
  sed -e 's/token.morph = MorphAnalysis(self\.vocab, morph)/#token.morph = MorphAnalysis(self\.vocab, morph)/' ${backup} > ${file}
  diff ${backup} ${file}
  spacy_ja_file=${file}
  spacy_ja_backup=${backup}
}

# Restore the spacy Japanese tokenizer to its original state.
function restore_spacy() {
  echo "Restoring ${spacy_ja_file}..."
  cp -a ${spacy_ja_backup} ${spacy_ja_file}
}

${python_bindir}/pip install -U pip setuptools wheel
${python_bindir}/pip install -U spacy[ja] unicodecsv wikiextractor

patch_wikiextractor
patch_spacy

for language in ${languages}; do
  wiki=${language}wiki
  set_spacy_language
  echo "${wiki}, using spacy language: ${spacy_language}"
  mkdir -p ${output_dir}/${wiki}
  pushd ${output_dir}/${wiki}
  wget https://dumps.wikimedia.org/${wiki}/${DUMP_DATE}/${wiki}-${DUMP_DATE}-pages-articles.xml.bz2
  mkdir -p data
  ${python_bindir}/wikiextractor -o data --processes=${cpus} --json  ${wiki}-${DUMP_DATE}-pages-articles.xml.bz2
  ${python_bindir}/python ${scripts_dir}/wikipediaidf-spacy.py -s ${spacy_language} -i data/*/*  -o idf -c ${cpus}
  popd
done

restore_spacy
