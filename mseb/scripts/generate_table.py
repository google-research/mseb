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

"""Generate an HTML table from flattened leaderboard results."""

import collections
import json
import os
import statistics
from typing import List, Sequence

from absl import app
from absl import flags
import markdown
from mseb import leaderboard

_INPUT_FILE = flags.DEFINE_string(
    "input_file", None, "Input JSONL file of flattened results.", required=True
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "Output HTML file.", required=True
)
_DOCS_DIR = flags.DEFINE_string(
    "docs_dir", None, "Directory containing documentation markdown files."
)


def format_value(value) -> str:
  return f"{value:.2f}" if isinstance(value, float) else str(value)


def render_documentation(doc_file: str | None) -> str:
  """Renders markdown documentation to HTML."""
  if not doc_file or not _DOCS_DIR.value:
    return ""

  doc_path = os.path.join(_DOCS_DIR.value, doc_file)
  if not os.path.exists(doc_path):
    return ""

  with open(doc_path, "r") as f:
    md_content = f.read()
  return markdown.markdown(md_content)


def generate_detail_table(
    data: dict[str, dict[str, float]],
    sorted_names: List[str],
    main_task_type: str,
    columns: List[str],
    task_info: dict[str, tuple[str, str, List[str], str | None]],
    name_urls: dict[str, str | None],
) -> str:
  """Generates an HTML table for individual sub-task results of a task type."""
  # Determine which encoders have any non-"N/A" values for this task type
  present_encoders = []
  for name in sorted_names:
    has_value = False
    for col in columns:
      if data[name].get(col) is not None and data[name].get(col) != "N/A":
        has_value = True
        break
    if has_value:
      present_encoders.append(name)

  if not present_encoders:
    return ""

  html = '<div class="detail-table-section">\n'
  html += f'<h2 id="{main_task_type}">{main_task_type}</h2>\n'
  html += '<div class="table-container">\n'
  html += f'<table id="details-table-{main_task_type}" border="1">\n'
  html += "  <thead>\n"
  html += "    <tr>\n"
  html += "      <th>Model</th>\n"
  for col in columns:
    _, task_type, task_languages, doc_file = task_info[col]
    doc_link = ""
    if doc_file:
      doc_link = (
          f' <a href="#type-{main_task_type.lower()}" class="doc-link">[?]</a>'
      )
    html += f"      <th>{col}{doc_link}</th>\n"
  html += "    </tr>\n"
  html += "  </thead>\n"
  html += "  <tbody>\n"

  # Find the maximum value per column (task) to highlight it
  max_vals = {}
  for col in columns:
    col_values = []
    for name in present_encoders:
      val = data[name].get(col)
      if val is not None and isinstance(val, (int, float)):
        col_values.append(val)
    max_vals[col] = max(col_values) if col_values else None

  for name in present_encoders:
    url = name_urls.get(name)
    name_html = f'<a href="{url}">{name}</a>' if url else name
    html += "    <tr>\n"
    html += f"     <td>{name_html}</td>\n"
    for col in columns:
      value = data[name].get(col, "N/A")
      max_val = max_vals[col]
      is_best = max_val is not None and value == max_val
      td_class = ' class="best-value"' if is_best else ""
      html += f"      <td{td_class}>{format_value(value)}</td>\n"
    html += "    </tr>\n"
  html += "  </tbody>\n"
  html += "</table>\n"
  html += "</div>\n"
  html += "</div>\n"
  return html


def generate_html_table(
    results: List[leaderboard.FlattenedLeaderboardResult],
) -> tuple[str, str, dict[str, dict[str, dict[str, List[str]]]]]:
  """Generates an HTML table and returns documentation grouped by type."""
  if not results:
    return "<p>No results to display.</p>", "", {}

  # Group results by base_model or name
  data = {}
  columns = set()
  task_info = {}
  task_type_descriptions = {}
  name_urls = {}
  docs_by_type = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(list))
  )

  accumulated_data = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )

  for r in results:
    key = r.base_model if r.base_model else r.name
    if key not in name_urls or (r.url and not name_urls[key]):
      name_urls[key] = r.url

    main_task_type = (
        r.task_subtypes[0] if r.task_subtypes else r.task_type
    ).lower()

    # Aggregate by dataset name if available
    aggregated_name = f"{r.dataset_name}/{main_task_type}/{r.sub_task_name}"
    column_key = f"{aggregated_name} ({r.main_score_metric})"
    columns.add(column_key)

    if column_key not in task_info:
      task_info[column_key] = [
          main_task_type,
          r.task_type,
          set(r.task_languages),
          r.documentation_file,
      ]
    else:
      task_info[column_key][2].update(r.task_languages)

    if r.documentation_file:
      tasks_list = docs_by_type[main_task_type][r.documentation_file][
          r.dataset_documentation_file
      ]
      if aggregated_name not in tasks_list:
        tasks_list.append(aggregated_name)

    if r.metric == r.main_score_metric:
      accumulated_data[key][column_key].append(r.metric_value)

      if main_task_type not in task_type_descriptions:
        task_type_descriptions[main_task_type] = r.metric_description

  # Convert sets to sorted lists and lists to tuples in task_info
  for col_key, info in task_info.items():
    task_info[col_key] = (
        info[0],
        info[1],
        sorted(list(info[2])),
        info[3],
    )

  # Compute means for accumulated data
  for key, task_scores in accumulated_data.items():
    if key not in data:
      data[key] = {}
    for column_key, scores in task_scores.items():
      data[key][column_key] = sum(scores) / len(scores) if scores else 0.0

  # Calculate mean scores by task type from aggregated data
  scores_by_type = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  for key, task_scores in data.items():
    for column_key, score in task_scores.items():
      main_task_type = task_info[column_key][0]
      scores_by_type[key][main_task_type].append(score)

  sorted_names = sorted(data.keys())

  # Calculate mean and std scores by task type
  mean_scores_by_type = collections.defaultdict(dict)
  std_scores_by_type = collections.defaultdict(dict)
  for name, type_scores in scores_by_type.items():
    for task_type, scores in type_scores.items():
      mean_scores_by_type[name][task_type] = sum(scores) / len(scores)
      if len(scores) >= 2:
        std_scores_by_type[name][task_type] = statistics.stdev(scores)
      else:
        std_scores_by_type[name][task_type] = 0.0

  main_task_types = sorted(list(set(ti[0] for ti in task_info.values())))

  # Calculate max mean scores by task type for highlighting
  max_means_by_type = {}
  for task_type in main_task_types:
    type_values = [
        mean_scores_by_type[name].get(task_type)
        for name in sorted_names
        if task_type in mean_scores_by_type[name]
    ]
    if type_values:
      max_means_by_type[task_type] = max(type_values)

  mean_cols_map = {
      task_type: f"{task_type} (mean)" for task_type in main_task_types
  }

  # Main Aggregated Table
  html = '<div class="table-container">\n'
  html += '<table id="results-table" border="1">\n'
  # Header row
  html += "  <thead>\n"
  html += "    <tr>\n"
  html += "      <th>Rank</th>\n"
  html += "      <th>Encoder Name</th>\n"
  for task_type in main_task_types:
    description = task_type_descriptions.get(task_type, "")
    html += (
        f'      <th title="{description}"><a'
        f' href="#{task_type}">{mean_cols_map[task_type]}</a> <span'
        ' class="sort-icon"></span></th>\n'
    )
  html += "    </tr>\n"
  html += "  </thead>\n"

  # Data rows for main table
  html += "  <tbody>\n"
  # Sort names by overall mean score (if available) - Assuming a combined mean
  # For now, let's sort by the first mean column
  def get_sort_key(name):
    first_task_type = main_task_types[0] if main_task_types else None
    return mean_scores_by_type[name].get(first_task_type, -1) * -1  # Descending

  sorted_names_by_mean = sorted(sorted_names, key=get_sort_key)

  for i, name in enumerate(sorted_names_by_mean):
    html += "    <tr>\n"
    html += f"     <td>{i + 1}</td>\n"
    url = name_urls.get(name)
    name_html = f'<a href="{url}">{name}</a>' if url else name
    html += f"     <td>{name_html}</td>\n"
    for task_type in main_task_types:
      mean_val = mean_scores_by_type[name].get(task_type, "N/A")
      std_val = std_scores_by_type[name].get(task_type, 0.0)
      is_best = (
          task_type in max_means_by_type
          and mean_val == max_means_by_type[task_type]
      )
      td_class = ' class="best-value"' if is_best else ""
      if isinstance(mean_val, (int, float)):
        val_str = f"{format_value(mean_val)} ± {format_value(std_val)}"
      else:
        val_str = mean_val
      html += f"      <td{td_class}>{val_str}</td>\n"
    html += "    </tr>\n"
  html += "  </tbody>\n"
  html += "</table>\n"
  html += "</div>\n"  # End of table-container for main table

  # Sub-task Detail Tables
  cols_by_type = collections.defaultdict(list)
  sorted_columns = sorted(list(columns))
  for col in sorted_columns:
    main_task_type, _, _, _ = task_info[col]
    cols_by_type[main_task_type].append(col)

  details_html = ""
  for task_type in main_task_types:
    if task_type in cols_by_type:
      details_html += generate_detail_table(
          data,
          sorted_names,
          task_type,
          cols_by_type[task_type],
          task_info,
          name_urls,
      )

  return html, details_html, docs_by_type


def generate_comparison_html_table(
    truth_results: List[leaderboard.FlattenedLeaderboardResult],
    cascaded_results: List[leaderboard.FlattenedLeaderboardResult],
    audio_results: List[leaderboard.FlattenedLeaderboardResult],
) -> str:
  """Generates an HTML table comparing Truth and Cascaded results."""

  def group_results(results):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    name_urls = {}
    for r in results:
      key = r.base_model if r.base_model else r.name
      if key not in name_urls or (r.url and not name_urls[key]):
        name_urls[key] = r.url
      if r.metric == r.main_score_metric:
        main_task_type = (
            r.task_subtypes[0] if r.task_subtypes else r.task_type
        ).lower()
        data[key][main_task_type].append(r.metric_value)

    means = collections.defaultdict(dict)
    for key, type_scores in data.items():
      for task_type, scores in type_scores.items():
        means[key][task_type] = sum(scores) / len(scores)
    return means, name_urls

  truth_means, truth_urls = group_results(truth_results)
  cascaded_means, cascaded_urls = group_results(cascaded_results)
  audio_means, audio_urls = group_results(audio_results)

  common_models = sorted(
      list(set(truth_means.keys()) & set(cascaded_means.keys()))
  )

  if not common_models:
    return "<p>No common models to compare for headroom.</p>"

  all_task_types = set()
  for model in common_models:
    all_task_types.update(truth_means[model].keys())
    all_task_types.update(cascaded_means[model].keys())
  sorted_task_types = sorted(list(all_task_types))

  max_truth_by_type = {}
  max_cascaded_by_type = {}
  max_audio_by_type = {}
  for task_type in sorted_task_types:
    truth_vals = [
        truth_means[model].get(task_type)
        for model in common_models
        if task_type in truth_means[model]
    ]
    truth_vals = [v for v in truth_vals if isinstance(v, (int, float))]
    if truth_vals:
      max_truth_by_type[task_type] = max(truth_vals)

    cascaded_vals = [
        cascaded_means[model].get(task_type)
        for model in common_models
        if task_type in cascaded_means[model]
    ]
    cascaded_vals = [v for v in cascaded_vals if isinstance(v, (int, float))]
    if cascaded_vals:
      max_cascaded_by_type[task_type] = max(cascaded_vals)

    audio_vals = [
        audio_means[model].get(task_type)
        for model in common_models
        if task_type in audio_means[model]
    ]
    audio_vals = [v for v in audio_vals if isinstance(v, (int, float))]
    if audio_vals:
      max_audio_by_type[task_type] = max(audio_vals)

  html = '<div class="table-container">\n'
  html += '<table id="comparison-table" border="1">\n'
  html += "  <thead>\n"
  html += "    <tr>\n"
  html += '      <th rowspan="2">Encoder Name</th>\n'
  for task_type in sorted_task_types:
    html += f'      <th colspan="3">{task_type}</th>\n'
  html += "    </tr>\n"
  html += "    <tr>\n"
  for task_type in sorted_task_types:
    html += "      <th>Transcript Truth</th>\n"
    html += "      <th>Cascaded ASR</th>\n"
    html += "      <th>Audio</th>\n"
  html += "    </tr>\n"
  html += "  </thead>\n"
  html += "  <tbody>\n"

  for model in common_models:
    html += "    <tr>\n"
    url = (
        truth_urls.get(model)
        or cascaded_urls.get(model)
        or audio_urls.get(model)
    )
    name_html = f'<a href="{url}">{model}</a>' if url else model
    html += f"     <td>{name_html}</td>\n"
    for task_type in sorted_task_types:
      truth_val = truth_means[model].get(task_type)
      cascaded_val = cascaded_means[model].get(task_type)
      audio_val = audio_means[model].get(task_type)

      is_best_truth = (
          task_type in max_truth_by_type
          and truth_val == max_truth_by_type[task_type]
      )
      is_best_cascaded = (
          task_type in max_cascaded_by_type
          and cascaded_val == max_cascaded_by_type[task_type]
      )
      is_best_audio = (
          task_type in max_audio_by_type
          and audio_val == max_audio_by_type[task_type]
      )

      td_class_truth = ' class="best-value"' if is_best_truth else ""
      td_class_cascaded = ' class="best-value"' if is_best_cascaded else ""
      td_class_audio = ' class="best-value"' if is_best_audio else ""

      truth_str = format_value(truth_val) if truth_val is not None else "N/A"
      cascaded_str = (
          format_value(cascaded_val) if cascaded_val is not None else "N/A"
      )
      audio_str = format_value(audio_val) if audio_val is not None else "N/A"

      if truth_val is not None and cascaded_val is not None:
        delta = cascaded_val - truth_val
        delta_str = format_value(delta)
        if delta > 0:
          delta_str = f"+{delta_str}"

        if truth_val != 0:
          rel_change = delta / truth_val * 100
          rel_str = f"{rel_change:.2f}%"
          if rel_change > 0:
            rel_str = f"+{rel_str}"
          cascaded_str = f"{cascaded_str} (&Delta;{delta_str}, {rel_str})"
        else:
          cascaded_str = f"{cascaded_str} (&Delta;{delta_str})"

      html += f"      <td{td_class_truth}>{truth_str}</td>\n"
      html += f"      <td{td_class_cascaded}>{cascaded_str}</td>\n"
      html += f"      <td{td_class_audio}>{audio_str}</td>\n"
    html += "    </tr>\n"

  html += "  </tbody>\n"
  html += "</table>\n"
  html += "</div>\n"

  return html


def generate_docs_section(
    docs_by_type: dict[str, dict[str, dict[str, List[str]]]],
) -> str:
  """Generates the documentation section with task/dataset separation."""
  if not docs_by_type:
    return ""

  html = '<div class="docs-section">\n'
  html += "  <h1>Task Definitions</h1>\n"

  all_dataset_docs = set()

  # Task Type Sections
  for task_type in sorted(docs_by_type.keys()):
    task_type_lower = task_type.lower()
    html += f'  <div id="type-{task_type_lower}" class="type-entry">\n'

    # 1. Abstract Type Description
    type_doc = f"type_{task_type_lower}.md"
    type_html = render_documentation(type_doc)
    if type_html:
      html += type_html
    else:
      html += f"<h2>{task_type.capitalize()}</h2>\n"

    # 2. Specific Implementation per Dataset
    for doc_file, dataset_map in sorted(docs_by_type[task_type].items()):
      doc_html = render_documentation(doc_file)
      if doc_html:
        html += '<div class="task-implementation">\n'
        html += doc_html
        # 3. List of tasks using this implementation and link to dataset
        for dataset_doc, tasks in sorted(dataset_map.items()):
          dataset_link = ""
          if dataset_doc:
            all_dataset_docs.add(dataset_doc)
            safe_dataset_id = dataset_doc.replace(".", "_")
            dataset_name = (
                dataset_doc.replace("dataset_", "").replace(".md", "").upper()
            )
            dataset_link = (
                f' (See <a href="#doc-{safe_dataset_id}">{dataset_name}</a>)'
            )
          html += f"<p><strong>Tasks{dataset_link}:</strong></p>\n<ul>\n"
          for task in sorted(tasks):
            html += f"  <li>{task}</li>\n"
          html += "</ul>\n"
        html += "</div>\n"

    html += '    <p><a href="#">[Back to top]</a></p>\n'
    html += "  </div>\n"
    html += "<hr/>\n"

  # Dataset Details Section
  if all_dataset_docs:
    html += "  <h1>Datasets</h1>\n"
    for dataset_doc in sorted(list(all_dataset_docs)):
      dataset_html = render_documentation(dataset_doc)
      if dataset_html:
        safe_dataset_id = dataset_doc.replace(".", "_")
        html += f'  <div id="doc-{safe_dataset_id}" class="dataset-entry">\n'
        html += dataset_html
        html += '    <p><a href="#">[Back to top]</a></p>\n'
        html += "  </div>\n"
        html += "  <hr/>\n"

  html += "</div>\n"
  return html


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  flattened_results = []
  with open(_INPUT_FILE.value, "r") as f:
    for line in f:
      try:
        data = json.loads(line)
        flattened_results.append(leaderboard.FlattenedLeaderboardResult(**data))
      except json.JSONDecodeError:
        print(f"Skipping invalid JSON line: {line.strip()}")
      except TypeError as e:
        print(f"Skipping line with missing fields: {line.strip()} - {e}")

  tasks = set(r.task_name for r in flattened_results)
  task_types = set(r.task_type for r in flattened_results)
  languages = set()
  for r in flattened_results:
    for lang in r.task_languages:
      languages.add(lang)

  stats_table = f"""
<table class="summary-table">
  <tr><td>Tasks:</td><td>{len(tasks)}</td></tr>
  <tr><td>Task Types:</td><td>{len(task_types)}</td></tr>
  <tr><td>Languages:</td><td>{len(languages)}</td></tr>
</table>
"""

  audio_results = [
      r
      for r in flattened_results
      if "transcript_truth" not in r.tags and "cascaded" not in r.tags
  ]
  transcript_truth_results = [
      r for r in flattened_results if "transcript_truth" in r.tags
  ]
  cascaded_results = [r for r in flattened_results if "cascaded" in r.tags]

  html_audio, details_audio, docs_audio = generate_html_table(audio_results)
  html_transcript_truth, details_transcript_truth, docs_transcript_truth = (
      generate_html_table(transcript_truth_results)
  )
  html_cascaded, details_cascaded, docs_cascaded = generate_html_table(
      cascaded_results
  )

  html_comparison = generate_comparison_html_table(
      transcript_truth_results, cascaded_results, audio_results
  )

  # Merge docs
  docs_by_type = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(list))
  )
  for d in [docs_audio, docs_transcript_truth, docs_cascaded]:
    for k1, v1 in d.items():
      for k2, v2 in v1.items():
        for k3, v3 in v2.items():
          for x in v3:
            if x not in docs_by_type[k1][k2][k3]:
              docs_by_type[k1][k2][k3].append(x)

  docs_section = generate_docs_section(docs_by_type)

  html_table = ""
  if transcript_truth_results and cascaded_results:
    html_table += "<h1>Headroom and Audio Comparison</h1>\n" + html_comparison

  if audio_results:
    html_table += "<h1>Audio Encoders</h1>\n" + html_audio
  if audio_results:
    html_table += "<h1>Audio Encoder Details</h1>\n" + details_audio

  html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy" content="script-src 'self'">
<title>Leaderboard</title>
<link rel="stylesheet" href="leaderboard.css">
<style>
.best-value {{
  background-color: #e6f3ff;
  font-weight: bold;
}}
</style>
<script src="leaderboard.js"></script>
</head>
<body>
  <div class="header-section">
    <h2>MSEB Leaderboard</h2>
    <p>This is a leaderboard for MSEB: Massive Speech Embedding Benchmark.</p>
    <p>For more information, see the <a href="https://github.com/google-research/mseb">MSEB GitHub repository</a>.</p>
{stats_table}
  </div>
  <h1>Leaderboard Results</h1>
  <div>
    <label for="encoder-filter">Encoder Name Filter:</label>
    <input type="text" id="encoder-filter" name="encoder-filter">
  </div>
  <div class="chart-section">
    <div class="chart-container">
      <canvas id="spider-chart"></canvas>
    </div>
  </div>
  <div class="table-container">
    {html_table}
  </div>
  <div class="docs-container">
    {docs_section}
  </div>
</body>
</html>
"""

  with open(_OUTPUT_FILE.value, "w") as f:
    f.write(html_content)

  print(f"HTML table written to {_OUTPUT_FILE.value}")


if __name__ == "__main__":
  app.run(main)
