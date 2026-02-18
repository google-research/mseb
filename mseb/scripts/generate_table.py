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
  return f"{value:.4f}" if isinstance(value, float) else str(value)


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
  html += "      <th>Metric</th>\n"
  for name in present_encoders:
    url = name_urls.get(name)
    name_html = f'<a href="{url}">{name}</a>' if url else name
    html += f"      <th>{name_html}</th>\n"
  html += "    </tr>\n"
  html += "  </thead>\n"
  html += "  <tbody>\n"
  for col in columns:
    _, task_type, task_languages, doc_file = task_info[col]
    lang_str = ", ".join(task_languages)
    html += "    <tr>\n"
    doc_link = ""
    if doc_file:
      # Link to anchor at the bottom of the page
      doc_link = (
          f' <a href="#type-{main_task_type.lower()}" class="doc-link">[?]</a>'
      )
    html += f"     <td>{col}<br/>({task_type}, {lang_str}){doc_link}</td>\n"
    for name in present_encoders:
      value = data[name].get(col, "N/A")
      html += f"      <td>{format_value(value)}</td>\n"
    html += "    </tr>\n"
  html += "  </tbody>\n"
  html += "</table>\n"
  html += "</div>\n"
  html += "</div>\n"
  return html


def generate_html_table(
    results: List[leaderboard.FlattenedLeaderboardResult],
) -> tuple[str, dict[str, dict[str, dict[str, List[str]]]]]:
  """Generates an HTML table and returns documentation grouped by type."""
  if not results:
    return "<p>No results to display.</p>", {}

  # Group results by name
  data = {}
  # Collect all unique task/metric combinations for columns
  columns = set()
  task_info = {}
  scores_by_type = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  task_type_descriptions = {}
  name_urls = {}
  # task_type -> doc_file -> dataset_doc_file -> list of task_names
  docs_by_type = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(list))
  )

  for r in results:
    if r.name not in data:
      data[r.name] = {}
    if r.name not in name_urls:
      name_urls[r.name] = r.url
    column_key = f"{r.task_name} ({r.main_score_metric})"
    columns.add(column_key)

    main_task_type = (
        r.task_subtypes[0] if r.task_subtypes else r.task_type
    ).lower()
    task_info[column_key] = (
        main_task_type,
        r.task_type,
        r.task_languages,
        r.documentation_file,
    )

    if r.documentation_file:
      tasks_list = docs_by_type[main_task_type][r.documentation_file][
          r.dataset_documentation_file
      ]
      if r.task_name not in tasks_list:
        tasks_list.append(r.task_name)

    # We only want to display the main score value
    if r.metric == r.main_score_metric:
      data[r.name][column_key] = r.metric_value
      scores_by_type[r.name][main_task_type].append(r.metric_value)
      if main_task_type not in task_type_descriptions:
        task_type_descriptions[main_task_type] = r.metric_description

  sorted_names = sorted(data.keys())

  # Calculate mean scores by task type
  mean_scores_by_type = collections.defaultdict(dict)
  for name, type_scores in scores_by_type.items():
    for task_type, scores in type_scores.items():
      mean_scores_by_type[name][task_type] = sum(scores) / len(scores)

  main_task_types = sorted(list(set(ti[0] for ti in task_info.values())))
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
      value = mean_scores_by_type[name].get(task_type, "N/A")
      html += f"      <td>{format_value(value)}</td>\n"
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

  for task_type in main_task_types:
    if task_type in cols_by_type:
      html += generate_detail_table(
          data,
          sorted_names,
          task_type,
          cols_by_type[task_type],
          task_info,
          name_urls,
      )

  return html, docs_by_type


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

  html_table, docs_by_type = generate_html_table(flattened_results)
  docs_section = generate_docs_section(docs_by_type)

  html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy" content="script-src 'self'">
<title>Leaderboard</title>
<link rel="stylesheet" href="leaderboard.css">
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
