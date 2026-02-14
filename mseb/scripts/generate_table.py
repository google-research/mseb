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
from typing import List, Sequence

from absl import app
from absl import flags
from mseb import leaderboard

_INPUT_FILE = flags.DEFINE_string(
    "input_file", None, "Input JSONL file of flattened results.", required=True
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", None, "Output HTML file.", required=True
)


def format_value(value) -> str:
  return f"{value:.4f}" if isinstance(value, float) else str(value)


def generate_detail_table(
    data: dict[str, dict[str, float]],
    sorted_names: List[str],
    main_task_type: str,
    columns: List[str],
    task_info: dict[str, tuple[str, str, List[str]]],
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
    html += f"      <th>{name}</th>\n"
  html += "    </tr>\n"
  html += "  </thead>\n"
  html += "  <tbody>\n"
  for col in columns:
    _, task_type, task_languages = task_info[col]
    lang_str = ", ".join(task_languages)
    html += "    <tr>\n"
    html += f"     <td>{col}<br/>({task_type}, {lang_str})</td>\n"
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
) -> str:
  """Generates an HTML table from flattened leaderboard results.

  Args:
    results: A list of FlattenedLeaderboardResult objects.

  Returns:
    An HTML table as a string.
  """
  if not results:
    return "<p>No results to display.</p>"

  # Group results by name
  data = {}
  # Collect all unique task/metric combinations for columns
  columns = set()
  task_info = {}
  scores_by_type = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  task_type_descriptions = {}

  for r in results:
    if r.name not in data:
      data[r.name] = {}
    column_key = f"{r.task_name} ({r.main_score_metric})"
    columns.add(column_key)

    main_task_type = r.task_subtypes[0] if r.task_subtypes else r.task_type
    task_info[column_key] = (main_task_type, r.task_type, r.task_languages)

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
    html += f"     <td>{name}</td>\n"
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
    main_task_type, _, _ = task_info[col]
    cols_by_type[main_task_type].append(col)

  for task_type in main_task_types:
    if task_type in cols_by_type:
      html += generate_detail_table(
          data, sorted_names, task_type, cols_by_type[task_type], task_info
      )

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

  html_table = generate_html_table(flattened_results)

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
  <div class="table-container">
    {html_table}
  </div>
</body>
</html>
"""

  with open(_OUTPUT_FILE.value, "w") as f:
    f.write(html_content)

  print(f"HTML table written to {_OUTPUT_FILE.value}")


if __name__ == "__main__":
  app.run(main)
