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

  for r in results:
    if r.name not in data:
      data[r.name] = {}
    column_key = f"{r.task_name} ({r.main_score_metric})"
    columns.add(column_key)
    task_info[column_key] = (r.task_type, r.task_languages)
    # We only want to display the main score value
    if r.metric == r.main_score_metric:
      data[r.name][column_key] = r.metric_value
      scores_by_type[r.name][r.task_type].append(r.metric_value)

  sorted_columns = sorted(list(columns))
  sorted_names = sorted(data.keys())

  # Calculate mean scores by task type
  mean_scores_by_type = collections.defaultdict(dict)
  for name, type_scores in scores_by_type.items():
    for task_type, scores in type_scores.items():
      mean_scores_by_type[name][task_type] = sum(scores) / len(scores)

  task_types = sorted(list(set(ti[0] for ti in task_info.values())))
  mean_cols_map = {}  # Map type to key
  for task_type in task_types:
    mean_col_key = f"{task_type} (mean)"
    mean_cols_map[task_type] = mean_col_key
    for name in sorted_names:
      if name in mean_scores_by_type and task_type in mean_scores_by_type[name]:
        data[name][mean_col_key] = mean_scores_by_type[name][task_type]

  html = '<table id="results-table" border="1">\n'
  # Header row
  html += "  <thead>\n"
  html += "    <tr>\n"
  html += "      <th>Rank</th>\n"
  html += "      <th>Encoder Name</th>\n"
  for task_type in task_types:
    html += (
        '      <th class="toggle-mean"'
        f' data-toggle-task-type="{task_type}"><span'
        f' class="toggle-icon">+</span>{mean_cols_map[task_type]} <span'
        ' class="sort-icon"></span></th>\n'
    )

  cols_by_type = collections.defaultdict(list)
  for col in sorted_columns:
    col_task_type, _ = task_info[col]
    cols_by_type[col_task_type].append(col)

  ordered_task_columns = []
  for task_type in task_types:
    ordered_task_columns.extend(cols_by_type[task_type])

  for col in ordered_task_columns:
    task_type, task_languages = task_info[col]
    lang_str = ", ".join(task_languages)
    html += (
        '      <th class="task-col-header"'
        f' data-task-type="{task_type}">{col}<br/>({task_type},'
        f" {lang_str})</th>\n"
    )
  html += "    </tr>\n"
  html += "  </thead>\n"

  # Data rows
  html += "  <tbody>\n"
  for i, name in enumerate(sorted_names):
    html += "    <tr>\n"
    html += f"     <td>{i + 1}</td>\n"
    html += f"     <td>{name}</td>\n"
    for task_type in task_types:
      value = data[name].get(mean_cols_map[task_type], "N/A")
      html += (
          f"      <td>{value:.4f}</td>\n"
          if isinstance(value, float)
          else f"      <td>{value}</td>\n"
      )
    for col in ordered_task_columns:
      task_type, _ = task_info[col]
      value = data[name].get(col, "N/A")
      cell_content = f"{value:.4f}" if isinstance(value, float) else str(value)
      html += (
          '      <td class="task-col-cell"'
          f' data-task-type="{task_type}">{cell_content}</td>\n'
      )
    html += "    </tr>\n"
  html += "  </tbody>\n"

  html += "</table>"
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
