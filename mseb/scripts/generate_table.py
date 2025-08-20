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

import json
import os
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

  for r in results:
    if r.name not in data:
      data[r.name] = {}
    column_key = f"{r.task_name} ({r.main_score_metric})"
    columns.add(column_key)
    # We only want to display the main score value
    if r.metric == r.main_score_metric:
      data[r.name][column_key] = r.metric_value

  sorted_columns = sorted(list(columns))
  sorted_names = sorted(data.keys())

  html = "<table border=\"1\">\n"
  # Header row
  html += "  <thead>\n"
  html += "    <tr>\n"
  html += "      <th>Name</th>\n"  # First column is the name
  for col in sorted_columns:
    html += f"      <th>{col}</th>\n"
  html += "    </tr>\n"
  html += "  </thead>\n"

  # Data rows
  html += "  <tbody>\n"
  for name in sorted_names:
    html += "    <tr>\n"
    html += f"     <td>{name}</td>\n"
    for col in sorted_columns:
      value = data[name].get(col, "N/A")
      html += (
          f"      <td>{value:.4f}</td>\n"
          if isinstance(value, float)
          else f"      <td>{value}</td>\n"
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

  html_table = generate_html_table(flattened_results)

  javascript_code = """

function sortTable(table, column, asc = true) {
  const dirModifier = asc ? 1 : -1;
  const tBody = table.tBodies[0];
  const rows = Array.from(tBody.querySelectorAll("tr"));

  const sortedRows = rows.sort((a, b) => {
    const aColText = a.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();
    const bColText = b.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();

    const aVal = parseFloat(aColText);
    const bVal = parseFloat(bColText);

    if (!isNaN(aVal) && !isNaN(bVal)) {
      return (aVal - bVal) * dirModifier;
    }
    return aColText.localeCompare(bColText) * dirModifier;
  });

  while (tBody.firstChild) {
    tBody.removeChild(tBody.firstChild);
  }
  tBody.append(...sortedRows);

  table.querySelectorAll("th").forEach(th => th.removeAttribute("data-sort-direction"));
  table.querySelector(`th:nth-child(${ column + 1 })`).setAttribute("data-sort-direction", asc ? "asc" : "desc");
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll("th").forEach((headerCell, headerIndex) => {
    if (headerIndex === 0) return; // Don't sort by the Name column
    headerCell.addEventListener("click", () => {
      const tableElement = headerCell.closest("table");
      const currentAsc = headerCell.getAttribute("data-sort-direction") === "asc";
      sortTable(tableElement, headerIndex, !currentAsc);
    });
  });
});
"""

  output_dir = os.path.dirname(_OUTPUT_FILE.value)
  js_file_path = os.path.join(output_dir, "leaderboard.js")
  with open(js_file_path, "w") as f:
    f.write(javascript_code)

  html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy" content="script-src 'self'">
<title>Leaderboard</title>
<style>
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background-color: #f2f2f2; cursor: pointer; }}
  tr:nth-child(even) {{ background-color: #f9f9f9; }}
</style>
<script src="leaderboard.js"></script>
</head>
<body>
  <h1>Leaderboard Results</h1>
  {html_table}
</body>
</html>
"""

  with open(_OUTPUT_FILE.value, "w") as f:
    f.write(html_content)

  print(f"HTML table written to {_OUTPUT_FILE.value}")
  print(f"JavaScript written to {js_file_path}")


if __name__ == "__main__":
  app.run(main)
