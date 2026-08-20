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

"""Custom CSS stylesheets and UI styling for the MSEB Gradio Leaderboard."""

CUSTOM_CSS = """
/* ==========================================================================
   MSEB Leaderboard Custom Stylesheet
   ========================================================================== */

/* Root & Global Typography */
:root {
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --bg-card: #ffffff;
    --border-color: #e5e7eb;
    --text-main: #1f2937;
    --text-muted: #6b7280;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* App Header & Title Banner */
.header-container {
    text-align: center;
    margin: 1.5rem auto 2rem auto;
    max-width: 900px;
    padding: 0 1rem;
}

.header-title {
    font-size: 2.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

.header-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    line-height: 1.6;
    margin-bottom: 1rem;
}

.stat-pills {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}

.stat-pill {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 9999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.825rem;
    font-weight: 600;
    color: #334155;
}

/* Navigation Tabs */
.tab-buttons button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: 0.5rem 0.5rem 0 0 !important;
    transition: all 0.15s ease-in-out;
}

.tab-buttons button.selected {
    color: var(--primary-color) !important;
    border-bottom: 3px solid var(--primary-color) !important;
}

/* Control Panel: Search & Column Toggles */
.filter-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}

.search-input input {
    border-radius: 0.5rem !important;
    border: 1px solid #cbd5e1 !important;
    padding: 0.6rem 0.85rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.search-input input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* Checkbox Column Selector */
.column-selector-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 0.25rem !important;
    gap: 0.75rem !important;
}

.selector-label {
    margin: 0 !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

.toggle-all-btn {
    padding: 0.2rem 0.65rem !important;
    font-size: 0.775rem !important;
    font-weight: 600 !important;
    border-radius: 0.375rem !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    width: auto !important;
    min-width: 90px !important;
    height: auto !important;
}

.toggle-all-btn:hover {
    background-color: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

.column-selector-group {
    margin-top: 0.25rem;
}

.column-selector-group .wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.column-selector-group label {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 0.375rem;
    padding: 0.25rem 0.6rem;
    font-size: 0.825rem;
    cursor: pointer;
    transition: background-color 0.15s, border-color 0.15s;
    color: #334155;
}

.column-selector-group label:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
}

.column-selector-group label.selected,
.column-selector-group label.checked,
.column-selector-group label:has(input:checked) {
    background-color: var(--primary-color) !important;
    color: #ffffff !important;
    border-color: var(--primary-color) !important;
}

/* Leaderboard Table Styling */
.leaderboard-table {
    border-radius: 0.5rem;
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-top: 0.75rem;
}

.leaderboard-table table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.925rem;
}

.leaderboard-table th {
    background-color: #f8fafc !important;
    color: #334155 !important;
    font-weight: 700 !important;
    padding: 0.3rem 0.5rem !important;
    text-align: center !important;
    border-bottom: 2px solid #e2e8f0 !important;
    white-space: nowrap !important;
    position: sticky;
    top: 0;
    z-index: 10;
}

.leaderboard-table th:first-child,
.leaderboard-table th:nth-child(2),
.leaderboard-table th:nth-child(3) {
    text-align: left !important;
}

.leaderboard-table td {
    padding: 0.3rem 0.5rem !important;
    border-bottom: 1px solid #f1f5f9 !important;
    text-align: center !important;
    vertical-align: middle !important;
}

.leaderboard-table .cell-wrap,
.leaderboard-table td > div,
.leaderboard-table th > div,
.leaderboard-table th span,
.leaderboard-table th button {
    padding: 0 !important;
    margin: 0 !important;
}

.leaderboard-table td:first-child,
.leaderboard-table td:nth-child(2),
.leaderboard-table td:nth-child(3) {
    text-align: left !important;
}

.leaderboard-table tr:hover td {
    background-color: #f8fafc !important;
}

/* Model Links in Table */
.leaderboard-table td a {
    color: #1d4ed8 !important;
    font-weight: 600 !important;
    text-decoration: none !important;
}

.leaderboard-table td a:hover {
    text-decoration: underline !important;
}

/* Number Monospace in Scores (columns after Rank, Model, Tags) */
.leaderboard-table td:not(:first-child):not(:nth-child(2)):not(:nth-child(3)) {
    font-family: var(--font-mono) !important;
    font-variant-numeric: tabular-nums;
}

/* Suppress Markdown list bullets and ::marker pseudo-elements in table cells */
.leaderboard-table td ul,
.leaderboard-table td ol,
.leaderboard-table td li {
    list-style: none !important;
    list-style-type: none !important;
    display: inline !important;
    margin: 0 !important;
    padding: 0 !important;
}

.leaderboard-table td li::marker,
.leaderboard-table td *::marker {
    content: "" !important;
    display: none !important;
}

.leaderboard-table td li:empty::before {
    content: "-" !important;
    display: inline !important;
}

/* Task Info Callout */
.task-info-banner {
    background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    border-radius: 0.5rem;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1rem;
}

.task-info-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #0369a1;
    margin-bottom: 0.25rem;
}

.task-info-desc {
    font-size: 0.9rem;
    color: #334155;
}

/* Inactive Task Warning Banner */
.inactive-task-banner {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 0.5rem;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1rem;
    color: #92400e;
    font-size: 0.9rem;
}

/* Citation Box */
#citation-textbox textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    background: #f8fafc !important;
}

/* Markdown Documentation Typography */
.markdown-text h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 1.25rem;
    margin-bottom: 0.75rem;
    color: #0f172a;
}

.markdown-text h2 {
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
    color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.35rem;
}

.markdown-text h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.4rem;
    color: #334155;
}

.markdown-text p, .markdown-text li {
    font-size: 0.95rem;
    line-height: 1.65;
    color: #374151;
}

.markdown-text table {
    width: 100%;
    margin: 1rem 0;
    border-collapse: collapse;
}

.markdown-text th, .markdown-text td {
    padding: 0.5rem 0.75rem;
    border: 1px solid #e2e8f0;
    font-size: 0.9rem;
}

.markdown-text th {
    background: #f8fafc;
    font-weight: 600;
}
"""

# Alias for backward compatibility
custom_css = CUSTOM_CSS
