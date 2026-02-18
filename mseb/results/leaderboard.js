

/**
 * Sorts an HTML table by a given column.
 * @param {!HTMLTableElement} table The table element to sort.
 * @param {number} column The index of the column to sort by.
 * @param {boolean} [asc=true] True for ascending sort, false for descending.
 */
function sortTable(table, column, asc = true) {
  const dirModifier = asc ? 1 : -1;
  const tBody = table.tBodies[0];
  const rows = Array.from(tBody.querySelectorAll("tr"));

  const sortedRows = rows.sort((a, b) => {
    const aColText = a.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();
    const bColText = b.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();

    const aVal = parseFloat(aColText);
    const bVal = parseFloat(bColText);
    const aIsNaN = isNaN(aVal);
    const bIsNaN = isNaN(bVal);

    if (aIsNaN && bIsNaN) {
      return aColText.localeCompare(bColText) * dirModifier;
    }
    if (aIsNaN) {  // a is N/A, b is a number. N/A is worse than number.
      return -1 * dirModifier;
    }
    if (bIsNaN) {  // b is N/A, a is a number. N/A is worse than number.
      return 1 * dirModifier;
    }
    return (aVal - bVal) * dirModifier;
  });

  while (tBody.firstChild) {
    tBody.removeChild(tBody.firstChild);
  }
  tBody.append(...sortedRows);

  table.querySelectorAll("th").forEach(th => th.removeAttribute("data-sort-direction"));
  table.querySelector(`th:nth-child(${ column + 1 })`).setAttribute("data-sort-direction", asc ? "asc" : "desc");

  // Update rank column if it exists (i.e., for the main results-table)
  if (table.id === 'results-table') {
    const numRows = tBody.querySelectorAll('tr').length;
    tBody.querySelectorAll('tr').forEach((row, index) => {
      row.querySelector('td:first-child').textContent =
          asc ? numRows - index : index + 1;
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const resultsTable = document.getElementById('results-table');
  if (!resultsTable) return;

  // Sorting listeners for the main table's mean columns
  resultsTable.querySelectorAll('th').forEach((headerCell, headerIndex) => {
    // Only sort columns starting from index 2 (after Rank and Encoder Name)
    if (headerIndex > 1) {
      headerCell.addEventListener('click', () => {
        const currentAsc =
            headerCell.getAttribute('data-sort-direction') === 'asc';
        sortTable(resultsTable, headerIndex, !currentAsc);
      });
    }
  });

  // Sorting for individual detail tables
  document.querySelectorAll('[id^="details-table-"]').forEach(detailTable => {
    detailTable.querySelectorAll('th').forEach((headerCell, headerIndex) => {
      // Don't sort by Encoder Name (index 0)
      if (headerIndex > 0) {
        headerCell.addEventListener('click', () => {
          const currentAsc =
              headerCell.getAttribute('data-sort-direction') === 'asc';
          sortTable(detailTable, headerIndex, !currentAsc);
        });
      }
    });
  });

  // Filter listener
  const filterInput = document.getElementById('encoder-filter');
  if (!filterInput) return;

  filterInput.addEventListener('input', () => {
    const filterValue = filterInput.value;
    let regex;
    try {
      regex = new RegExp(filterValue, 'i');
    } catch (e) {
      // Invalid regex. Show all rows/columns.
      document.querySelectorAll('#results-table tbody tr')
          .forEach(row => row.style.display = '');
      document.querySelectorAll('[id^="details-table-"]')
          .forEach(detailTable => {
            detailTable.querySelectorAll('th').forEach((th, index) => {
              if (index > 0) th.style.display = '';
            });
            detailTable.querySelectorAll('td').forEach((td, index) => {
              if (index > 0) td.style.display = '';
            });
          });
      return;
    }

    // Filter main results table
    const resultsTableBody = resultsTable.querySelector('tbody');
    if (resultsTableBody) {
      resultsTableBody.querySelectorAll('tr').forEach(row => {
        const nameCell = row.querySelector('td:nth-child(2)');
        if (nameCell) {
          const encoderName = nameCell.textContent.trim();
          row.style.display = regex.test(encoderName) ? '' : 'none';
        }
      });
    }

    // Filter detail tables
    document.querySelectorAll('[id^="details-table-"]').forEach(detailTable => {
      const headerCells = detailTable.querySelectorAll('thead th');
      const rows = detailTable.querySelectorAll('tbody tr');

      headerCells.forEach((headerCell, index) => {
        if (index === 0) return;  // Skip "Metric" column

        const encoderName = headerCell.textContent.trim();
        const shouldShow = regex.test(encoderName);
        headerCell.style.display = shouldShow ? '' : 'none';

        // Hide corresponding data cells in each row
        rows.forEach(row => {
          const dataCell = row.querySelector(`td:nth-child(${index + 1})`);
          if (dataCell) {
            dataCell.style.display = shouldShow ? '' : 'none';
          }
        });
      });
    });

    renderSpiderChart();
  });

  renderSpiderChart();
});

/**
 * Renders a spider (radar) chart on the canvas based on the visible table data.
 */
function renderSpiderChart() {
  const canvas = document.getElementById('spider-chart');
  if (!canvas) return;

  const table = document.getElementById('results-table');
  if (!table) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = (rect.height || 500) * dpr;
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height || 500;

  // Positioning the graph more to the left
  const radius = Math.min(width / 4, 150);
  const centerX = radius + 110;  // Space for labels on the left
  const centerY = radius + 60;   // Space for labels on top

  // 1. Extract Data
  const headers = Array.from(table.querySelectorAll('thead th')).slice(2);
  const labels =
      headers.map(th => th.textContent.replace(' (mean)', '').trim());
  const numAxes = labels.length;

  const visibleRows = Array.from(table.querySelectorAll('tbody tr'))
                          .filter(row => row.style.display !== 'none')
                          .slice(0, 5);  // Show top 5 visible encoders

  const datasets = visibleRows.map(row => {
    const name = row.querySelector('td:nth-child(2)').textContent.trim();
    const scores = Array.from(row.querySelectorAll('td')).slice(2).map(td => {
      const val = parseFloat(td.textContent);
      return isNaN(val) ? 0 : val;
    });
    return {name, scores};
  });

  if (numAxes === 0) return;

  ctx.clearRect(0, 0, width, height);

  // 2. Draw Grid (Circles/Polygons)
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  for (let i = 1; i <= 5; i++) {
    const r = (radius / 5) * i;
    ctx.beginPath();
    for (let j = 0; j < numAxes; j++) {
      const angle = (Math.PI * 2 * j) / numAxes - Math.PI / 2;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      if (j === 0)
        ctx.moveTo(x, y);
      else
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();

    // Score labels (0.2, 0.4, ...)
    ctx.fillStyle = '#9aa0a6';
    ctx.font = '10px Roboto';
    ctx.fillText((0.2 * i).toFixed(1), centerX + 5, centerY - r + 10);
  }

  // 3. Draw Axes and Labels
  ctx.strokeStyle = '#bdbdbd';
  labels.forEach((label, i) => {
    const angle = (Math.PI * 2 * i) / numAxes - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = '#3c4043';
    ctx.font = 'bold 12px Roboto';
    const labelRadius = radius + 25;
    const lx = centerX + labelRadius * Math.cos(angle);
    const ly = centerY + labelRadius * Math.sin(angle);

    ctx.textAlign = 'center';
    if (Math.cos(angle) > 0.1)
      ctx.textAlign = 'left';
    else if (Math.cos(angle) < -0.1)
      ctx.textAlign = 'right';

    ctx.textBaseline = 'middle';
    if (Math.sin(angle) > 0.1)
      ctx.textBaseline = 'top';
    else if (Math.sin(angle) < -0.1)
      ctx.textBaseline = 'bottom';

    ctx.fillText(label, lx, ly);
  });

  // 4. Draw Data
  const colors = [
    'rgba(26, 115, 232, 0.7)',  // Google Blue
    'rgba(217, 48, 37, 0.7)',   // Google Red
    'rgba(249, 171, 0, 0.7)',   // Google Yellow
    'rgba(30, 142, 62, 0.7)',   // Google Green
    'rgba(161, 66, 244, 0.7)',  // Purple
  ];

  datasets.forEach((dataset, i) => {
    const color = colors[i % colors.length];
    ctx.strokeStyle = color.replace('0.7', '1');
    ctx.fillStyle = color;
    ctx.lineWidth = 2;

    ctx.beginPath();
    dataset.scores.forEach((score, j) => {
      const angle = (Math.PI * 2 * j) / numAxes - Math.PI / 2;
      const r = radius * Math.min(score, 1.0);  // Cap at 1.0
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      if (j === 0)
        ctx.moveTo(x, y);
      else
        ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
  });

  // 5. Draw Legend
  const legendX = 20;
  let legendY =
      centerY + radius + 60;  // Position below the graph and its labels
  datasets.forEach((dataset, i) => {
    const color = colors[i % colors.length];
    ctx.fillStyle = color.replace('0.7', '1');
    ctx.fillRect(legendX, legendY, 15, 15);
    ctx.fillStyle = '#3c4043';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = '12px Roboto';
    ctx.fillText(dataset.name, legendX + 20, legendY + 8);
    legendY += 20;
  });
}
