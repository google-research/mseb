

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
  });
});
