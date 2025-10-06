

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

  // Update rank column
  const numRows = tBody.querySelectorAll('tr').length;
  tBody.querySelectorAll('tr').forEach((row, index) => {
    row.querySelector('td:first-child').textContent =
        asc ? numRows - index : index + 1;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const resultsTable = document.getElementById('results-table');
  if (!resultsTable) return;

  // Sorting listeners
  resultsTable.querySelectorAll('th').forEach((headerCell, headerIndex) => {
    // Don't sort by Rank, Name, or Mean columns. Only sort score columns.
    if (headerIndex > 1 && !headerCell.classList.contains('toggle-mean')) {
      headerCell.addEventListener('click', () => {
        const tableElement = headerCell.closest('table');
        const currentAsc =
            headerCell.getAttribute('data-sort-direction') === 'asc';
        sortTable(tableElement, headerIndex, !currentAsc);
      });
    }
  });

  resultsTable.querySelectorAll('.sort-icon').forEach(sortIcon => {
    sortIcon.addEventListener('click', (e) => {
      const headerCell = sortIcon.closest('th');
      const headerIndex =
          Array.from(headerCell.parentNode.children).indexOf(headerCell);
      const tableElement = headerCell.closest("table");
      const currentAsc = headerCell.getAttribute("data-sort-direction") === "asc";
      sortTable(tableElement, headerIndex, !currentAsc);
      e.stopPropagation();  // prevent toggle listener on parent th
    });
  });

  // Filter listener
  const filterInput = document.getElementById('encoder-filter');
  const tableBody = resultsTable.querySelector('tbody');
  if (filterInput && tableBody) {
    filterInput.addEventListener('input', () => {
      const filterValue = filterInput.value;
      let regex;
      try {
        regex = new RegExp(filterValue, 'i');
      } catch (e) {
        // Invalid regex. Show all rows and skip filtering.
        tableBody.querySelectorAll('tr').forEach(row => row.style.display = '');
        return;
      }
      tableBody.querySelectorAll('tr').forEach(row => {
        const nameCell = row.querySelector('td:nth-child(2)');
        if (nameCell) {
          const encoderName = nameCell.textContent.trim();
          row.style.display = regex.test(encoderName) ? '' : 'none';
        }
      });
    });
  }

  // Hide task columns by default for collapsibility
  document.querySelectorAll('.task-col-header, .task-col-cell')
      .forEach(cell => {
        cell.classList.add('task-col-hidden');
      });

  // Collapsible column listeners
  document.querySelectorAll('.toggle-mean').forEach(toggleTh => {
    toggleTh.addEventListener('click', () => {
      const taskType = toggleTh.dataset.toggleTaskType;
      const icon = toggleTh.querySelector('.toggle-icon');
      const cellsToToggle = document.querySelectorAll(
          `.task-col-header[data-task-type="${taskType}"], ` +
          `.task-col-cell[data-task-type="${taskType}"]`);

      if (icon.textContent === '+') {
        icon.textContent = '-';
        cellsToToggle.forEach(cell => cell.classList.remove('task-col-hidden'));
      } else {
        icon.textContent = '+';
        cellsToToggle.forEach(cell => cell.classList.add('task-col-hidden'));
      }
    });
  });
});
