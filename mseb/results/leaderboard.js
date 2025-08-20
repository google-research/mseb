

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
