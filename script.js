
const search = document.querySelector("#paper-search");
const filter = document.querySelector("#paper-filter");
const cards = [...document.querySelectorAll(".paper-card")];
const count = document.querySelector("#paper-count");

function applyFilters() {
  const query = (search.value || "").trim().toLowerCase();
  const selected = filter.value;
  let visible = 0;

  for (const card of cards) {
    const text = card.dataset.search || "";
    const tags = card.dataset.tags || "";
    const matchesQuery = !query || text.includes(query);
    const matchesTag = selected === "all" || tags.includes(selected);
    const show = matchesQuery && matchesTag;
    card.hidden = !show;
    if (show) visible += 1;
  }

  count.textContent = String(visible);
}

search?.addEventListener("input", applyFilters);
filter?.addEventListener("change", applyFilters);
applyFilters();

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", () => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (!target) return;
    target.setAttribute("tabindex", "-1");
    setTimeout(() => target.focus({ preventScroll: true }), 350);
  });
});
