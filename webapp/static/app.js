// paper_pipeline UI: KaTeX rendering, keyboard review, bulk-select helpers.

const KATEX_OPTS = {
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "\\[", right: "\\]", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
  ],
  throwOnError: false,
};
function renderMath(root) {
  if (window.renderMathInElement) {
    try { window.renderMathInElement(root || document.body, KATEX_OPTS); } catch (e) {}
  }
}
window.addEventListener("DOMContentLoaded", () => renderMath(document.body));
document.body.addEventListener("htmx:afterSwap", (e) => { renderMath(e.target); updatePickCount(); });

// ── Bulk selection ────────────────────────────────────────────────────────────
function picks() { return Array.from(document.querySelectorAll(".rowpick")); }
function updatePickCount() {
  const el = document.getElementById("pick-count");
  if (el) el.textContent = picks().filter((p) => p.checked).length + " selected";
}
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-pick]");
  if (!b) return;
  const mode = b.dataset.pick;
  picks().forEach((p) => {
    if (mode === "all") p.checked = true;
    else if (mode === "none") p.checked = false;
    else if (mode === "high") {
      const card = p.closest(".concept-card");
      p.checked = card && parseFloat(card.dataset.conf || "0") >= 0.8;
    }
  });
  updatePickCount();
});
document.addEventListener("change", (ev) => {
  if (ev.target.classList && ev.target.classList.contains("rowpick")) updatePickCount();
});

// ── Keyboard-driven review ──────────────────────────────────────────────────────
let focusIdx = -1;
function cards() { return Array.from(document.querySelectorAll(".concept-card")); }
function focusCard(i) {
  const cs = cards();
  if (!cs.length) return;
  focusIdx = Math.max(0, Math.min(i, cs.length - 1));
  cs.forEach((c, n) => c.classList.toggle("focused", n === focusIdx));
  cs[focusIdx].scrollIntoView({ block: "center", behavior: "smooth" });
}
function actIn(sel) {
  const cs = cards();
  if (focusIdx < 0 || focusIdx >= cs.length) return;
  const btn = cs[focusIdx].querySelector(sel);
  if (btn) btn.click();
}
document.addEventListener("keydown", (ev) => {
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  switch (ev.key) {
    case "j": focusCard(focusIdx + 1); break;
    case "k": focusCard(focusIdx <= 0 ? 0 : focusIdx - 1); break;
    case "v": actIn(".act-verify"); break;
    case "x": actIn(".act-reject"); break;
    case "e": actIn(".act-edit"); ev.preventDefault(); break;
    case " ": {
      const cs = cards();
      if (focusIdx >= 0 && focusIdx < cs.length) {
        const p = cs[focusIdx].querySelector(".rowpick");
        if (p) { p.checked = !p.checked; updatePickCount(); ev.preventDefault(); }
      }
      break;
    }
    default: return;
  }
});

window.addEventListener("DOMContentLoaded", updatePickCount);
