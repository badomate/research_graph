// paper_pipeline UI: KaTeX rendering + fast keyboard review.

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
// Re-render after any HTMX swap (verify/reject/edit replace card fragments).
document.body.addEventListener("htmx:afterSwap", (e) => renderMath(e.target));

// ── Keyboard-driven review ────────────────────────────────────────────────────
// j/k move focus, v verify, x reject, e edit. Ignored while typing in a field.
let focusIdx = -1;

function cards() {
  return Array.from(document.querySelectorAll(".concept-card"));
}
function focusCard(i) {
  const cs = cards();
  if (!cs.length) return;
  focusIdx = Math.max(0, Math.min(i, cs.length - 1));
  cs.forEach((c, n) => c.classList.toggle("focused", n === focusIdx));
  cs[focusIdx].scrollIntoView({ block: "center", behavior: "smooth" });
}
function clickIn(sel) {
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
    case "v": clickIn(".act-verify"); break;
    case "x": clickIn(".act-reject"); break;
    case "e": clickIn(".act-edit"); ev.preventDefault(); break;
    default: return;
  }
});

// Keep focus highlight on the card that was just swapped.
document.body.addEventListener("htmx:afterSwap", () => {
  const cs = cards();
  if (focusIdx >= 0 && focusIdx < cs.length) cs[focusIdx].classList.add("focused");
});
