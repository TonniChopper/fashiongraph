// Tiny, safe markdown-lite → HTML for chat + card prose.
// Escapes first, then renders **bold**, *italic*, `code`, bullet/numbered lists,
// and paragraph breaks. No dependencies, no dangerouslySet-anything-unescaped.
export function mdToHtml(src = "") {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) =>
    esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>");

  const lines = String(src).replace(/\r/g, "").split("\n");
  const out = [];
  let list = null; // 'ul' | 'ol'

  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const head = line.match(/^\s*(#{1,4})\s+(.*)/);
    const bullet = line.match(/^\s*[-•]\s+(.*)/);
    const num = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (head) {
      closeList();
      out.push(`<div class="md-h">${inline(head[2])}</div>`);
    } else if (bullet) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(bullet[1])}</li>`);
    } else if (num) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(num[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("");
}
