// Vue « Documents » (numrev/viewers/context.py) : deux colonnes, chacune dans
// l'ordre de son annuaire, et une gouttière SVG qui relie les entrées
// appariées. Les liens qui se croisent (inversions) sont en orange.
// Événements renvoyés à Python : `focus` (clic sur une entrée), `pair` (clic
// en mode « choisir le partenaire »), `more` (clic sur « ⋯ »).
// La fonction est rappelée à chaque changement de `data` : elle redessine
// tout, en gardant à l'écran la ligne courante si elle y était déjà.

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const STEP = 40;

function lineHtml(line, side, data) {
  const classes = ["ln", line.t];
  if (line.t === "title") classes.push(`l${line.l || 9}`);
  if (line.u === data.focus[side === "left" ? 0 : 1]) classes.push("focus");
  if (line.f) classes.push("hit");
  if (data.pairing && line.t === "entry") classes.push(line.e ? "eligible" : "dimmed");
  let dot = "";
  if (line.t === "entry" && line.s !== undefined) {
    const dotClasses = ["dot", line.s];
    if (line.d) dotClasses.push("local");
    else if (line.m) dotClasses.push("manual");
    const what = { pair: "appariée", candidate: "candidate non appariée", alone: "sans correspondance" }[line.s] || "";
    dot = `<span class="${dotClasses.join(" ")}" title="${esc(what)}"></span>`;
  }
  const task = line.k ? `<span class="task t${line.k}" title="tâche de relecture">!</span>` : "";
  const tag = line.d ? `<span class="tag">non enregistrée</span>` : "";
  const page = line.t === "entry" ? `<span class="pg">p. ${esc(line.p)}</span>` : "";
  const copy = line.t === "title" ? `<button class="copy-uuid" data-copy="${esc(line.u)}" title="Copier l’uuid de la rubrique">uuid</button>` : "";
  return `<div class="${classes.join(" ")}" data-uuid="${esc(line.u)}" data-side="${side}">${dot}${task}${page}${line.h}${tag}${copy}</div>`;
}

function column(lines, side, data) {
  const [before, after] = data.more[side];
  const top = before ? `<button class="ctx-more" data-dir="-1">⋯ ${STEP} lignes précédentes</button>` : "";
  const bottom = after ? `<button class="ctx-more" data-dir="1">⋯ ${STEP} lignes suivantes</button>` : "";
  return `<div class="ctx-col ${side}">${top}${lines.map((line) => lineHtml(line, side, data)).join("")}${bottom}</div>`;
}

function draw(root, data) {
  const grid = root.querySelector(".ctx-grid");
  const svg = root.querySelector(".ctx-gutter");
  if (!grid || !svg) return;
  const left = grid.querySelector(".ctx-col.left");
  const right = grid.querySelector(".ctx-col.right");

  // Aligne les deux lignes courantes à la même hauteur.
  left.style.paddingTop = right.style.paddingTop = "0px";
  const focusLeft = left.querySelector(".ln.focus");
  const focusRight = right.querySelector(".ln.focus");
  if (focusLeft && focusRight) {
    const delta = focusLeft.offsetTop - focusRight.offsetTop;
    if (delta > 0) right.style.paddingTop = `${delta}px`;
    else if (delta < 0) left.style.paddingTop = `${-delta}px`;
  }

  const box = grid.getBoundingClientRect();
  const x1 = left.getBoundingClientRect().right - box.left - 1;
  const x2 = right.getBoundingClientRect().left - box.left + 1;
  const dotY = (element) => {
    const dot = element.querySelector(".dot");
    const rect = (dot || element).getBoundingClientRect();
    return rect.top - box.top + rect.height / 2;
  };
  const rightByUuid = new Map();
  right.querySelectorAll(".ln.entry").forEach((element) => rightByUuid.set(element.dataset.uuid, element));
  const lines = new Map(data.left.map((line) => [line.u, line]));
  const rightLines = new Map(data.right.map((line) => [line.u, line]));

  const links = [];
  left.querySelectorAll(".ln.entry").forEach((element) => {
    const line = lines.get(element.dataset.uuid);
    if (!line || !line.x) return;
    const partner = rightByUuid.get(line.x);
    if (!partner) return;
    const partnerLine = rightLines.get(line.x) || {};
    links.push({ l: line.u, r: line.x, y1: dotY(element), y2: dotY(partner), kind: line.s,
                 manual: line.m || line.d || partnerLine.m || partnerLine.d });
  });
  // Inversions : deux liens dont l'ordre diffère d'un côté à l'autre.
  links.sort((a, b) => a.y1 - b.y1);
  let maxY2 = -Infinity;
  const crossing = new Set();
  links.forEach((link, index) => {
    if (link.y2 < maxY2) crossing.add(index);
    maxY2 = Math.max(maxY2, link.y2);
  });
  let minY2 = Infinity;
  for (let index = links.length - 1; index >= 0; index--) {
    if (links[index].y2 > minY2) crossing.add(index);
    minY2 = Math.min(minY2, links[index].y2);
  }

  const width = box.width, height = grid.scrollHeight;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  const bend = (x2 - x1) * 0.5;
  const parts = links.map((link, index) => {
    const classes = ["link"];
    if (link.kind === "candidate") classes.push("candidate");
    if (link.manual) classes.push("manual");
    if (crossing.has(index)) classes.push("cross");
    if (link.l === data.focus[0] && link.r === data.focus[1]) classes.push("focus");
    return `<path class="${classes.join(" ")}" d="M${x1},${link.y1} C${x1 + bend},${link.y1} ${x2 - bend},${link.y2} ${x2},${link.y2}"/>`;
  });
  // Partenaire hors de la fenêtre : flèche au bord de la gouttière.
  const stub = (column, x, anchor, lookup) => column.querySelectorAll(".ln.entry").forEach((element) => {
    const line = lookup.get(element.dataset.uuid);
    if (!line || !line.o) return;
    parts.push(`<text class="stub" x="${x}" y="${dotY(element) + 4}" text-anchor="${anchor}">${line.o < 0 ? "↑" : "↓"}</text>`);
  });
  stub(left, x1 + 6, "start", lines);
  stub(right, x2 - 6, "end", rightLines);
  svg.innerHTML = parts.join("");
}

// Position à l'écran (relative au conteneur défilant) des lignes visibles,
// pour garder en place, après le nouveau rendu, celle qui y était déjà.
function screenPositions(root) {
  const scroll = root.querySelector(".ctx-scroll");
  const positions = new Map();
  if (!scroll) return positions;
  const top = scroll.getBoundingClientRect().top;
  root.querySelectorAll(".ln[data-uuid]").forEach((element) => {
    positions.set(`${element.dataset.side}:${element.dataset.uuid}`, element.getBoundingClientRect().top - top);
  });
  return positions;
}

// Défile de `delta` pixels ; si le haut du contenu l'empêche (défilement
// négatif), un espace en tête de grille compense, pour que la ligne
// d'ancrage ne bouge pas à l'écran.
function scrollBy(scroll, delta) {
  const grid = scroll.querySelector(".ctx-grid");
  const target = scroll.scrollTop + delta;
  if (target < 0 && grid) {
    grid.style.marginTop = `${-target}px`;
    scroll.scrollTop = 0;
  } else {
    scroll.scrollTop = target;
  }
}

function place(root, data, previous) {
  const scroll = root.querySelector(".ctx-scroll");
  if (!scroll) return;
  const top = () => scroll.getBoundingClientRect().top;
  // 1. l'élément d'ancrage choisi au clic, 2. la ligne courante si elle était
  // déjà affichée, 3. sinon la ligne courante au tiers de la hauteur.
  const candidates = [];
  if (root.__anchor) candidates.push(root.__anchor);
  ["left", "right"].forEach((side, index) => { if (data.focus[index]) candidates.push(`${side}:${data.focus[index]}`); });
  for (const key of candidates) {
    const [side, uuid] = key.split(/:(.*)/s);
    const element = root.querySelector(`.ln[data-side="${side}"][data-uuid="${CSS.escape(uuid)}"]`);
    if (element && previous.has(key)) {
      scrollBy(scroll, element.getBoundingClientRect().top - top() - previous.get(key));
      root.__anchor = null;
      return;
    }
  }
  root.__anchor = null;
  const focus = root.querySelector(".ln.focus");
  if (focus) scrollBy(scroll, focus.getBoundingClientRect().top - top() - scroll.clientHeight / 3);
}

async function copyText(button) {
  try { await navigator.clipboard.writeText(button.dataset.copy); } catch (error) { /* presse-papiers indisponible */ }
  const label = button.textContent;
  button.textContent = "✓ copié";
  setTimeout(() => { button.textContent = label; }, 1200);
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data) return;
  let root = parentElement.querySelector(".ctx");
  if (!root) {
    root = document.createElement("div");
    root.className = "ctx";
    parentElement.appendChild(root);
    root.addEventListener("click", (event) => {
      const copy = event.target.closest("button.copy-uuid");
      if (copy) { event.stopPropagation(); copyText(copy); return; }
      const more = event.target.closest("button.ctx-more");
      if (more) {
        // Garde en place la première ligne visible du côté cliqué.
        const element = more.dataset.dir === "-1" ? more.nextElementSibling : null;
        if (element && element.dataset.uuid) root.__anchor = `${element.dataset.side}:${element.dataset.uuid}`;
        setTriggerValue("more", { dir: Number(more.dataset.dir) });
        return;
      }
      const element = event.target.closest(".ln.entry");
      if (!element) return;
      const current = root.__data;
      if (current && current.pairing) {
        if (!element.classList.contains("eligible")) return;
        setTriggerValue("pair", { side: element.dataset.side, uuid: element.dataset.uuid });
        return;
      }
      root.__anchor = `${element.dataset.side}:${element.dataset.uuid}`;
      root.querySelectorAll(".ln.focus").forEach((other) => other.classList.remove("focus"));
      element.classList.add("focus");
      setTriggerValue("focus", { side: element.dataset.side, uuid: element.dataset.uuid });
    });
  }
  const previous = screenPositions(root);
  const [leftTitle, rightTitle] = data.titles || ["gauche", "droite"];
  const height = typeof data.height === "number" ? `${data.height}px` : data.height;
  const banner = data.pairing
    ? `<div class="ctx-banner">Choisir le partenaire : cliquer l’entrée correspondante (les entrées estompées sont hors des rubriques appariées).</div>`
    : "";
  root.innerHTML =
    banner +
    `<div class="ctx-head"><div>${esc(leftTitle)}</div><div></div><div>${esc(rightTitle)}</div></div>` +
    `<div class="ctx-scroll${data.pairing ? " pairing" : ""}" style="height:${height}"><div class="ctx-grid">` +
    column(data.left, "left", data) +
    `<div></div>` +
    column(data.right, "right", data) +
    `<svg class="ctx-gutter"></svg></div></div>` +
    `<div class="ctx-legend"><span><i style="border-color:var(--pair)"></i>appariées</span>` +
    `<span><i style="border-color:var(--candidate);border-top-style:dashed"></i>candidate</span>` +
    `<span><i style="border-color:var(--manual)"></i>décision du patch</span>` +
    `<span><i style="border-color:var(--cross)"></i>inversion (liens croisés)</span>` +
    `<span><b class="task t1">!</b><b class="task t2">!</b> tâche de relecture</span>` +
    `<span>↑ ↓ partenaire hors de la fenêtre · clic : ligne courante</span></div>`;

  root.__data = data;
  const redraw = () => draw(root, root.__data);
  redraw();
  place(root, data, previous);
  requestAnimationFrame(redraw);
  if (!root.__observer) {
    root.__observer = new ResizeObserver(() => requestAnimationFrame(redraw));
    root.__observer.observe(root);
  }
}
