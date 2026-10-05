// Vue « Documents » (numrev/viewers/context.py) : deux colonnes, chacune dans
// l'ordre de son annuaire, et une gouttière SVG qui relie les entrées
// appariées. Les liens qui se croisent (inversions) sont en rouge.
// Événements renvoyés à Python : `focus` (clic sur une entrée ou un lien),
// `pair` (clic en mode « choisir le partenaire »), `more` (clic sur « ⋯ »),
// `zoom` (loupe de la ligne courante : vue Relecture).
// La fonction est rappelée à chaque changement de `data` : elle redessine
// tout, puis place la ligne courante (voir `place`).

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const STEP = 10;
const CENTER = 0.4;
const LOUPE_ICON = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" ` +
  `stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`; // hauteur, dans la fenêtre, où l'on place une nouvelle ligne courante

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
  const title = line.t === "title" ? ` data-title="${line.h}"` : ""; // déjà échappé
  return `<div class="${classes.join(" ")}" data-uuid="${esc(line.u)}" data-side="${side}"${title}>${dot}${task}${page}${line.h}${tag}${copy}</div>`;
}

function column(lines, side, data) {
  const [before, after] = data.more[side];
  // « ⋯ » ajoute quelques lignes à cette colonne ; « Page » la fait glisser
  // d'une page, l'autre colonne montrant tous les partenaires de la page
  // (sans changer la ligne courante).
  const nav = (dir, more, page) =>
    `<div class="ctx-nav"><button class="ctx-more" data-dir="${dir}">⋯ ${STEP} ${more}</button>` +
    `<button class="ctx-page" data-dir="${dir}">${page}</button></div>`;
  const top = before ? nav(-1, "lignes précédentes", "⇡ Page précédente") : "";
  const bottom = after ? nav(1, "lignes suivantes", "Page suivante ⇣") : "";
  return `<div class="ctx-col ${side}">${top}${lines.map((line) => lineHtml(line, side, data)).join("")}${bottom}</div>`;
}

// Fond sombre ou clair, d'après le thème Streamlit (clair / sombre).
function isDark(root) {
  const value = getComputedStyle(root).getPropertyValue("--st-background-color").trim();
  let rgb = null;
  if (/^#[0-9a-f]{6}$/i.test(value)) rgb = [1, 3, 5].map((i) => parseInt(value.slice(i, i + 2), 16));
  else {
    const match = value.match(/rgba?\(([^)]+)\)/);
    if (match) rgb = match[1].split(",").slice(0, 3).map(Number);
  }
  if (!rgb) return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 128;
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
                 manual: line.m || line.d || partnerLine.m || partnerLine.d, uncertain: line.q || partnerLine.q });
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
  let loupe = null; // position de la loupe : milieu du lien courant
  // Trait plein : paire sûre ; tirets : paire incertaine ; tirets rouges :
  // candidate non appariée. Gris : automatique ; foncé : décision humaine.
  // Une inversion est surlignée (halo jaune) sans changer son trait.
  const parts = links.map((link, index) => {
    const classes = ["link"];
    if (link.kind === "candidate") classes.push("candidate");
    else if (link.uncertain) classes.push("uncertain");
    if (link.manual) classes.push("manual");
    const current = link.l === data.focus[0] && link.r === data.focus[1];
    if (current) {
      classes.push("focus");
      loupe = { x: (x1 + x2) / 2, y: (link.y1 + link.y2) / 2 };
    }
    const path = `M${x1},${link.y1} C${x1 + bend},${link.y1} ${x2 - bend},${link.y2} ${x2},${link.y2}`;
    const halo = crossing.has(index) ? `<path class="halo" d="${path}"/>` : "";
    return `<g class="lk">${halo}<path class="${classes.join(" ")}" d="${path}"/>` +
      `<path class="hit" d="${path}" data-l="${esc(link.l)}" data-r="${esc(link.r)}"><title>Sélectionner cette paire</title></path></g>`;
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

  // Loupe : sur le lien courant, sinon à côté de l'entrée courante seule.
  if (!loupe && focusLeft && !focusRight) loupe = { x: x1 + 18, y: dotY(focusLeft) };
  if (!loupe && focusRight && !focusLeft) loupe = { x: x2 - 18, y: dotY(focusRight) };
  grid.querySelectorAll(".loupe").forEach((element) => element.remove());
  if (loupe && !data.pairing) {
    const button = document.createElement("button");
    button.className = "loupe";
    button.innerHTML = LOUPE_ICON;
    button.title = `Relire en détail${data.info ? " — " + data.info : ""}`;
    button.style.left = `${loupe.x}px`;
    button.style.top = `${loupe.y}px`;
    grid.appendChild(button);
  }
}

// Position à l'écran (relative au conteneur défilant) des lignes affichées.
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

// Défile de `delta` pixels. `spacer` : si le haut du contenu l'empêche, un
// espace en tête de grille compense (seulement après un clic, pour que
// l'élément cliqué ne bouge pas).
function scrollBy(scroll, delta, spacer) {
  const grid = scroll.querySelector(".ctx-grid");
  const target = scroll.scrollTop + delta;
  if (target < 0 && spacer && grid) {
    grid.style.marginTop = `${-target}px`;
    scroll.scrollTop = 0;
  } else {
    scroll.scrollTop = Math.max(0, target);
  }
}

// Place la nouvelle vue : 1. après un clic dans les documents (entrée,
// lien, « ⋯ »), l'élément cliqué ou la ligne voisine du bouton reste où il
// était ; 2. si ni la ligne courante ni la page n'ont changé (une
// décision), la vue ne bouge pas : la ligne courante, sinon la première
// ligne visible, reste où elle était ; 3. une nouvelle page s'affiche depuis
// le haut ; 4. sinon (tâche suivante, recherche, rubrique…), la ligne
// courante est centrée.
function place(root, data, previous, previousKey) {
  const scroll = root.querySelector(".ctx-scroll");
  if (!scroll) return;
  const top = () => scroll.getBoundingClientRect().top;
  const find = (key) => {
    const [side, uuid] = key.split(/:(.*)/s);
    return root.querySelector(`.ln[data-side="${side}"][data-uuid="${CSS.escape(uuid)}"]`);
  };
  const anchor = root.__anchor;
  root.__anchor = null;
  if (anchor && previous.has(anchor) && find(anchor)) {
    scrollBy(scroll, find(anchor).getBoundingClientRect().top - top() - previous.get(anchor), true);
    return;
  }
  if (viewKey(data) === previousKey) {
    const isVisible = (key) => previous.has(key) && previous.get(key) >= 0 && previous.get(key) <= scroll.clientHeight && find(key);
    const focused = ["left", "right"].map((side, index) => `${side}:${data.focus[index]}`);
    const kept = focused.find(isVisible) || [...previous.keys()].find(isVisible);
    if (kept) {
      scrollBy(scroll, find(kept).getBoundingClientRect().top - top() - previous.get(kept), false);
      return;
    }
  }
  if (data.page) {
    scroll.scrollTop = 0;
    return;
  }
  const target = root.querySelector(".ln.focus");
  if (target) scrollBy(scroll, target.getBoundingClientRect().top - top() - scroll.clientHeight * CENTER, false);
}

// Ligne courante et page : la vue n'est replacée que s'ils changent.
function viewKey(data) {
  return [...data.focus, data.page || ""].join("|");
}

// Rubrique courante de chaque colonne, à côté du nom de la liste : celle de
// l'entrée sélectionnée si elle est visible, sinon celle de la ligne en haut
// de la zone visible (mise à jour au défilement). Rubrique = dernier titre de niveau 1
// ou 2 au-dessus (comme `records.section_of`) ; au-dessus de la fenêtre,
// `data.sections` la donne.
function showSections(root, data) {
  const scroll = root.querySelector(".ctx-scroll");
  if (!scroll) return;
  const { top, bottom } = scroll.getBoundingClientRect();
  ["left", "right"].forEach((side) => {
    const label = root.querySelector(`.ctx-head .sec[data-side="${side}"]`);
    const column = root.querySelector(`.ctx-col.${side}`);
    if (!label || !column) return;
    let line = column.querySelector(".ln.focus");
    const box = line && line.getBoundingClientRect();
    if (!line || box.bottom < top || box.top > bottom) line = [...column.querySelectorAll(".ln")].find((element) => element.getBoundingClientRect().bottom > top + 4);
    let title = (data.sections || {})[side] || "";
    for (let element = line; element; element = element.previousElementSibling) {
      if (element.matches(".ln.title.l1, .ln.title.l2")) { title = element.dataset.title; break; }
    }
    label.innerHTML = title ? `<span class="sep">›</span>${title}` : ""; // déjà échappé
    label.title = label.textContent.replace(/^›/, "");
  });
}

async function copyText(button) {
  try { await navigator.clipboard.writeText(button.dataset.copy); } catch (error) { /* presse-papiers indisponible */ }
  const label = button.textContent;
  button.textContent = "✓ copié";
  setTimeout(() => { button.textContent = label; }, 1200);
}

// Z / S font défiler les documents (Q / D, tâche précédente / suivante, et
// les touches de décision sont des raccourcis Streamlit : ZQSD est le pavé
// de flèches d'un clavier AZERTY). Ignorées pendant une saisie ou avec une
// touche de modification (Ctrl+Z annule). Un seul écouteur pour la page ; il
// agit sur la vue la plus récente.
const SCROLL_KEYS = { z: -1, s: 1 };
const SCROLL_STEP = 90; // pixels par appui
function installScrollKeys(root) {
  window.__numrevDocuments = root;
  if (window.__numrevScrollKeys) return;
  window.__numrevScrollKeys = true;
  window.addEventListener("keydown", (event) => {
    const direction = SCROLL_KEYS[event.key.toLowerCase()];
    if (!direction || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.composedPath()[0];
    if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
    const current = window.__numrevDocuments;
    const scroll = current && current.isConnected && current.querySelector(".ctx-scroll");
    if (!scroll) return;
    event.preventDefault();
    scroll.scrollBy({ top: direction * SCROLL_STEP });
  });
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
      if (event.target.closest("button.loupe")) { setTriggerValue("zoom", { view: "detail" }); return; }
      const more = event.target.closest("button.ctx-more");
      const sideOf = (button) => (button.closest(".ctx-col.left") ? "left" : "right");
      if (more) {
        // La ligne voisine du bouton reste en place : les lignes ajoutées
        // apparaissent sans que la vue saute.
        const lines = more.closest(".ctx-col").querySelectorAll(".ln[data-uuid]");
        const neighbour = Number(more.dataset.dir) < 0 ? lines[0] : lines[lines.length - 1];
        if (neighbour) root.__anchor = `${neighbour.dataset.side}:${neighbour.dataset.uuid}`;
        setTriggerValue("more", { dir: Number(more.dataset.dir), side: sideOf(more) });
        return;
      }
      const page = event.target.closest("button.ctx-page");
      if (page) {
        setTriggerValue("page", { dir: Number(page.dataset.dir), side: sideOf(page) });
        return;
      }
      const current = root.__data;
      const link = event.target.closest("path.hit");
      if (link) {
        if (current && current.pairing) return;
        const element = root.querySelector(`.ctx-col.left .ln[data-uuid="${CSS.escape(link.dataset.l)}"]`);
        if (element) root.__anchor = `left:${link.dataset.l}`;
        setTriggerValue("focus", { side: "left", uuid: link.dataset.l });
        return;
      }
      const element = event.target.closest(".ln.entry");
      if (!element) return;
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
  const previousKey = root.__viewKey || "";
  root.dataset.scheme = isDark(root) ? "dark" : "light";
  const [leftTitle, rightTitle] = data.titles || ["gauche", "droite"];
  const height = typeof data.height === "number" ? `${data.height}px` : data.height;
  const banner = data.pairing
    ? `<div class="ctx-banner">Choisir le partenaire : cliquer l’entrée correspondante (les entrées estompées sont hors des rubriques appariées).</div>`
    : "";
  const span = (label) => `<mark class="span" title="${label}">${label}</mark>`;
  const legend =
    `<div class="ctx-legend">${span("SUBJ")}${span("DESC")}${span("ADDR")}` +
    `<span><i class="sure"></i>sûre</span>` +
    `<span><i class="unsure"></i>incertaine</span>` +
    `<span><i class="cand"></i>candidate non appariée</span>` +
    `<span><i class="sure"></i>automatique · <i class="sure human"></i>décision humaine</span>` +
    `<span><i class="sure inv"></i>inversion</span>` +
    `<span><b class="task t1">!</b><b class="task t2">!</b> tâche de relecture</span>` +
    `<span class="hint">clic : ligne courante · loupe : relire en détail · ↑ ↓ partenaire hors fenêtre</span></div>`;
  root.innerHTML =
    banner +
    legend +
    `<div class="ctx-head"><div>${esc(leftTitle)}<span class="sec" data-side="left"></span></div><div></div>` +
    `<div>${esc(rightTitle)}<span class="sec" data-side="right"></span></div></div>` +
    `<div class="ctx-scroll${data.pairing ? " pairing" : ""}" style="height:${height}"><div class="ctx-grid">` +
    column(data.left, "left", data) +
    `<div></div>` +
    column(data.right, "right", data) +
    `<svg class="ctx-gutter"></svg></div></div>`;


  root.__data = data;
  root.__viewKey = viewKey(data);
  installScrollKeys(root);
  const redraw = () => draw(root, root.__data);
  redraw();
  place(root, data, previous, previousKey);
  showSections(root, data);
  const scroll = root.querySelector(".ctx-scroll");
  let pending = false;
  scroll.addEventListener("scroll", () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; showSections(root, root.__data); });
  }, { passive: true });
  requestAnimationFrame(redraw);
  if (!root.__observer) {
    root.__observer = new ResizeObserver(() => requestAnimationFrame(redraw));
    root.__observer.observe(root);
  }
}
