// Vue « Documents » (numrev/viewers/context.py) : deux colonnes, chacune dans
// l'ordre de son annuaire, et une gouttière SVG qui relie les entrées
// appariées. Les liens qui se croisent (inversions) sont en orange. Un clic
// sur une entrée la renvoie à Python (déclencheur `pick`).
// La fonction est rappelée à chaque changement de `data` : elle redessine tout.

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function lineHtml(line, side, focus, picks) {
  const classes = ["ln", line.t];
  if (line.t === "title") classes.push(`l${line.l || 9}`);
  if (line.u === focus[side === "left" ? 0 : 1]) classes.push("focus");
  if (picks && picks[side] === line.u) classes.push("picked");
  let dot = "";
  if (line.t === "entry" && line.s !== undefined) {
    const dotClasses = ["dot", line.s];
    if (line.d) dotClasses.push("local");
    else if (line.m) dotClasses.push("manual");
    const what = { pair: "appariée", candidate: "candidate non appariée", alone: "sans correspondance" }[line.s] || "";
    dot = `<span class="${dotClasses.join(" ")}" title="${esc(what)}"></span>`;
  }
  const tag = line.d ? `<span class="tag">non enregistrée</span>` : "";
  const page = line.t === "entry" ? `<span class="pg">p. ${esc(line.p)}</span>` : "";
  return `<div class="${classes.join(" ")}" data-uuid="${esc(line.u)}" data-side="${side}">${dot}${page}${line.h}${tag}</div>`;
}

function column(lines, side, focus, picks, more) {
  const top = more[0] ? `<div class="ctx-more">⋯ lignes précédentes</div>` : "";
  const bottom = more[1] ? `<div class="ctx-more">⋯ lignes suivantes</div>` : "";
  return `<div class="ctx-col ${side}">${top}${lines.map((line) => lineHtml(line, side, focus, picks)).join("")}${bottom}</div>`;
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

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data) return;
  let root = parentElement.querySelector(".ctx");
  if (!root) {
    root = document.createElement("div");
    root.className = "ctx";
    parentElement.appendChild(root);
    root.addEventListener("click", (event) => {
      const element = event.target.closest(".ln.entry");
      if (!element) return;
      root.querySelectorAll(`.ln.picked[data-side="${element.dataset.side}"]`).forEach((other) => {
        if (other !== element) other.classList.remove("picked");
      });
      element.classList.toggle("picked");
      setTriggerValue("pick", { side: element.dataset.side, uuid: element.dataset.uuid });
    });
  }
  const [leftTitle, rightTitle] = data.titles || ["gauche", "droite"];
  root.innerHTML =
    `<div class="ctx-head"><div>${esc(leftTitle)}</div><div></div><div>${esc(rightTitle)}</div></div>` +
    `<div class="ctx-scroll" style="height:${data.height}px"><div class="ctx-grid">` +
    column(data.left, "left", data.focus, data.picks, data.more.left) +
    `<div></div>` +
    column(data.right, "right", data.focus, data.picks, data.more.right) +
    `<svg class="ctx-gutter"></svg></div></div>` +
    `<div class="ctx-legend"><span><i style="border-color:var(--pair)"></i>appariées</span>` +
    `<span><i style="border-color:var(--candidate);border-top-style:dashed"></i>candidate</span>` +
    `<span><i style="border-color:var(--manual)"></i>décision du patch</span>` +
    `<span><i style="border-color:var(--cross)"></i>inversion (liens croisés)</span>` +
    `<span>↑ ↓ partenaire hors de la fenêtre · clic : sélectionner pour apparier</span></div>`;

  const redraw = () => draw(root, root.__data);
  root.__data = data;
  requestAnimationFrame(() => {
    redraw();
    const scroll = root.querySelector(".ctx-scroll");
    const focus = root.querySelector(".ln.focus");
    if (scroll && focus) scroll.scrollTop = Math.max(0, focus.offsetTop - scroll.clientHeight / 3);
  });
  if (!root.__observer) {
    root.__observer = new ResizeObserver(() => requestAnimationFrame(redraw));
    root.__observer.observe(root);
  }
}
