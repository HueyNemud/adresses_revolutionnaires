import json
import re
import sys
import html


def detect_warnings(text, results):
    """Détecte d'éventuelles anomalies ou incohérences dans les annotations."""
    warnings = []

    # Tri des entités
    sorted_res = sorted(results, key=lambda x: x["value"]["start"])

    for i, res in enumerate(sorted_res):
        val = res.get("value", {})
        labels = val.get("labels", [])
        ent_text = text[val.get("start", 0) : val.get("end", 0)]

        # Règle 3: Ponctuation / espace au début
        if re.match(r"^[\s,;.:!\-—–]+", ent_text):
            warnings.append("Début avec ponctuation/espace")

        # Règle 2: SUBJ se terminant par parenthèse courte
        if "SUBJ" in labels:
            match = re.search(r"(\s*\(\s*([^)]+)\s*\))\s*$", ent_text)
            if match:
                inner = match.group(2)
                alpha = re.search(r"[a-zA-ZÀ-ÿ]", inner)
                if alpha and alpha.group(0).islower():
                    warnings.append("SUBJ se termine par parenthèse (min)")

        # Règle 1: ADDR suivi de DESC commençant par tiret
        if "ADDR" in labels and i + 1 < len(sorted_res):
            next_res = sorted_res[i + 1]
            if "DESC" in next_res.get("value", {}).get("labels", []):
                between_and_desc = text[
                    val.get("end", 0) : next_res["value"].get("end", 0)
                ]
                if re.search(r"[\s\.,;:!\?]*[—–\-]", between_between := text[val.get("end", 0):next_res["value"].get("start", 0)]) or re.match(r"^[\s\.,;:!\?\-—–]*[—–\-]", next_res["value"].get("text", "")):
                    warnings.append("ADDR suivi de DESC avec tiret")

    return warnings


def build_highlighted_text(text, results):
    """Reconstruit le texte avec des balises HTML colorées pour chaque entité."""
    if not results:
        return html.escape(text)

    sorted_res = sorted(results, key=lambda x: x["value"]["start"])
    chunks = []
    last_idx = 0

    label_colors = {
        "SUBJ": "bg-blue-100 text-blue-900 border-blue-300",
        "ADDR": "bg-emerald-100 text-emerald-900 border-emerald-300",
        "DESC": "bg-amber-100 text-amber-900 border-amber-300",
        "DATE": "bg-purple-100 text-purple-900 border-purple-300",
        "PERS": "bg-pink-100 text-pink-900 border-pink-300",
    }

    for res in sorted_res:
        val = res.get("value", {})
        start = val.get("start", 0)
        end = val.get("end", 0)
        labels = val.get("labels", ["ENTITE"])

        if start > last_idx:
            chunks.append(html.escape(text[last_idx:start]))

        ent_text = html.escape(text[start:end])
        lbl_str = "/".join(labels)
        color_cls = label_colors.get(
            labels[0] if labels else "",
            "bg-gray-100 text-gray-800 border-gray-300",
        )

        badge = f'<mark class="{color_cls} px-1.5 py-0.5 rounded border text-xs font-mono inline-flex items-center gap-1 mx-0.5" title="{lbl_str} [{start}:{end}]"><span class="font-bold opacity-75 text-[10px] uppercase">{lbl_str}</span> {ent_text}</mark>'
        chunks.append(badge)
        last_idx = end

    if last_idx < len(text):
        chunks.append(html.escape(text[last_idx:]))

    return "".join(chunks)


def generate_html_viewer(json_data, output_filepath):
    parsed_items = []

    stats = {
        "total": len(json_data),
        "annotated": 0,
        "warnings": 0,
        "labels_count": {},
    }

    for idx, item in enumerate(json_data):
        item_id = item.get("id", idx + 1)
        full_text = item.get("data", {}).get("text", "")

        predictions = item.get("predictions", [])
        results = (
            predictions[0].get("result", [])
            if predictions
            else item.get("annotations", [{}])[0].get("result", [])
        )

        warnings = detect_warnings(full_text, results)
        highlighted = build_highlighted_text(full_text, results)

        entities_summary = []
        for r in results:
            val = r.get("value", {})
            lbls = val.get("labels", ["-"])
            for l in lbls:
                stats["labels_count"][l] = stats["labels_count"].get(l, 0) + 1
            entities_summary.append(
                {
                    "label": "/".join(lbls),
                    "text": val.get("text", full_text[val.get("start",0):val.get("end",0)]),
                    "start": val.get("start"),
                    "end": val.get("end"),
                }
            )

        if results:
            stats["annotated"] += 1
        if warnings:
            stats["warnings"] += 1

        parsed_items.append(
            {
                "id": item_id,
                "text_raw": full_text,
                "text_html": highlighted,
                "entities": entities_summary,
                "warnings": warnings,
                "has_entities": len(results) > 0,
            }
        )

    json_payload = json.dumps(parsed_items, ensure_ascii=False)
    stats_payload = json.dumps(stats, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="fr" class="h-full bg-slate-50">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualiseur d'Annotations JSON</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        mark {{ font-style: normal; }}
        body {{ font-family: system-ui, -apple-system, sans-serif; }}
        .table-container {{ max-height: calc(100vh - 220px); }}
    </style>
</head>
<body class="h-full flex flex-col text-slate-800">

    <!-- Header & Statistiques -->
    <header class="bg-white border-b border-slate-200 px-6 py-4 shadow-sm shrink-0">
        <div class="flex flex-wrap items-center justify-between gap-4">
            <div>
                <h1 class="text-xl font-bold text-slate-900 flex items-center gap-2">
                    📋 Visualiseur d'Annotations Label Studio
                </h1>
                <p class="text-xs text-slate-500 mt-0.5">Contrôle qualité et vérification rapide des annotations</p>
            </div>
            
            <!-- Cards Stats -->
            <div class="flex items-center gap-3 text-xs">
                <div class="bg-slate-100 border border-slate-200 px-3 py-1.5 rounded-lg text-center">
                    <span class="block text-slate-500 font-medium">Total Lignes</span>
                    <span class="text-base font-bold text-slate-800" id="stat-total">0</span>
                </div>
                <div class="bg-emerald-50 border border-emerald-200 px-3 py-1.5 rounded-lg text-center">
                    <span class="block text-emerald-600 font-medium">Annotées</span>
                    <span class="text-base font-bold text-emerald-700" id="stat-annotated">0</span>
                </div>
                <div class="bg-amber-50 border border-amber-200 px-3 py-1.5 rounded-lg text-center">
                    <span class="block text-amber-600 font-medium">Alertes / Anom.</span>
                    <span class="text-base font-bold text-amber-700" id="stat-warnings">0</span>
                </div>
            </div>
        </div>

        <!-- Toolbar de Recherche et Filtres -->
        <div class="mt-4 flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
            <div class="flex items-center gap-2 flex-1 min-w-[280px]">
                <input type="text" id="search-input" placeholder="Rechercher dans le texte ou les entités..." 
                       class="w-full max-w-md px-3 py-1.5 text-xs border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none">
            </div>

            <div class="flex items-center gap-2 text-xs">
                <span class="font-medium text-slate-600">Filtrer par :</span>
                <button onclick="setFilter('all')" id="btn-filter-all" class="filter-btn px-2.5 py-1 rounded-md bg-blue-600 text-white font-medium shadow-sm">Tous</button>
                <button onclick="setFilter('warnings')" id="btn-filter-warnings" class="filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200">⚠️ Alertes uniquement</button>
                <button onclick="setFilter('SUBJ')" id="btn-filter-subj" class="filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200">SUBJ</button>
                <button onclick="setFilter('ADDR')" id="btn-filter-addr" class="filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200">ADDR</button>
                <button onclick="setFilter('DESC')" id="btn-filter-desc" class="filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200">DESC</button>
                <button onclick="setFilter('empty')" id="btn-filter-empty" class="filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200">Non-annotés</button>
            </div>
        </div>
    </header>

    <!-- Content Table -->
    <main class="flex-1 overflow-hidden p-6">
        <div class="bg-white rounded-lg border border-slate-200 shadow-sm h-full flex flex-col">
            <div class="overflow-y-auto table-container flex-1">
                <table class="w-full text-left border-collapse text-xs">
                    <thead class="sticky top-0 bg-slate-100 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider z-10">
                        <tr>
                            <th class="py-2.5 px-3 w-16 text-center">#</th>
                            <th class="py-2.5 px-4">Texte Annoté</th>
                            <th class="py-2.5 px-4 w-72">Entités Extraites</th>
                            <th class="py-2.5 px-3 w-44 text-center">Statut / Alertes</th>
                        </tr>
                    </thead>
                    <tbody id="table-body" class="divide-y divide-slate-100 font-normal">
                        <!-- Rempli dynamiquement en JS -->
                    </tbody>
                </table>
            </div>

            <!-- Table Footer -->
            <div class="p-3 bg-slate-50 border-t border-slate-200 flex justify-between items-center text-xs text-slate-500">
                <span id="rendered-count">Affichage de 0 éléments</span>
                <span>Astuce: Survolez une entité pour voir ses offsets (start:end).</span>
            </div>
        </div>
    </main>

    <script>
        const RAW_DATA = {json_payload};
        const STATS = {stats_payload};

        let currentFilter = 'all';
        let searchQuery = '';
        let displayedCount = 0;
        const BATCH_SIZE = 150; // Affichage progressif ultra rapide
        let filteredData = [];

        // Initialisation des stats
        document.getElementById('stat-total').innerText = STATS.total.toLocaleString();
        document.getElementById('stat-annotated').innerText = STATS.annotated.toLocaleString();
        document.getElementById('stat-warnings').innerText = STATS.warnings.toLocaleString();

        function applyFilters() {{
            filteredData = RAW_DATA.filter(item => {{
                // Filtre par catégorie
                if (currentFilter === 'warnings' && item.warnings.length === 0) return false;
                if (currentFilter === 'empty' && item.has_entities) return false;
                if (['SUBJ', 'ADDR', 'DESC'].includes(currentFilter)) {{
                    const hasLabel = item.entities.some(e => e.label.includes(currentFilter));
                    if (!hasLabel) return false;
                }}

                // Filtre par recherche textuelle
                if (searchQuery.trim() !== '') {{
                    const q = searchQuery.toLowerCase();
                    const textMatch = item.text_raw.toLowerCase().includes(q);
                    const entityMatch = item.entities.some(e => e.text.toLowerCase().includes(q));
                    return textMatch || entityMatch;
                }}

                return true;
            }});

            // Reset affichage
            displayedCount = 0;
            document.getElementById('table-body').innerHTML = '';
            renderBatch();
        }}

        function renderBatch() {{
            const tbody = document.getElementById('table-body');
            const nextBatch = filteredData.slice(displayedCount, displayedCount + BATCH_SIZE);

            const fragment = document.createDocumentFragment();

            nextBatch.forEach((item, idx) => {{
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-50/80 transition-colors';

                // Badges d'alerte
                let warningBadges = '';
                if (item.warnings.length > 0) {{
                    warningBadges = item.warnings.map(w => `<span class="inline-block bg-amber-100 text-amber-800 border border-amber-300 px-1.5 py-0.5 rounded text-[10px] font-medium my-0.5">⚠️ ${{w}}</span>`).join(' ');
                }} else if (item.has_entities) {{
                    warningBadges = '<span class="text-emerald-600 font-medium text-[11px]">✓ Valide</span>';
                }} else {{
                    warningBadges = '<span class="text-slate-400 italic text-[11px]">Aucune entité</span>';
                }}

                // Entités liste
                const entList = item.entities.map(e => {{
                    let badgeColor = 'bg-gray-100 text-gray-700';
                    if (e.label.includes('SUBJ')) badgeColor = 'bg-blue-50 text-blue-700 border-blue-200';
                    if (e.label.includes('ADDR')) badgeColor = 'bg-emerald-50 text-emerald-700 border-emerald-200';
                    if (e.label.includes('DESC')) badgeColor = 'bg-amber-50 text-amber-700 border-amber-200';

                    return `<div class="truncate my-0.5">
                        <span class="inline-block px-1 py-0.2 text-[9px] font-bold uppercase rounded border ${{badgeColor}}">${{e.label}}</span>
                        <span class="text-slate-700 font-mono text-[11px]" title="${{e.text}}">${{e.text}}</span>
                    </div>`;
                }}).join('');

                tr.innerHTML = `
                    <td class="py-2 px-3 text-center text-slate-400 font-mono border-r border-slate-100">${{item.id}}</td>
                    <td class="py-2 px-4 leading-relaxed font-sans">${{item.text_html}}</td>
                    <td class="py-2 px-4 border-l border-r border-slate-100">${{entList || '<span class="text-slate-300">-</span>'}}</td>
                    <td class="py-2 px-3 text-center font-sans">${{warningBadges}}</td>
                `;

                fragment.appendChild(tr);
            }});

            tbody.appendChild(fragment);
            displayedCount += nextBatch.length;

            document.getElementById('rendered-count').innerText = `Affichage de ${{displayedCount.toLocaleString()}} sur ${{filteredData.length.toLocaleString()}} lignes filtrées`;
        }}

        // Infinite Scroll pour charger les lots suivants
        document.querySelector('.table-container').addEventListener('scroll', (e) => {{
            const {{ scrollTop, scrollHeight, clientHeight }} = e.target;
            if (scrollTop + clientHeight >= scrollHeight - 100) {{
                if (displayedCount < filteredData.length) {{
                    renderBatch();
                }}
            }}
        }});

        function setFilter(filterType) {{
            currentFilter = filterType;
            document.querySelectorAll('.filter-btn').forEach(btn => {{
                btn.className = 'filter-btn px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200';
            }});
            const activeBtn = document.getElementById(`btn-filter-${{filterType}}`);
            if (activeBtn) activeBtn.className = 'filter-btn px-2.5 py-1 rounded-md bg-blue-600 text-white font-medium shadow-sm';

            applyFilters();
        }}

        document.getElementById('search-input').addEventListener('input', (e) => {{
            searchQuery = e.target.value;
            applyFilters();
        }});

        // Lancement initial
        applyFilters();
    </script>
</body>
</html>
"""

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Visualiseur généré avec succès : {output_filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_viewer.py <fichier_annotations.json> [output.html]"
        )
        sys.exit(1)

    json_input = sys.argv[1]
    html_output = (
        sys.argv[2] if len(sys.argv) > 2 else "annotation_viewer.html"
    )

    with open(json_input, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_html_viewer(data, html_output)
