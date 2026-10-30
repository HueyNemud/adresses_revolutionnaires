# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pipeline turning scanned historical Paris directories (annuaires, early 1800s) into structured entries (ENTRY) nested under a title hierarchy (TITLE), splitting each entry into NER spans `SUBJ` / `DESC` / `ADDR`, exporting each volume as two linked JSON files, and aligning the entries of two editions. Code, comments, docstrings, CLI help and docs are in **French**; command and option names are in **English**; keep new code consistent with that.

## Commands

Managed with `uv` (Python ≥ 3.12); the package is `src/numrev/` (hatchling), installed by `uv sync` with the single entry point `numrev`; optional group `ocr` = latynna (`uv sync --group ocr`, editable path source `../finetuning_chandra_for_directories/packages/latynna`, imported only in `numrev ocr`'s `run`). Run everything from the repo root (data paths are relative). Tests use **unittest**, not pytest. Format with **black** then lint with **ruff** (line length 140; flake8's default rules `E`/`W`/`F` + import sorting `I`; flake8 itself is deliberately not used) before committing code.

```bash
uv sync                                                    # deps + the `numrev` command
uv run numrev                                              # list of commands (pipeline steps 0–8, then tools)
uv run numrev status [annuaires/<volume>]                  # per range: steps done / missing / stale + next command; volume export state
uv run numrev <command> --help
uv run python -m unittest discover -s tests                # all tests (one module: tests.test_tabulate; one test: tests.test_tabulate.<Class>.<test>)
uv run black src tests && uv run ruff check src tests [--fix]

# Steps 0–5 take a range annuaires/<volume>/<first>-<last> or a whole volume annuaires/<volume> (ocr and label: one range)
uv run latynna serve --detach && uv run latynna serve --wait   # OCR server (GPU machine)
uv run numrev ocr      annuaires/<volume>/<range> --apply  # 0 <volume>.pdf → <doc>.ocr.json
uv run numrev extract  annuaires/<volume> --apply          # 1 → <doc>.lines.json
uv run numrev label    annuaires/<volume>/<range>          # 2 → <doc>.labeled.json (+ .label-session.json); interactive, no --apply
uv run numrev tabulate annuaires/<volume> --apply          # 3 → <doc>.lines.csv (hand-curated)
uv run numrev assemble annuaires/<volume> --apply          # 4 → <doc>.entities.csv (+ .entities.report.txt)
uv run numrev tag      annuaires/<volume> --apply          # 5 → <doc>.ner.csv (hand-curated); model paths.NER_MODEL unless --model
uv run numrev export   annuaires/<volume> --apply          # 6 → <volume>.document.json + .structure.json (+ .export.report.txt)
uv run numrev align nw annuaires/<A> annuaires/<B> --apply                 # 7 order-based (Needleman-Wunsch + pair-HMM)
uv run numrev align dedupe annuaires/<A> annuaires/<B> [--label|--patch-only] --apply   # 7 Dedupe + entry patch
uv run numrev join annuaires/alignments/<A>__<B>.nw.csv [--excel] --apply  # 8 readable joined CSV
uv run numrev view directory|ocr|alignment [folder]        # viewers (read-only)
uv run numrev gold ner|alignment …  /  train-set  /  train data/ner/train.ls.json  /  audit crf|ner|alignment …
uv run --group explorations jupyter nbconvert --to notebook --execute --inplace explorations/<nb>.ipynb
```

## Documentation

Developer reference (French): `docs/pipeline.md` — every step, its files and design rules, and the code layout; read it before changing a step. `README.md` is a short overview. User docs (French, no code knowledge assumed, no code identifiers; keep them in sync with behaviour): `docs/guide_traitement.md` (a volume from PDF to deliverables), `docs/guide_annotation_lignes.md`, `docs/guide_curation.md`, `docs/guide_annotation_ner.md`, `docs/guide_relecture_alignement.md`, `docs/donnees.md` (deliverable columns for data users). `docs/format_export.md` is the spec of record of step 6: keep it, the JSON Schemas (`src/numrev/export/schemas/`, `$defs` header duplicated and kept identical — a test checks it; bump `schema` = `numrev-document/N` / `numrev-structure/N` on a non-additive change), the code and `viewers/directory.html` in sync. `docs/alignement_ordonne.md`: method and evaluation of `align nw`.

## Architecture

**Layout (`src/numrev/`).** `cli.py` (tables `PIPELINE` / `TOOLS` of `Command(module, summary, files)`, French help built by `build_parser`; a module is imported only when its command runs, so `numrev --help` never loads torch/Dedupe/Streamlit); `pipeline/` (one module per step 0–8); `viewers/` (`numrev view`: Streamlit `alignment.py` launched by `streamlit run`; standalone pages `directory.html` and `ocr.html` served with their files on a local `http.server` that answers `Range` requests); `devtools/` (`status`, golds, NER training set and training, audits); domain libraries `crf/`, `ner/`, `export/`, `alignment/`; shared `command.py`, `paths.py`, `curation.py`, `document.py`, `titles.py`, `stats.py`, `reporting.py`. A **step** transforms volume data on the way to the deliverables; viewers, golds, training and audits are **tools**. Tests import the package (`from numrev.pipeline.tabulate import …`); `tests/test_cli.py` builds every command's parser.

**Command convention (`command.py`).** Each command module exposes `add_arguments(parser)`, `run(args)` and `DESCRIPTION` (plain-French help). An expected failure raises `CommandError` (also from `require_file`, `require_dir`, `require_ranges`, `require_volume`, `require_input` — which names the previous command); `cli.main` prints `Erreur : …` and exits 1, and does the same for `CurationConflict`: commands don't catch these. **Keep the CLI simple**: common use needs few arguments; expert options go in the single `fine_tuning(parser)` group. Steps find input and output through `paths.py` — no `-o`, no path-override options (`-o/--output` only for audit report folders and the `train` model folder). Option names mean the same everywhere: `--model` = a trained model folder (`train` takes `--base-model`), `--force` = override a refusal (conflict, orphan, existing gold or OCR), `--seed`, `--bootstrap`, `--threshold`, `--min-score`, `--batch-size`, `--margin`.

**Dry run by default.** Every command that writes data computes and checks everything, lists the files it would write and writes nothing; `--apply` writes. Route every write through `Writes.add(path, fn)` and end with `writes.finish(console)`: writes are deferred to `finish` (all or nothing; bind loop variables in closures), except `add(…, now=True)` for human work a later failure must not lose (Dedupe console labels). Library functions take an optional `writes` and use `write_or_plan` (None = write now, for tests). Exceptions: `label` saves its session after each action and exports at each stop; `ocr` without `--apply` checks only; `train` without `--apply` stops before training. Audits and viewers have no `--apply`.

**File names (`paths.py`, the single place for them).** A *document* is a page range `<volume>.<range>` (no dots in names) in `annuaires/<volume>/<range>/`, ranges named `<first>-<last>` (1-based PDF pages); each step appends its suffix (`STEP_SUFFIXES`); volume deliverables are `volume_file(volume_dir, suffix)`. `document_name` strips a known suffix (unknown = `ValueError`), `step_path`, `range_document`, `selected_ranges`, `discover(root, suffix)` (exactly `<root>/<volume>/<range>/<volume>.<range><suffix>`, sorted: sampling depends on it). `Pair` holds a pair's files (`data/alignment/<A>__<B>.patch.csv` / `.sections.csv` / `.training.json` / `.gold-inversions.csv`, outputs `annuaires/alignments/<A>__<B><suffix>`).

**Replayable human corrections (`curation.py`; user guide `docs/guide_curation.md`, keep in sync).** A curated step's output CSV (lines: step `lines`; NER: step `ner`) is both machine output and the hand-edited file; rows have a stable key, editable fields, `corrige` and `empreinte`. Each run: capture `corrige = oui` rows into the versioned `data/curation/<document>.<step>.patch.csv`, generate, apply the patch (patch wins); all or nothing — `CurationConflict` on an edited row without `corrige`, a correction that no longer applies, or a machine line erased or moved. `--force` takes the machine output for those rows; `--no-capture` restarts from the patch. Line key `cle` = 12-hex sha1 of the frozen-normalized OCR text (`normalize_text`: never change it) + `~n` from the 2nd identical text; added rows get `<key>+n` (`debut+n` before the first line, with position `apres`), copy their neighbour's columns except `Step.machine_only`; delete = class `SUPPRIMÉE`; move = `SUPPRIMÉE` + keyless copy. Entity `uuid` = uuid5(document name + `cle` of the root line). Alignment patches follow the same rules (patch wins, orphans panic unless `--force`) but are hand-edited, not captured; their shared machinery (`PatchLine`, `Resolution`, `resolve_lines`: re-anchoring of lost uuids) is in `alignment/patch.py`. Data commits stay separate from code commits (prefix `données:`); commands never commit.

**Steps — invariants worth knowing** (details in `docs/pipeline.md`):

0. `ocr`: latynna (Chandra 2 fine-tuned, latynna-v4, vLLM) writes Chandra JSON where every line is one `<span>` with `data-word-confidence` and `data-line-bbox-estimate`; `page_index` rewritten to the 1-based PDF page (the only 0-based value is the latynna call). latynna-v4 writes braces inline (`… }` / `{ …`); only step 6 interprets them.
1. `extract`: one `<span>` = one line; `uid` = `page.block.line` (1-based); `word_confidence` dropped, never shifted, if counts differ; visible text outside any `<span>` is a `ValueError` naming page, block and excerpt. The pages → `data_blocks` → `lines` JSON shape is read and validated only in `document.py` (`load_pages`, `iter_line_locations`).
2. `label`: python-crfsuite active learning (`crf/active_learning.py`; `pipeline/label.py` is only the Rich UI and sessions); session tied to the sha256 of the `.lines.json`; probability = posterior-marginal argmax (not Viterbi). Changing `crf/features.py` `PRODUCTION_GROUPS` changes the annotator: try a feature as a candidate group and run `audit crf` instead.
3. `tabulate`: `confiance` / `douteuse` (`OCR_DOUBTFUL_THRESHOLD` 0.9, to recalibrate for v4) are not editable and outside the fingerprint; `class_check` refuses a correction whose class is empty or not in `crf.labels.CLASSES` + `SUPPRIMÉE`.
4. `assemble`: two open tracks (ENTRY, TITLE) that survive `OUT OF SCOPE` lines until a new `B-` root; `parent_uuid` from `#` levels, `ROOT_UUID` (nil uuid) for top titles and rows before any title. The tree is per range: entries before a range's first title get no rubric and are never attached to the previous range (often not contiguous); the remedy is a title added by hand at the top of the lines CSV.
5. `tag`: GLiNER on normalized text, spans mapped back to the raw `markdown`; `ner_suspect` reasons are structural (`ner/suspicion.py`), no corpus lexicon; corrections captured before the model loads, re-applied by `uuid`.
6. `export`: pure derived step (`pipeline/export.py` I/O, `export/layers.py` pure building, `export/braces.py` brace groups); refuses a range with a missing or stale step; compact JSON, all boxes in latynna's 0–1000 frame; provenance = numrev commit, date, sha256 of inputs and PDF.
7. `align nw` / `align dedupe`: entries are only ever matched between paired sections (`alignment/sections.py`, section patch, `restrict_to_corresponding`); runs without rubric are never aligned. `align nw` never applies the entry patch to its output but loads, checks and re-anchors it like `align dedupe` (`alignment/pair.load_entry_patch`). Review reasons and the ordinal uncertainty `faible` / `moyenne` / `forte` (`alignment/review.py`) are computed after the fact and never change the alignment; keep the three labels on that single axis. Dedupe 3.0.3 needs `btrees<6`. For a new pair, label a small inversion gold and run `audit alignment` before trusting thresholds.
8. `join`: natural-order join (`alignment/export.py`, shared with the viewer), entry patch applied in memory, never rewritten.

Test the alignment viewer with `streamlit.testing.v1.AppTest` or a wrapper that overrides `numrev.paths.ALIGNMENTS_DIR` / `ALIGNMENT_DATA_DIR` / `ANNUAIRES_DIR`, never in the real folders. The line classes (steps 2–4, `crf/labels.py`, read by `assemble`) and the span classes `SUBJ/DESC/ADDR` (step 5) are separate label spaces. Audits: `audit ner` evaluates on the reviewed gold `data/ner/gold.ls.json` (`dev`/`test` split fixed — never tune on `test`; hand-corrected NER rows are silver, not ground truth; exclude training data by normalized text, not uid); `audit crf` uses the hand-corrected `.lines.csv` as silver truth with observations from the sibling `.lines.json`. The git tag `avant-menage` keeps deleted scripts (`git show avant-menage:<file>`).

## Code style

Clarity first: short, explicit French docstrings and descriptive function names. Error messages state what was observed (where, what), never a guessed cause (e.g. `page 12, bloc 3 : texte hors de toute balise <span> : « … »`). Numbers shown to readers (pages, blocks, lines, `uid`) are 1-based; convert to 0-based only at the boundary of an external API, with a comment.

## Data

`annuaires/`, `*.pdf`, `*.gpkg` and `uv.lock` are git-ignored. Range outputs live next to their inputs, volume deliverables in `annuaires/<volume>/`, alignment outputs in `annuaires/alignments/`; human decisions in versioned `data/` (`curation/`, `alignment/`, `ner/`); audit reports in `reports/`; exploratory notebooks (committed with outputs) in `explorations/`.

**Format changes: ask, don't default to backward compatibility.** Before a change that alters the shape of data files (JSON/CSV columns, keys, uuids, file names, patch formats), ask the user which way to go: (a) keep the code compatible with the old files, or (b) migrate the existing data once (a one-off script, dry run first, backup of the git-ignored `annuaires/` files it touches) and make the code accept **only** the new format (fail loudly on old files, no fallback branches). Default recommendation: (b). After a migration, delete the one-off script and the obsolete files/options it made unnecessary, and remove "formerly…" / "older file" mentions from code, docs and this file.
