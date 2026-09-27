# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pipeline turning scanned historical Paris directories (annuaires, early 1800s) into structured entries (ENTRY) nested under a title hierarchy (TITLE), then splitting each entry into NER spans `SUBJ` / `DESC` / `ADDR`. Code, comments, CLI help and README are in **French**; keep new code consistent with that.

## Commands

Managed with `uv` (Python ≥ 3.12). No linter/formatter or pytest is configured.

```bash
uv sync                                                    # install deps into .venv
uv run python -m unittest discover -s tests                # all tests (unittest, not pytest)
uv run python -m unittest tests.test_export_lines_csv      # one test module
uv run python -m unittest tests.test_export_lines_csv.<Class>.<test_name>   # one test
./run_pipeline.sh annuaires/<dossier>                      # all steps on the first *.ocr.json in a folder
uv run streamlit run tools/display_directory.py            # viewer for a volume's final NER CSV (*.merged.ner[.curated].csv)
uv run align_directories.py annuaires/<A> annuaires/<B> [--label]   # Dedupe alignment of two editions
uv run streamlit run tools/display_alignment.py            # viewer for an alignment CSV (pairs + unmatched)
uv run audit_crf_features.py                               # CRF/feature audit on curated CSVs → rapports/audit_crf/
uv run tools/sample_ner_gold.py                            # (once) stratified NER gold sample → data/ner/gold.ls.json
uv run audit_ner.py [--model DIR] [--sweep]                # NER audit on the reviewed gold → rapports/audit_ner/
```

Tests import the top-level scripts as modules, so run them from the repo root. As of this writing several tests in `test_annotate_lines_crf.py` and `test_extract_chandra_lines.py` are stale (they expect older class names such as `TITLE`/`OUT_OF_SCOPE` and older `--notables` cell ordering) and fail.

## Architecture

OCR is **out of scope** for this repo: it is run locally with Chandra in a separate project. A volume's starting state is its OCR output, `<nom>.ocr.json` (per-page `markdown` + `html` with `data-block-id` blocks) and `<nom>.ocr.md`, e.g. in `annuaires/1808_AD75-PER292/`.

Each step is a standalone CLI script at the repo root (argparse, `rich` console output, `-o/--output` defaulting to a name derived from the input), chained by file:

1. `extract_chandra_lines.py` — parses each block's HTML into Markdown lines; adds `data_blocks[].lines[]` (`uid`, `line_index`, `markdown`) to every page. `--notables` explodes table cells into separate lines.
2. `annotate_lines_crf.py` — interactive active-learning annotator (python-crfsuite). Labels lines `B-ENTRY`, `I-ENTRY`, `SUB-ENTRY`, `B-TITLE`, `I-TITLE`, `OUT OF SCOPE`, uncertain. Persists a resumable `.crf-session.json` after every action; exports each line's `prediction`/`provenance`/`probability`/`timestamp`. Probability is posterior-marginal argmax (not Viterbi) so it always matches the exported label.
3. `export_lines_csv.py` — flattens the JSON (annotated or raw) to one CSV row per line (`CSV_FIELDS`).
4. `build_entity_tree.py` (formerly `merge_annotated_lines.py`; outputs keep the `.merged` suffix) — two roles. (a) Rebuilds logical entities from line labels using two independent open "tracks" (ENTRY, TITLE) that stay open across `OUT OF SCOPE` / other-family lines until a new `B-` root, and gives each a deterministic `uuid` (uuid5 of document name + source line `uid`s, `#n` suffix for duplicate compositions; never the text). (b) Sets `parent_uuid` on every row (TITLE, ENTRY and OUT OF SCOPE): title level = number of leading `#` (no `#` = deepest level, flagged in the report); a TITLE's parent is the last preceding title of strictly lower level, any other row's the last preceding title; top-level titles and rows before any title get the nil UUID `ROOT_UUID` (single root shared by all documents). Writes `*.merged.csv` and a `*.merged.report.txt` (indented title tree with recursive and direct ENTRY counts per title — replaces the former `tools/entry_counter.py`; orphan continuations, titles without `#`, alphabetical-order breaks reset at each TITLE, unknown classes).
5. NER on merged ENTRY rows (conventions: `docs/guide_annotation_ner.md`):
   - `infer_gliner.py` — runs the trained model on a merged CSV, inserting `tagged_text` and per-class counts after the `entity` column. Requires `<model>/ner_config.json` (labels); feeds normalized text and maps spans back to the raw `markdown`. Also writes `ner_confidence` (min span score) and `ner_suspect` (review-priority reasons from `lib/ner/suspicion.py`: model score + generic structural checks, no corpus-specific lexicon); `audit_ner.py` reports the flag's coverage and recall of errors.
   - `tools/display_directory.py` — Streamlit viewer of a volume's final NER CSV (sections from heading levels, span colours, suspect filters, per-section stats).
   - `tools/build_ner_training.py` — builds the versioned training set `data/ner/train.ls.json` (needs local `annuaires/`) from the NER CSVs (`*.ner.curated.csv`, else `*.ner.csv`), gold texts excluded, shape-stratified (√) sampling. The CSVs must follow the annotation guide: a model relearns its data's convention errors.
   - `tools/train_gliner.py` — trains a GLiNER-bi model (run on the remote GPU machine) on one or more Label Studio JSON files (converts char spans → whitespace-token spans, auto-computes `max_width`); always on Markdown-normalized text; excludes gold texts, validates on a per-page split; writes `<model>/ner_config.json` (labels).
   - `autoclassify_labelstudio.py` — optional LLM pre-annotation (local Ollama, structured output) of a CSV into Label Studio predictions; offsets found by searching the text, altered entries logged as failures. Too costly for routine use.

6. `align_directories.py` — aligns the ENTRY rows of two **complete** directories with Dedupe `RecordLink` (one-to-one). `lib/alignment.py` loads a volume folder: its page-range subfolders sorted by first page, each read from its own `<volume>.<range>` + `CURATED_NER_SUFFIX` file (missing → error). Fields: `section` (ancestor title of level 2, else 1, via `parent_uuid`, so no inheritance across ranges; accent/punctuation-insensitive key), `subj` (SUBJ spans, Markdown-normalized), `text` (full normalized text); lowercased for Dedupe. Labels come from `dedupe.console_label` on first run or `--label`, saved to `data/alignement/<left>__<right>.training.json` (versioned) and reused. Output `annuaires/alignements/<left>__<right>.csv` (`left_file, left_uuid, right_uuid, right_file, score`, then snapshot columns for hand review: `left_section, right_section` = readable titles `Record.section_title`, `left_tagged_text, right_tagged_text`); unmatched entries are those absent from it. A hand-validated copy `<left>__<right>.curated.csv` is meant as input for later steps: only the id columns are authoritative, `score` may be empty for hand-added pairs (the viewer reads only id columns + score). `tools/display_alignment.py` reloads both volumes from the file names (prefix before the first `.` = volume folder). Dedupe 3.0.3 needs `btrees<6` (BTrees 6 dropped `byValue`); `prepare_training` takes a few minutes on ~17 k × 16 k entries.

**Shared code lives in `lib/`** (scripts at the repo root import it as `lib.…`, so run them from the root):
- `lib/chandra_document.py` — the pages → `data_blocks` → `lines` JSON shape, validated and iterated in one place (`iter_line_locations`), used by `annotate_lines_crf.py`, `export_lines_csv.py` and the CRF core. Change the schema there.
- `lib/stats.py` (page bootstrap, intervals, calibration, ROC AUC, review capture) and `lib/reporting.py` (Markdown helpers) are shared by both audits.
- `lib/alignment.py` — volume loading and Dedupe fields for step 6.
- `lib/ner/` — NER: spans and their representations (`spans.py`), HTML rendering of spans shared by both viewers (`html.py`), typographic shapes (`shapes.py`), NER CSV reader (`corpus.py`), gold metrics (`metrics.py`), review-priority reasons (`suspicion.py`), model loading/prediction (`gliner.py`).
- `lib/crf/` — CRF core. `features.py` defines named feature groups (`PRODUCTION_GROUPS` = exactly what the annotator uses, computed on Markdown-normalized text via `normalize_line` — emphasis markers, leading `#` and trailing spaces stripped, italic kept as the `italic` bucket feature; `LEGACY_GROUPS` reproduces the pre-v2 production set bit-for-bit for comparison (`production_v1` in the audit); `CANDIDATE_GROUPS` and `PLACEBO_GROUPS` are only evaluated by the audit). `model.py` wraps python-crfsuite training/marginals, `active_learning.py` holds `SourceLine`/`load_json_lines`/`ActiveCRF`; `annotate_lines_crf.py` keeps only the Rich UI, sessions and CLI.
- Changing production features changes annotator behaviour; to try a feature, add a candidate group and run the audit instead.

**CRF audit (`audit_crf_features.py`):** uses `*.ocr.lines.annotated.curated.csv` (column `prediction_curated`) as silver ground truth. `lib/crf/silver.py` takes observations from the sibling `*.ocr.lines.json` (what the annotator saw) and aligns curated labels by `uid` — curators also edited text (notably `#` heading markers), so computing features on the curated CSV text would leak labels. Curated labels differ from the original model predictions on only ~0.2 % of lines, so absolute scores are optimistic. `lib/crf/evaluation.py` runs page-contiguous within-document CV and leave-one-volume-out CV in parallel processes, with page-level (cluster) bootstrap; placebo feature groups give the noise floor used for verdicts.

**NER audit (`audit_ner.py`):** Ground truth is the hand-reviewed stratified, weighted gold `data/ner/gold.ls.json` (Label Studio; committed; `dev`/`test` split fixed — never tune on `test`). Hand-corrected `*.ner.curated.csv` files are only partly reviewed: silver, not ground truth. `audit_ner.py` only evaluates the systems it is given (`--model`, `--predictions`); on the dev split one `courant` entry weighs ~1.5 points, so also compare raw error counts. Annotations are compared on Markdown-normalized text (`normalize_markdown` + `project_spans`); join NER CSVs to merged CSVs on `uid`; exclude training data by normalized text, not uid. Scripts in `tools/` add the repo root to `sys.path` to import `lib`.

The line-level BIO classes (steps 2–4) and the span classes `SUBJ/DESC/ADDR` (step 5) are separate label spaces. If you rename a line class, update `build_entity_tree.py` too — unknown classes are treated as out-of-scope and only flagged in the report.

`tools/` holds the NER training and gold tools (`train_gliner.py`, `build_ner_training.py`, `sample_ner_gold.py`) and the result viewers (`display_directory.py`, `display_alignment.py`). `main.py` is a legacy one-off script with hardcoded filenames.

## Data

`annuaires/` (per-volume working folders), `*.pdf`, `*.gpkg` and `uv.lock` are git-ignored. Pipeline outputs live next to their inputs in `annuaires/<volume>/`.
