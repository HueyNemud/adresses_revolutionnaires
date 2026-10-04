"""Journal des décisions du viewer d'alignement gardé dans le `localStorage`
du navigateur : il survit à un rechargement de la page et à la mise en
veille de l'application (Streamlit Community Cloud), sans serveur.

Un composant `st.components.v2` invisible fait le pont :

- au montage (`data.journal` absent), il lit la clé et la renvoie à Python
  (état `restored` = {key, journal}) ;
- ensuite, il réécrit la clé avec le journal que Python lui passe (chaîne
  vide = clé effacée).

Python ne passe aucun journal avant la restauration de la clé courante :
le journal sauvegardé n'est jamais écrasé par un journal vide. Si le
stockage est indisponible (navigation privée, test sans navigateur), le
viewer fonctionne, sans persistance.
"""

import streamlit as st

STORAGE_JS = """
export default function ({ data, setStateValue }) {
  if (!data || !data.key) return;
  if (data.journal === null || data.journal === undefined) {
    let saved = "";
    try { saved = window.localStorage.getItem(data.key) || ""; } catch (error) { saved = ""; }
    setStateValue("restored", { key: data.key, journal: saved });
    return;
  }
  try {
    if (data.journal === "") window.localStorage.removeItem(data.key);
    else window.localStorage.setItem(data.key, data.journal);
  } catch (error) { /* stockage indisponible : pas de persistance */ }
}
"""

_storage = st.components.v2.component("numrev_journal_storage", js=STORAGE_JS)


def storage_key(pair_name: str) -> str:
    return f"numrev-alignement:{pair_name}"


def browser_journal(key: str, journal: str | None) -> str | None:
    """Monte le pont. `journal` : le journal à écrire, None tant que la clé
    n'est pas restaurée. Renvoie le journal sauvegardé pour `key` dès que le
    navigateur l'a lu ("" s'il n'y en a pas), sinon None."""
    result = _storage(key="journal_storage", data={"key": key, "journal": journal}, height=0, on_restored_change=lambda: None)
    restored = result.get("restored")
    if isinstance(restored, dict) and restored.get("key") == key:
        return restored.get("journal") or ""
    return None
