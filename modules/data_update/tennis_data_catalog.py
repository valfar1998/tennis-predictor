"""Catalogo tornei tennis-data.co.uk (estratto dal portale Buchdahl)."""

from __future__ import annotations

# Grand Slam + Masters 1000 — sync prioritario ATP/WTA
PRIORITY_SLUGS = (
    "ausopen",
    "frenchopen",
    "wimbledon",
    "usopen",
    "indianwells",
    "miami",
    "montecarlo",
    "madrid",
    "rome-tms",
    "montreal",
    "toronto",
    "cincinnati",
    "shanghai",
    "paris-tms",
)

# Slug pagina PHP → stem file CSV remoto (se diverso)
SLUG_FILE_MAP: dict[str, str] = {
    "frenchopen": "frenchopen",
    "rome-tms": "rome",
    "paris-tms": "paris",
}


def csv_stem(slug: str) -> str:
    return SLUG_FILE_MAP.get(slug, slug.replace("-tms", "").replace("-", ""))
