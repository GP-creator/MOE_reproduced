"""Dashboard entry point (DASHBOARD_SPEC §2.2): ``make dashboard`` or
``.venv/bin/streamlit run dashboard/app.py``. Set ``MOE_RESULTS_DIR`` to read another results
tree (e.g. ``results/sample`` or a temp dir in tests); default ``<repo>/results``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:          # so `import dashboard.*` and `import moe.*` work
    sys.path.insert(0, str(_REPO))

import streamlit as st  # noqa: E402

from dashboard import layout, theme  # noqa: E402,F401  (theme import registers the Plotly templates)

st.set_page_config(page_title="MoE repro", layout="wide", initial_sidebar_state="expanded")

_PAGES = Path(__file__).resolve().parent / "pages"
pg = st.navigation({
    "Results": [st.Page(_PAGES / "overview.py", title="Overview", default=True),
                st.Page(_PAGES / "training.py", title="Training"),
                st.Page(_PAGES / "routing.py", title="Routing"),
                st.Page(_PAGES / "capacity.py", title="Capacity sweep (E2)"),
                st.Page(_PAGES / "ablations.py", title="Ablations (E3/E4)")],
    "Systems": [st.Page(_PAGES / "systems.py", title="Systems profile (E5)"),
                st.Page(_PAGES / "placement.py", title="Expert placement (E6)")],
    "Learn": [st.Page(_PAGES / "how_it_works.py", title="How it works")],
})
layout.sidebar()   # renders into st.sidebar and fills st.session_state (spec §2.3)
pg.run()
