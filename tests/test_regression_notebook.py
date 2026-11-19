"""Check the applied notebook; full integration requires the local field archive."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks/01-sonic-regression.ipynb"


def test_regression_notebook_contract():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    for section in (
        "Investigation contract",
        "Data provenance",
        "Failure analysis",
        "Mathematical model",
        "Conclusion and production decision",
    ):
        assert section in markdown
    assert "dsm_project.experiments.force_linear" in code
    assert "train_test_split" not in code
    assert markdown.count("$$") % 2 == 0


@pytest.mark.skipif(
    not (ROOT / "data/local/force-train.zip").is_file(),
    reason="Full FORCE integration needs the ignored, audited local archive",
)
def test_regression_notebook_clean_kernel():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="dsm-project",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    assert all(
        c.execution_count is not None for c in notebook.cells if c.cell_type == "code"
    )
