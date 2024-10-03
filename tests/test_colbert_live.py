import pytest

from colbert_live import ColbertLive
from colbert_live.models import ColpaliModel
from .db import InMemoryDB  # Use absolute import here

@pytest.mark.parametrize("model_name", [
    'vidore/colqwen2-v0.1',
    'vidore/colpali-v1.2'
])
def test_colpali(model_name):
    model = ColpaliModel(model_name)
    db = InMemoryDB(model)
    colbert = ColbertLive(db, model, 1, 0)
    for fname in db.encodings.keys():
        keyword = fname.split('.')[0]
        r = colbert.search(keyword)
        top_fname, _ = r[0]
        assert top_fname == fname
