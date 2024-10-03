from colbert_live import ColbertLive
from colbert_live.models import ColpaliModel
from .db import InMemoryDB

def test_colpali():
    model = ColpaliModel()
    db = InMemoryDB(model)
    colbert = ColbertLive(db, model, 1, 0)
    for fname in db.encodings.keys():
        keyword = fname.split('.')[0]
        r = colbert.search(keyword)
        top_fname, _ = r[0]
        assert top_fname == fname
