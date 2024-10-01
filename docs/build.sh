#!/bin/sh

#
# Run this from the main colbert-live source root
#

echo 'Extracting ColbertLive.md'
python docs/pydoc-md.py colbert_live ColbertLive > docs/sources/ColbertLive.md
echo 'Extracting AstraCQL.md'
python docs/pydoc-md.py colbert_live.db.astra AstraCQL > docs/sources/AstraCQL.md
echo 'interpolating to colbert-live-astra.md'
python docs/interpolate.py colbert-live-astra.md.in colbert-live-astra.md

echo 'Extracting DB.md'
python docs/pydoc-md.py colbert_live.db DB > docs/sources/DB.md
echo 'interpolating to colbert-live-db.md'
python docs/interpolate.py colbert-live-db.md.in colbert-live-db.md
