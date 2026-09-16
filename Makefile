# Convenience targets. Everything here is also runnable by hand.
.PHONY: setup up down data-1m data-10m verify verify-strict

setup:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt
	@echo "Remember: source .venv/bin/activate && pip freeze > requirements.lock"

up:
	docker compose up -d
	@echo "waiting for Milvus healthz..." && sleep 5
	@until curl -sf http://localhost:9091/healthz > /dev/null; do sleep 2; done
	@echo "Milvus v2.6.18 is up on :19530"

down:
	docker compose down

data-1m:
	python scripts/fixtures/download_corpus.py --rows 1000000 && python scripts/fixtures/download_nq.py

data-10m:
	python scripts/fixtures/download_corpus.py --rows 10000000 && python scripts/fixtures/download_nq.py

# Fails on corrupt files only; absent entries are expected on a fresh clone.
# Use verify-strict once every artifact has been generated.
verify:
	python scripts/fixtures/verify_checksums.py

verify-strict:
	python scripts/fixtures/verify_checksums.py --strict
