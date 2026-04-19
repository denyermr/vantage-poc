.PHONY: poc export-fixtures mvp all gate3-and-export

# Run full science pipeline through Phase 3
poc:
	cd echo-poc && python poc/pipeline.py --from-raw

# Export fixtures from Phase 3 results into MVP
export-fixtures:
	cd echo-poc && python poc/export/export_fixtures.py

# Build MVP (requires fixtures to be populated first)
mvp:
	cd vantage-mvp && pnpm install && pnpm build

# Full end-to-end: science → fixtures → MVP build
all: poc export-fixtures mvp

# Run Phase 3 gate then export (most common workflow)
gate3-and-export:
	cd echo-poc && python poc/gates/gate_3.py --confirm-deviations
	cd echo-poc && python poc/export/export_fixtures.py
