main: coverage benchmark build_docs serve

build_docs: build_readme
	PYTHONPATH=src poetry run mkdocs build

serve:
	PYTHONPATH=src poetry run mkdocs serve

typecheck:
	poetry run pyright src/

lint:
	poetry run ruff check --fix && poetry run ruff format

test:
	PYTHONPATH=src poetry run pytest tests

coverage:
	PYTHONPATH=src poetry run coverage run --source=onorm -m pytest tests && poetry run coverage report -m && poetry run coverage html

build_readme: README.ipynb
	PYTHONPATH=src poetry run jupyter nbconvert --to markdown README.ipynb 
	rm -rf docs/README_files/ 
	mv README_files docs/ 
	cp -r docs/README_files .github/
	cp README.md .github/
	pandoc --from=markdown --to=rst --output=README README.md

benchmark: simulation/benchmark_results.md
	poetry run python simulation/benchmark_performance.py
	cp simulation/scaling_*.png docs/
	cp simulation/benchmark_results.md docs/
