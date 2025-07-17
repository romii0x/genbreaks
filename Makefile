install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	mkdir -p models output

clean:
	rm -rf models/* output/*

clean-models:
	rm -rf models/*

clean-output:
	rm -rf output/*