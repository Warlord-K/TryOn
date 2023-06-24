install:
	pip install -r requirements.txt
run:
	python3 main.py -i image.jpeg -c cloth.jpg
lint:
	pylint --disable=R,C,E1120,E1101 main.py utils/*.py
test:
	pytest *.py
format:
	black main.py utils/*.py