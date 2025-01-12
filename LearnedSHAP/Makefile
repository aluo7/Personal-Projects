PORT=8888

build:
	docker build -t shap .

run:
	docker run -p $(PORT):8888 -v $(PWD):/app shap

clean:
	docker rm $$(docker ps -a -q -f ancestor=shap -n 1)
