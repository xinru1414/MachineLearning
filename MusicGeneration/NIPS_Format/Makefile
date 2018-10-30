all: open


OS := $(shell uname)
ifeq ($(OS),Darwin)
open: build
	open main.pdf
else
open: build
	xdg-open main.pdf
endif

build:
	xelatex main
	biber main
	xelatex main

clean:
	rm *.aux *.bbl *.log *.out *.run.xml *.toc *.bcf *.blg *.lof *.lot
