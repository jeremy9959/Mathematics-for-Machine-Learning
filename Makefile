CONTROLS = Makefile
CHAPTERS = $(shell find chapters/ -type f -name '*.md' | sort)
#FILTERS = --filter pandoc-xnos 
#OPTIONS = -N --standalone --mathjax --toc --top-level-division=chapter
OUTPUT = --to pdf --output index.pdf

book: index.pdf

index.pdf : index.qmd $(CHAPTERS) $(CONTROLS)
	quarto render index.qmd $(OUTPUT)


