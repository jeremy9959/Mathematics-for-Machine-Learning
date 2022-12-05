CONTROLS = metadata.yml Makefile
CHAPTERS = $(shell find chapters/ -type f -name '*.md' | sort)
#FILTERS = --filter pandoc-xnos 
#OPTIONS = -N --standalone --mathjax --toc --top-level-division=chapter
METADATA = --metadata-file metadata.yml
OUTPUT = -o build.pdf

book: build.pdf
	
build.pdf : $(CHAPTERS) $(CONTROLS)
	cat $(CHAPTERS) | quarto render $(METADATA) $(OUTPUT)


