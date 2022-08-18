PAPER = Nebula
TEX = $(wildcard *.tex)
FIG = $(wildcard figures/*.pdf)
SRC = $(wildcard *.cls *.sty)

.PHONY: distclean clean

$(PAPER).pdf: $(TEX) $(FIG) $(SRC)
	@rm -rf *.glsdefs
	@pdflatex $(PAPER) > /dev/null
	@pdflatex $(PAPER) > /dev/null

distclean:
	rm -rf *.glsdefs *.aux *.bbl *.blg *.log *.out

clean:
	make distclean; rm -f $(PAPER).pdf

