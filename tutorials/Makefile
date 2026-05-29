# ==========================================
# BoolForge tutorial pipeline (src â†’ outputs)
# ==========================================

# Canonical sources
PY_TUTORIALS := $(wildcard tutorials/src/tutorial*.py)

# Generated notebooks (same basename, different folder)
IPYNBS := $(patsubst tutorials/src/%.py,tutorials/%.ipynb,$(PY_TUTORIALS))

# Execution timeout (seconds)
TIMEOUT := 300

.PHONY: tutorials html clean distclean

# ------------------------------------------
# Convert + execute all tutorials
# ------------------------------------------
tutorials: $(IPYNBS)
	@echo "âœ… All tutorials converted and executed successfully."

# Rule: src/*.py â†’ tutorials/*.ipynb â†’ executed
tutorials/%.ipynb: tutorials/src/%.py
	@echo "â–¶ Converting $< â†’ $@"
	jupytext --to notebook $< --output $@
	@echo "â–¶ Executing $@"
	jupyter nbconvert \
	  --execute \
	  --to notebook \
	  --inplace \
	  --ExecutePreprocessor.timeout=$(TIMEOUT) \
	  $@

# ------------------------------------------
# Render HTML previews
# ------------------------------------------
html: tutorials
	@echo "â–¶ Rendering HTML previews"
	jupyter nbconvert --to html $(IPYNBS)

# ------------------------------------------
# Clean generated artifacts
# ------------------------------------------
clean:
	rm -f tutorials/*.html

distclean:
	rm -f tutorials/*.ipynb tutorials/*.html
	
# ------------------------------------------
# Create pdfs
# ------------------------------------------
.PHONY: tutorial_pdfs

tutorial_pdfs:
	@mkdir -p tutorials/pdf
	@ls tutorials/*.ipynb | xargs -n 1 -P 4 sh -c '\
		echo "▶ Converting $$0 → PDF"; \
		jupyter nbconvert --to webpdf \
		  --HTMLExporter.extra_css="[\"pre { white-space: pre-wrap !important; word-break: break-word !important; }\"]" \
		  --TemplateExporter.exclude_input_prompt=True \
		  "$$0" --output-dir tutorials/pdf \
	'