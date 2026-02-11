# ==========================================
# BoolForge tutorial pipeline (src → outputs)
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
	@echo "✅ All tutorials converted and executed successfully."

# Rule: src/*.py → tutorials/*.ipynb → executed
tutorials/%.ipynb: tutorials/src/%.py
	@echo "▶ Converting $< → $@"
	jupytext --to notebook $< --output $@
	@echo "▶ Executing $@"
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
	@echo "▶ Rendering HTML previews"
	jupyter nbconvert --to html $(IPYNBS)

# ------------------------------------------
# Clean generated artifacts
# ------------------------------------------
clean:
	rm -f tutorials/*.html

distclean:
	rm -f tutorials/*.ipynb tutorials/*.html
