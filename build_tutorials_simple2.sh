#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Build a single PDF from already-executed tutorial notebooks
# Robust against YAML + future Pandoc strictness
# ------------------------------------------------------------

TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
OUT_PDF="BoolForge_Tutorials.pdf"
HEADER_TEX="tutorials_header.tex"

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$OUT_PDF" "$HEADER_TEX"
mkdir -p "$MD_DIR"

# ------------------------------------------------------------
# Convert notebooks to Markdown
#   - exclude metadata (prevents YAML front matter)
#   - preserve figures
# ------------------------------------------------------------
echo "==> Converting notebooks to Markdown"

for nb in "$TUTORIAL_DIR"/*.ipynb; do
  jupyter nbconvert \
    --to markdown \
    --TemplateExporter.exclude_metadata=True \
    --TemplateExporter.exclude_input_prompt=True \
    --TemplateExporter.exclude_output_prompt=True \
    "$nb" \
    --output-dir "$MD_DIR"
done

# ------------------------------------------------------------
# Create LaTeX header
#   - allow alt= in includegraphics
#   - new page per section
# ------------------------------------------------------------
echo "==> Writing LaTeX header"

cat > "$HEADER_TEX" <<'EOF'
\usepackage{graphicx}

% Allow alt= key in \includegraphics without error
\makeatletter
\define@key{Gin}{alt}{}
\makeatother

\usepackage{etoolbox}
\pretocmd{\section}{\clearpage}{}{}
EOF

# ------------------------------------------------------------
# Build PDF with Pandoc
#   - disable YAML parsing explicitly
#   - use modern markdown extensions
# ------------------------------------------------------------
echo "==> Building $OUT_PDF"

pandoc \
  "$MD_DIR"/*.md \
  --resource-path="$MD_DIR" \
  --from markdown+implicit_figures-yaml_metadata_block \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  -V geometry:margin=1in \
  -V documentclass=article \
  -V monofont=Menlo \
  -H "$HEADER_TEX" \
  -o "$OUT_PDF"

echo "==> Done: $OUT_PDF"