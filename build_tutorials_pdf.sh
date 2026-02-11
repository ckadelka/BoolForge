#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Build a single PDF from already-executed tutorial notebooks
# ------------------------------------------------------------

TUTORIAL_DIR="tutorials"
MD_DIR="tutorials_md"
MD_CLEAN_DIR="tutorials_md_clean"
OUT_MD="BoolForge_Tutorials.md"
OUT_PDF="BoolForge_Tutorials.pdf"
HEADER_TEX="tutorials_header.tex"

echo "==> Cleaning previous build artifacts"
rm -rf "$MD_DIR" "$MD_CLEAN_DIR" "$OUT_MD" "$OUT_PDF" "$HEADER_TEX"

mkdir -p "$MD_DIR" "$MD_CLEAN_DIR"

# ------------------------------------------------------------
# Convert notebooks to Markdown
# ------------------------------------------------------------
echo "==> Converting notebooks to Markdown"
for nb in "$TUTORIAL_DIR"/*.ipynb; do
  jupyter nbconvert \
    --to markdown \
    "$nb" \
    --output-dir "$MD_DIR"
done

# ------------------------------------------------------------
# Strip YAML front matter ONLY (top of file)
# ------------------------------------------------------------
echo "==> Stripping YAML front matter"
for md in "$MD_DIR"/*.md; do
  awk '
    NR==1 && $0=="---" {
      in_yaml=1
      next
    }
    in_yaml && $0=="---" {
      in_yaml=0
      next
    }
    !in_yaml {
      print
    }
  ' "$md" > "$MD_CLEAN_DIR/$(basename "$md")"
done



# ------------------------------------------------------------
# Strip ALL heading markers and manual numbering
# ------------------------------------------------------------
echo "==> Normalizing headings (keep structure, strip junk)"
for md in "$MD_CLEAN_DIR"/*.md; do
  sed -i '' -E '
    /^#+[[:space:]]+/ {
      s/^(\#+)[[:space:]]+(#+[[:space:]]*)?[0-9]+([.:-][0-9]+)*[.:-]?[[:space:]]+/\1 /
    }
  ' "$md"
done



# ------------------------------------------------------------
# Strip image-only Markdown attributes (safe for math)
# ------------------------------------------------------------
echo "==> Stripping image attributes only"
for md in "$MD_CLEAN_DIR"/*.md; do
  sed -i '' 's/!\(\[[^]]*\]\([^)]*\)\){[^}]*}/!\1/g' "$md"
done

# ------------------------------------------------------------
# Create a small LaTeX header for Pandoc (formatting only)
# ------------------------------------------------------------
echo "==> Writing Pandoc LaTeX header"
cat > "$HEADER_TEX" << 'EOF'
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{keyval}

% ---- Ignore Pandoc-inserted image attributes ----
\makeatletter
\define@key{Gin}{alt}{}
\makeatother
EOF

# ------------------------------------------------------------
# Concatenate cleaned Markdown files in sorted order
# ------------------------------------------------------------
echo "==> Concatenating Markdown files"
for md in "$MD_CLEAN_DIR"/*.md; do
  echo "\\newpage" >> "$OUT_MD"
  cat "$md" >> "$OUT_MD"
done

# ------------------------------------------------------------
# Convert combined Markdown to a single PDF
# ------------------------------------------------------------
echo "==> Building final PDF"
pandoc "$OUT_MD" \
  -o "$OUT_PDF" \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  -H "$HEADER_TEX" \
  --resource-path=.:tutorials_md \
  -f markdown-implicit_figures \
  --metadata=figurePosition=H

echo "==> Done!"
echo "Generated: $OUT_PDF"
