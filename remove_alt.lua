function Image(el)
  -- remove image title (this is what becomes alt= in LaTeX)
  if el.target then
    el.target[2] = ""
  end
  return el
end