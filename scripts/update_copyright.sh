#!/bin/sh

# Read the copyright text from the file
copyright_text=$(cat ./doc/copyright.txt)
for i in $(find . -name "*.cc" -o -name "*.h" -o -name "*.dox"); do
  # Compute the current number of header lines, i.e., 
  # the number of consecutive lines that start with //
  expected_lines=$(cat ./doc/copyright.txt | wc -l | tr -d '[:space:]')

  header_lines=$(head -n $expected_lines $i | grep '^//' | wc -l | tr -d '[:space:]')

  # Make sure that the first $header_lines lines of the file are actually
  # commented lines. If not, skip the file, as we don't want to mess with it.
  # Output its name so tthat the user can manually fix it.
  commented_lines_at_top=$(head -n $header_lines $i | grep '^//' | wc -l | tr -d '[:space:]')

  if [ "$header_lines" -ne "$commented_lines_at_top" ]; then
    echo "Skipping $i. Expecting $header_lines commented lines at the top of the file. Got $commented_lines_at_top."
    continue
  fi

  # Extract the current header of the file and filter for commented lines
  current_header=$(head -n $header_lines $i | grep '^//')

  # Check if the current header is different from the copyright text or if it contains non-commented lines
  if [ "$current_header" != "$copyright_text" ]; then
    # increase header_lines by 1 to account for the empty line between the header and the rest of the file
    header_lines=$((header_lines + 1))
    # Replace the header with the copyright text
    {
      echo "$copyright_text"
      echo ""
      tail -n +$header_lines $i | sed '/./,$!d' # Output the rest of the file without leading empty lines
    } > $i.new && mv $i.new $i
    echo "Updated $i"
  fi
done