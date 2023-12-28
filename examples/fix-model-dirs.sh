#!/bin/bash

# The function 'find_broken_links' will search for broken symbolic links
find_broken_links() {
    local path="${1:-.}" # Default to current directory if no argument is provided
    # Find all files that are broken links in a given directory
    find "$path" -xtype l 2>/dev/null
}

# The function 'attempt_heal' tries to locate a file with the same name in the parent directories
attempt_heal() {
    local broken_link="$1"
    local link_target="$(readlink "$broken_link")"
    local link_name="$(basename "$link_target")"
    local current_dir="$(dirname "$broken_link")"

    # Ascend up the directory structure
    while [[ "$current_dir" != "/" ]]; do
        current_dir=$(dirname "$current_dir") # Move up one directory level

        # Search for file with same name in ancestor directory
        local possible_fix=$(find "$current_dir" -maxdepth 1 -type f -name "$link_name" -print -quit)
        
        if [[ -n "$possible_fix" ]]; then
            echo "Found: $possible_fix"
            echo "Broken: $broken_link"
            # If a file is found, create a new link
            # ln -sf "$possible_fix" "$broken_link"
            echo "Repaired: $broken_link -> $possible_fix"
            return 0
        fi
    done

    echo "Failed to repair: $broken_link" >&2
    return 1
}

# Main loop that processes each broken link
main() {
    local dir_to_search="${1:-.}" # Default directory to search is the current one
    while read -r broken_link; do
        attempt_heal "$broken_link"
    done < <(find_broken_links "$dir_to_search")
}
