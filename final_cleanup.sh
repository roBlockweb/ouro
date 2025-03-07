#!/bin/bash
# Final cleanup for Ouro project before GitHub push

# Remove temporary files and directories
rm -rf examples/ test_venv/ venv/ .requirements_installed
rm -f test.sh reset.sh

echo "Cleanup complete. Project is ready for GitHub."