#!/bin/bash

echo "=== Downloading BFG Repo-Cleaner ==="
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar

echo "=== BFG downloaded successfully ==="
echo "Usage: java -jar bfg.jar [options]" 