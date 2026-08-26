'use strict';
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..', 'source_en');
let updated = 0;
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(file);
    else if (file.endsWith('.md')) {
      const source = fs.readFileSync(file, 'utf8');
      const next = source.replace(/^(permalink:\s*['"]?)\/en\//m, '$1/');
      if (next !== source) { fs.writeFileSync(file, next, 'utf8'); updated += 1; }
    }
  }
}
walk(root);
console.log(`[permalinks] normalized ${updated} English sources.`);
