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
      const marker = source.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
      if (!marker) continue;
      const head = marker[0];
      const body = source.slice(head.length);
      const sanitized = body
        .replace(/\$\{/g, '&#36;&#123;')
        .replace(/\$/g, '&#36;')
        .replace(/\{\{/g, '&#123;&#123;')
        .replace(/\}\}/g, '&#125;&#125;');
      if (sanitized !== body) {
        fs.writeFileSync(file, head + sanitized, 'utf8');
        updated += 1;
      }
    }
  }
}

walk(root);
console.log(`[sanitize] updated ${updated} English sources with CJK content.`);
