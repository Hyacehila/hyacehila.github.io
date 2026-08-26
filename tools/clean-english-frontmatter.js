'use strict';
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..', 'source_en');
function clean(file) {
  const source = fs.readFileSync(file, 'utf8');
  const first = source.match(/^---[\r\n]+[\s\S]*?[\r\n]+---[\r\n]*/);
  if (!first) return;
  const rest = source.slice(first[0].length);
  const embedded = rest.match(/^\s*---[\r\n]+[\s\S]*?[\r\n]+---[\r\n]*/);
  if (embedded) fs.writeFileSync(file, source.slice(0, first[0].length) + rest.slice(embedded[0].length).replace(/^\s+/, ''), 'utf8');
}
function walk(dir) { for (const ent of fs.readdirSync(dir, {withFileTypes:true})) { const full=path.join(dir,ent.name); if(ent.isDirectory()) walk(full); else if(full.endsWith('.md')) clean(full); } }
walk(root);
console.log('cleaned embedded English front matter');
