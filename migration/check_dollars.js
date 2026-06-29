const fs = require('fs');
const path = require('path');
const dir = 'source/_posts';
let flagged = 0;
for (const fn of fs.readdirSync(dir).filter(x => x.endsWith('.md'))) {
  const txt = fs.readFileSync(path.join(dir, fn), 'utf8');
  // strip fenced + inline code (KaTeX filter also skips these)
  const noCode = txt.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '');
  // count unescaped $ (not preceded by backslash)
  const dollars = (noCode.match(/(?<!\\)\$/g) || []).length;
  if (dollars % 2 !== 0) {
    flagged++;
    console.log('  ODD $ count(' + dollars + '): ' + fn);
  }
}
console.log('flagged: ' + flagged);
