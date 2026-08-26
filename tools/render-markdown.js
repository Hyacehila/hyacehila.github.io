#!/usr/bin/env node
'use strict';

const { marked } = require('marked');

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  const html = marked.parse(input, {
    gfm: true,
    breaks: false,
    mangle: false,
    headerIds: true
  });
  process.stdout.write(html);
});
