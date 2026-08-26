'use strict';
const fs = require('fs');
const path = require('path');
const targets = {
  '2024-01-30-multivariate-statistical-analysis-notes.md': [
    ['/en/blog/2024/09/11/multivariate-statistics-introduction-notes/', 'the introductory multivariate statistics notes']
  ],
  '2024-11-14-generative-and-diffusion-models.md': [
    ['/en/blog/2026/05/04/science-fiction-and-chinese-sci-fi/', 'the related generative media essay'],
    ['/en/blog/2024/11/14/self-attention-and-transformer-architecture/', 'the transformer architecture notes']
  ],
  '2024-12-30-financial-markets.md': [
    ['/en/blog/2024/12/29/economic-foundations/', 'the economic foundations notes'],
    ['/en/blog/2024/01/30/linear-time-series-analysis-notes/', 'the time-series analysis notes']
  ],
  '2025-05-13-algorithm-design-and-analysis.md': [
    ['/en/blog/2025/05/12/data-structures-introduction/', 'the data structures introduction'],
    ['/en/blog/2024/12/18/search-and-sorting-algorithms/', 'the search and sorting algorithms notes']
  ]
};
const dir = path.resolve(__dirname, '..', 'source_en', '_posts');
for (const [name, links] of Object.entries(targets)) {
  const file = path.join(dir, name);
  if (!fs.existsSync(file)) continue;
  const source = fs.readFileSync(file, 'utf8');
  const body = links.map(([href, label]) => `<a href="${href}">${label}</a>`).join(' and ');
  if (source.includes('translation-notice')) {
    const next = source.replace(/\n?$/, `\n\n<p>For context, compare ${body}.</p>\n`);
    fs.writeFileSync(file, next, 'utf8');
  }
}
