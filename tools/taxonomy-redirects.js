'use strict';

const fs = require('fs');
const path = require('path');

const escapeHtml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const tagSlug = value => encodeURI(String(value).trim().replace(/[\s/]+/g, '-'));

function taxonomyRedirects(config) {
  const seo = config.seo || {};
  return [
    ...Object.entries(seo.tag_aliases || {}).map(([from, to]) => ({
      from: `/tags/${tagSlug(from)}/`, to: `/tags/${tagSlug(to)}/`
    })),
    ...Object.entries(seo.category_redirects || {}).map(([from, to]) => ({
      from: `/categories/programming/${from}/`, to: `/categories/programming/${to}/`
    }))
  ];
}

function writeTaxonomyRedirects(config, publicDir) {
  const base = String(config.url).replace(/\/$/, '');
  const redirects = taxonomyRedirects(config);
  for (const { from, to } of redirects) {
    const targetFile = path.join(publicDir, decodeURI(to).replace(/^\//, ''), 'index.html');
    if (!fs.existsSync(targetFile)) throw new Error(`Redirect target does not exist: ${to}`);
    const file = path.join(publicDir, decodeURI(from).replace(/^\//, ''), 'index.html');
    const target = escapeHtml(`${base}${to}`);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex,follow"><link rel="canonical" href="${target}"><meta http-equiv="refresh" content="0;url=${target}"><title>Redirect</title></head><body><a href="${target}">Continue</a></body></html>\n`, 'utf8');
  }
  console.log(`[build] wrote ${redirects.length} taxonomy redirects`);
}

module.exports = { taxonomyRedirects, writeTaxonomyRedirects };
