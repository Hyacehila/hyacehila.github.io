'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { frontMatter, metadataErrors } = require('../../scripts/validate-i18n');

const source = { categories: ['Programming', 'CS Foundations'], tags: ['SQL', 'Databases'] };

test('flow and block YAML arrays compare identically; missing visibility uses defaults', () => {
  const zh = frontMatter('---\ncategories: [Programming, CS Foundations]\ntags: [SQL, Databases]\n---\n');
  const en = frontMatter('---\ncategories:\n- Programming\n- CS Foundations\ntags:\n- SQL\n- Databases\nhidden: false\npublished: true\n---\n');
  assert.deepEqual(metadataErrors(zh, en), []);
});

for (const key of ['categories', 'tags', 'hidden', 'published']) {
  test(`rejects ${key} drift, including unpublished sources`, () => {
    const zh = { ...source, published: false };
    const en = { ...zh, [key]: Array.isArray(zh[key]) ? [...zh[key]].reverse() : !zh[key] };
    assert.ok(metadataErrors(zh, en).some(error => error.includes(key)));
  });
}

test('quoted booleans and null are rejected even when both languages agree', () => {
  for (const key of ['hidden', 'published']) {
    for (const value of ['false', null, 0]) {
      const fm = { ...source, [key]: value };
      assert.equal(metadataErrors(fm, fm).filter(error => error.includes('boolean')).length, 2);
    }
  }
});
