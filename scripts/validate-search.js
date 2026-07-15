'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const zlib = require('zlib');
const MiniSearch = require('minisearch');
const SearchTokenizer = require('../source/assets/js/search-tokenizer.js');
const SearchEngine = require('../source/assets/js/search-engine.js');
const evaluationCases = require('./search-eval.json');

const publicSearchDir = path.join(__dirname, '..', 'public', 'assets', 'search');
const manifestPath = path.join(publicSearchDir, 'manifest.json');
const MAX_SHARD_GZIP_BYTES = 1.5 * 1024 * 1024;
const MAX_QUERY_P95_MS = 50;
const failures = [];

const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const readJson = filePath => JSON.parse(fs.readFileSync(filePath, 'utf8'));
const normalize = value => SearchTokenizer.normalizeForMatch(value);
const percentile = (values, ratio) => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)] || 0;
};

const findControlCharacters = (value, location, found) => {
  if (typeof value === 'string') {
    if (SearchTokenizer.CONTROL_CHARACTERS.test(value)) found.push(location);
    SearchTokenizer.CONTROL_CHARACTERS.lastIndex = 0;
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => findControlCharacters(item, `${location}[${index}]`, found));
    return;
  }
  if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, item]) => findControlCharacters(item, `${location}.${key}`, found));
  }
};

const loadSearch = () => {
  check(fs.existsSync(manifestPath), 'Search manifest is missing. Run the Hexo build first.');
  if (!fs.existsSync(manifestPath)) return null;
  const manifest = readJson(manifestPath);
  check(manifest.schemaVersion === SearchEngine.INDEX_SCHEMA_VERSION, 'Search manifest schema is incompatible.');
  check(manifest.engineVersion === '7.2.0', `Expected MiniSearch 7.2.0, received ${manifest.engineVersion}.`);
  check(manifest.shards && manifest.shards.core && manifest.shards.archive, 'Core/archive search shards are missing.');

  const vendorPath = path.join(publicSearchDir, '..', 'vendor', `minisearch-${manifest.engineVersion}.js`);
  check(fs.existsSync(vendorPath), 'MiniSearch browser bundle is missing.');
  if (fs.existsSync(vendorPath)) {
    const vendorSource = fs.readFileSync(vendorPath, 'utf8');
    const sandbox = {};
    try {
      vm.runInNewContext(vendorSource, sandbox, { filename: vendorPath });
      check(typeof sandbox.MiniSearch === 'function', 'MiniSearch browser bundle does not expose MiniSearch.');
    } catch (error) {
      check(false, `MiniSearch browser bundle is invalid JavaScript: ${error.message}`);
    }
  }

  const indexes = [];
  const data = { posts: {}, records: {}, aliases: {} };
  for (const shardName of ['core', 'archive']) {
    const shard = manifest.shards[shardName];
    if (!shard) continue;
    const indexPath = path.join(publicSearchDir, shard.index);
    const documentsPath = path.join(publicSearchDir, shard.documents);
    check(fs.existsSync(indexPath), `${shardName} index file is missing.`);
    check(fs.existsSync(documentsPath), `${shardName} document file is missing.`);
    if (!fs.existsSync(indexPath) || !fs.existsSync(documentsPath)) continue;

    const indexJson = fs.readFileSync(indexPath, 'utf8');
    const documentsJson = fs.readFileSync(documentsPath, 'utf8');
    const actualIndexGzip = zlib.gzipSync(indexJson).length;
    const actualDocumentsGzip = zlib.gzipSync(documentsJson).length;
    check(actualIndexGzip === shard.indexGzipBytes, `${shardName} index gzip size does not match the manifest.`);
    check(actualDocumentsGzip === shard.documentsGzipBytes, `${shardName} document gzip size does not match the manifest.`);
    check(actualIndexGzip <= MAX_SHARD_GZIP_BYTES, `${shardName} index exceeds 1.5 MiB gzip.`);
    check(actualDocumentsGzip <= MAX_SHARD_GZIP_BYTES, `${shardName} documents exceed 1.5 MiB gzip.`);

    const indexObject = JSON.parse(indexJson);
    const documents = JSON.parse(documentsJson);
    const controlLocations = [];
    findControlCharacters(indexObject, `${shardName}.index`, controlLocations);
    findControlCharacters(documents, `${shardName}.documents`, controlLocations);
    check(controlLocations.length === 0, `${shardName} output contains illegal control characters: ${controlLocations.slice(0, 3).join(', ')}`);
    check(documents.schemaVersion === SearchEngine.INDEX_SCHEMA_VERSION, `${shardName} document schema is incompatible.`);

    indexes.push(MiniSearch.loadJSON(indexJson, SearchEngine.createMiniSearchOptions()));
    Object.assign(data.posts, documents.posts || {});
    Object.assign(data.records, documents.records || {});
    Object.assign(data.aliases, documents.aliases || {});
  }

  check(Object.keys(data.posts).length === manifest.postCount, 'Search post count does not match the manifest.');
  check(Object.keys(data.records).length === manifest.recordCount, 'Search record count does not match the manifest.');
  return { manifest, indexes, data };
};

const validateExactTitles = (indexes, data) => {
  let reciprocalRank = 0;
  let topThree = 0;
  let queryCount = 0;
  Object.values(data.posts).forEach(post => {
    Array.from(new Set([post.titleEn, post.titleZh].filter(Boolean))).forEach(title => {
      queryCount += 1;
      const response = SearchEngine.search(indexes, data, title, { limit: 10 });
      const rank = response.results.findIndex(result => result.postId === post.postId) + 1;
      if (rank > 0) reciprocalRank += 1 / rank;
      if (rank > 0 && rank <= 3) topThree += 1;
    });
  });
  const mrrAtTen = reciprocalRank / queryCount;
  const topThreeRate = topThree / queryCount;
  check(mrrAtTen >= 0.95, `Exact-title MRR@10 is ${mrrAtTen.toFixed(3)}, expected at least 0.95.`);
  check(topThreeRate === 1, `Exact-title Top-3 rate is ${topThreeRate.toFixed(3)}, expected 1.0.`);
  return { queryCount, mrrAtTen, topThreeRate };
};

const validateTaxonomyRecall = (indexes, data, fieldName) => {
  const groups = new Map();
  Object.values(data.posts).forEach(post => {
    (post[fieldName] || []).forEach(value => {
      if (!groups.has(value)) groups.set(value, []);
      groups.get(value).push(post.postId);
    });
  });
  let expected = 0;
  let found = 0;
  groups.forEach((postIds, query) => {
    const response = SearchEngine.search(indexes, data, query, { limit: 1000, maxRawResults: 5000 });
    const resultIds = new Set(response.results.map(result => result.postId));
    postIds.forEach(postId => {
      expected += 1;
      if (resultIds.has(postId)) found += 1;
    });
  });
  const recall = expected ? found / expected : 1;
  check(recall === 1, `${fieldName} recall is ${recall.toFixed(3)}, expected 1.0.`);
  return { groupCount: groups.size, expected, recall };
};

const validateEvaluationSet = (indexes, data) => {
  const failedCases = [];
  evaluationCases.forEach(testCase => {
    const topK = testCase.topK || 5;
    const response = SearchEngine.search(indexes, data, testCase.query, { limit: Math.max(10, topK) });
    const expected = normalize(testCase.expectedTitle);
    const matched = response.results.slice(0, topK).some(result => {
      return normalize(result.titleEn).includes(expected) || normalize(result.titleZh).includes(expected);
    });
    if (!matched) {
      failedCases.push({
        query: testCase.query,
        expected: testCase.expectedTitle,
        top: response.results.slice(0, topK).map(result => result.titleEn)
      });
    }
  });
  const successAtFive = (evaluationCases.length - failedCases.length) / evaluationCases.length;
  check(successAtFive >= 0.8, `Curated topic success@5 is ${successAtFive.toFixed(3)}, expected at least 0.8.`);
  return { caseCount: evaluationCases.length, successAtFive, failedCases };
};

const validateNegativeBoundaries = (indexes, data) => {
  const aiTitles = SearchEngine.search(indexes, data, 'AI', { limit: 10 }).results.map(result => normalize(result.titleEn));
  const ragTitles = SearchEngine.search(indexes, data, 'RAG', { limit: 10 }).results.map(result => normalize(result.titleEn));
  check(!aiTitles.some(title => title.includes('matrix theory')), 'AI incorrectly matches Matrix Theory through a substring boundary.');
  check(!ragTitles.some(title => title.includes('storage') || title.includes('paragraph')), 'RAG incorrectly matches storage/paragraph through a substring boundary.');
};

const validatePerformance = (indexes, data) => {
  const queries = evaluationCases.map(testCase => testCase.query);
  queries.forEach(query => SearchEngine.search(indexes, data, query, { limit: 20 }));
  const durations = [];
  for (let round = 0; round < 3; round += 1) {
    queries.forEach(query => {
      const startedAt = performance.now();
      SearchEngine.search(indexes, data, query, { limit: 20 });
      durations.push(performance.now() - startedAt);
    });
  }
  const p95 = percentile(durations, 0.95);
  check(p95 < MAX_QUERY_P95_MS, `Warm query P95 is ${p95.toFixed(2)} ms, expected below 50 ms.`);
  return { sampleCount: durations.length, p95 };
};

const run = () => {
  const loaded = loadSearch();
  if (loaded) {
    const exactTitles = validateExactTitles(loaded.indexes, loaded.data);
    const tags = validateTaxonomyRecall(loaded.indexes, loaded.data, 'tags');
    const categories = validateTaxonomyRecall(loaded.indexes, loaded.data, 'categories');
    const evaluation = validateEvaluationSet(loaded.indexes, loaded.data);
    validateNegativeBoundaries(loaded.indexes, loaded.data);
    const performanceResult = validatePerformance(loaded.indexes, loaded.data);

    console.log(JSON.stringify({
      posts: loaded.manifest.postCount,
      records: loaded.manifest.recordCount,
      sourceControlCharactersCleaned: loaded.manifest.sourceControlCharactersCleaned,
      indexShardsKiB: Object.fromEntries(Object.entries(loaded.manifest.shards).map(([name, shard]) => [
        name,
        Math.round(shard.indexGzipBytes / 1024)
      ])),
      exactTitles: {
        queries: exactTitles.queryCount,
        mrrAtTen: Number(exactTitles.mrrAtTen.toFixed(4)),
        topThreeRate: Number(exactTitles.topThreeRate.toFixed(4))
      },
      taxonomyRecall: {
        tags: Number(tags.recall.toFixed(4)),
        categories: Number(categories.recall.toFixed(4))
      },
      curatedSuccessAtFive: Number(evaluation.successAtFive.toFixed(4)),
      curatedFailures: evaluation.failedCases,
      warmQueryP95Ms: Number(performanceResult.p95.toFixed(2))
    }, null, 2));
  }

  if (failures.length) {
    console.error('\nSearch validation failed:');
    failures.forEach(failure => console.error(`- ${failure}`));
    process.exitCode = 1;
    return;
  }

  console.log('Search validation passed.');
};

if (require.main === module) run();

module.exports = { run };
