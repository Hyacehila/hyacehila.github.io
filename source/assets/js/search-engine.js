(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./search-tokenizer.js"));
  } else {
    root.SearchEngine = factory(root.SearchTokenizer);
  }
})(typeof self !== "undefined" ? self : this, function (SearchTokenizer) {
  "use strict";

  const INDEX_SCHEMA_VERSION = 1;
  const INDEX_FIELDS = [
    "titleEn",
    "titleZh",
    "heading",
    "tags",
    "categories",
    "excerpt",
    "body",
    "codeTerms",
  ];
  const FIELD_BOOSTS = {
    titleEn: 10,
    titleZh: 10,
    heading: 5,
    tags: 5,
    categories: 3,
    excerpt: 2,
    body: 1,
    codeTerms: 0.5,
  };

  const createMiniSearchOptions = () => ({
    fields: INDEX_FIELDS,
    idField: "id",
    storeFields: ["postId", "recordType"],
    tokenize: SearchTokenizer.tokenize,
    processTerm: (term) => term,
    searchOptions: {
      boost: FIELD_BOOSTS,
      weights: { fuzzy: 0.3, prefix: 0.65 },
      maxFuzzy: 2,
      bm25: { k: 1.2, b: 0.75, d: 0.5 },
    },
  });

  const normalizeList = (values) => {
    return (Array.isArray(values) ? values : [])
      .map((value) => SearchTokenizer.normalizeForMatch(value))
      .filter(Boolean);
  };

  const fieldFactor = (record, query) => {
    const titleValues = [record.titleEn, record.titleZh]
      .map(SearchTokenizer.normalizeForMatch)
      .filter(Boolean);
    const heading = SearchTokenizer.normalizeForMatch(record.heading);
    const body = SearchTokenizer.normalizeForMatch(record.body || record.excerpt);
    const tags = normalizeList(record.tags);
    const categories = normalizeList(record.categories);
    let factor = 1;

    if (titleValues.some((title) => title === query)) factor = Math.max(factor, 4);
    else if (titleValues.some((title) => title.startsWith(query))) factor = Math.max(factor, 2.8);
    else if (titleValues.some((title) => title.includes(query))) factor = Math.max(factor, 2.2);
    if (heading === query) factor = Math.max(factor, 2.2);
    else if (heading && heading.includes(query)) factor = Math.max(factor, 1.65);
    if (tags.includes(query)) factor = Math.max(factor, 1.8);
    if (categories.includes(query)) factor = Math.max(factor, 1.5);
    if (body && body.includes(query)) factor = Math.max(factor, 1.25);
    return factor;
  };

  const makeSearchOptions = (combineWith) => ({
    boost: FIELD_BOOSTS,
    combineWith,
    prefix: (term) => SearchTokenizer.isPrefixTerm(term),
    fuzzy: (term, index, terms) => (terms.length === 1 && SearchTokenizer.isFuzzyTerm(term) ? 0.14 : false),
    maxFuzzy: 2,
    weights: { fuzzy: 0.3, prefix: 0.65 },
    bm25: { k: 1.2, b: 0.75, d: 0.5 },
  });

  const matchedPostCount = (results) => {
    return new Set(results.map((result) => result.postId).filter(Boolean)).size;
  };

  const addSearchResults = (target, results, source, sourceWeight, originalQuery, records, maxRawResults) => {
    results.slice(0, maxRawResults).forEach((result) => {
      const record = records[result.id];
      if (!record) return;
      const score = result.score * sourceWeight * fieldFactor(record, originalQuery);
      const previous = target.get(result.id);
      const candidate = {
        id: result.id,
        postId: record.postId,
        record,
        score,
        source,
        terms: result.terms || [],
      };
      if (!previous || candidate.score > previous.score) target.set(result.id, candidate);
    });
  };

  const findSnippetPosition = (text, query, terms) => {
    const normalizedText = SearchTokenizer.normalizeForMatch(text);
    const candidates = [query, ...terms]
      .map(SearchTokenizer.normalizeForMatch)
      .filter((term) => term.length >= 2)
      .sort((left, right) => right.length - left.length);
    for (const candidate of candidates) {
      const position = normalizedText.indexOf(candidate);
      if (position >= 0) return position;
    }
    return 0;
  };

  const makeSnippet = (record, post, query, terms) => {
    const source = String(record.body || record.excerpt || post.excerpt || "").trim();
    if (!source) return "";
    const position = findSnippetPosition(source, query, terms);
    let start = Math.max(0, position - 72);
    let end = Math.min(source.length, start + 230);
    if (start > 0) {
      const boundary = source.slice(Math.max(0, start - 24), start + 24).search(/[\s，。！？；,.!?;:]/);
      if (boundary >= 0) start = Math.max(0, start - 24 + boundary + 1);
    }
    if (end < source.length) {
      const boundary = source.slice(end, end + 36).search(/[\s，。！？；,.!?;:]/);
      if (boundary >= 0) end += boundary;
    }
    return `${start > 0 ? "…" : ""}${source.slice(start, end).trim()}${end < source.length ? "…" : ""}`;
  };

  const aggregateResults = (hits, data, query, limit) => {
    const grouped = new Map();
    hits.forEach((hit) => {
      if (!hit.postId || !data.posts[hit.postId]) return;
      if (!grouped.has(hit.postId)) grouped.set(hit.postId, []);
      grouped.get(hit.postId).push(hit);
    });

    const articles = [];
    grouped.forEach((articleHits, postId) => {
      articleHits.sort((left, right) => right.score - left.score);
      const post = data.posts[postId];
      const best = articleHits[0];
      const score = best.score +
        (articleHits[1] ? articleHits[1].score * 0.16 : 0) +
        (articleHits[2] ? articleHits[2].score * 0.08 : 0);
      const matchedTerms = Array.from(new Set(articleHits.flatMap((hit) => hit.terms || [])));

      articles.push({
        postId,
        score,
        source: best.source,
        url: best.record.anchorUrl || post.url,
        postUrl: post.url,
        titleEn: post.titleEn,
        titleZh: post.titleZh,
        date: post.date,
        tags: post.tags || [],
        categories: post.categories || [],
        heading: best.record.heading || "",
        snippet: makeSnippet(best.record, post, query, matchedTerms),
        matchedTerms,
      });
    });

    return articles
      .sort((left, right) => right.score - left.score || String(left.postId).localeCompare(String(right.postId)))
      .slice(0, limit);
  };

  const search = (miniSearch, data, rawQuery, options = {}) => {
    const analysis = SearchTokenizer.analyzeQuery(rawQuery);
    const limit = Number.isFinite(options.limit) ? options.limit : 20;
    const maxRawResults = Number.isFinite(options.maxRawResults) ? options.maxRawResults : 360;

    if (!analysis.valid) {
      return {
        query: String(rawQuery || ""),
        normalizedQuery: analysis.normalized,
        status: analysis.reason,
        results: [],
      };
    }

    const records = data.records || {};
    const merged = new Map();
    const indexes = (Array.isArray(miniSearch) ? miniSearch : [miniSearch]).filter(Boolean);
    const searchAll = (query, searchOptions) => indexes.flatMap(index => index.search(query, searchOptions));
    const andResults = searchAll(rawQuery, makeSearchOptions("AND"));
    addSearchResults(merged, andResults, "and", 1, analysis.normalized, records, maxRawResults);

    const primaryPostCount = matchedPostCount(andResults);
    if (analysis.terms.length > 1 && primaryPostCount < 3) {
      const orResults = searchAll(rawQuery, makeSearchOptions("OR"));
      addSearchResults(merged, orResults, "or", 0.42, analysis.normalized, records, maxRawResults);
    }

    if (primaryPostCount < 5) {
      const aliases = (data.aliases && data.aliases[analysis.normalized]) || [];
      aliases.slice(0, 6).forEach((alias) => {
        const aliasResults = searchAll(alias, makeSearchOptions("AND"));
        addSearchResults(merged, aliasResults, "alias", 0.5, analysis.normalized, records, maxRawResults);
      });
    }

    return {
      query: String(rawQuery || ""),
      normalizedQuery: analysis.normalized,
      status: "ok",
      results: aggregateResults(Array.from(merged.values()), data, analysis.normalized, limit),
    };
  };

  return {
    FIELD_BOOSTS,
    INDEX_FIELDS,
    INDEX_SCHEMA_VERSION,
    createMiniSearchOptions,
    search,
  };
});
