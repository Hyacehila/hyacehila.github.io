(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SearchTokenizer = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CONTROL_CHARACTERS = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g;
  const TOKEN_PATTERN = /c\+\+|c#|node\.js|[a-z][a-z0-9]*(?:[._+#-][a-z0-9+#]+)*|[0-9]+(?:\.[0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/giu;
  const HAN_PATTERN = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g;
  const HAN_SEQUENCE = /^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$/;
  const LATIN_TERM = /^[a-z][a-z0-9]*$/;
  const METADATA_FIELDS = new Set([
    "titleEn",
    "titleZh",
    "heading",
    "tags",
    "categories",
    "codeTerms",
  ]);
  const STRICT_SHORT_FIELDS = new Set([
    "titleEn",
    "titleZh",
    "tags",
    "categories",
    "codeTerms",
  ]);
  const PROTECTED_SHORT_TERMS = new Set([
    "ai",
    "ml",
    "rl",
    "llm",
    "rag",
    "mcp",
    "bm25",
    "r",
    "c",
    "go",
    "js",
    "ts",
  ]);
  const STOP_WORDS = new Set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
  ]);

  const normalizeText = (value) => {
    return String(value || "")
      .normalize("NFKC")
      .replace(CONTROL_CHARACTERS, " ")
      .replace(/[\u2028\u2029]/g, "\n");
  };

  const normalizeForMatch = (value) => {
    return normalizeText(value)
      .toLocaleLowerCase()
      .replace(/[\s\u00a0]+/g, " ")
      .trim();
  };

  const addChineseTerms = (terms, value, isQuery) => {
    const characters = Array.from(value);
    if (characters.length < 2) return;
    if (!isQuery && characters.length <= 8) terms.push(characters.join(""));
    for (let index = 0; index < characters.length - 1; index += 1) {
      terms.push(characters[index] + characters[index + 1]);
    }
  };

  const splitCamelCase = (value) => {
    return value
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .split(/[^A-Za-z0-9]+/)
      .map((part) => part.toLocaleLowerCase())
      .filter(Boolean);
  };

  const addIndexedLatinTerms = (terms, rawToken, normalizedToken) => {
    const variants = new Set([normalizedToken]);
    if (normalizedToken === "c++") variants.add("cpp");
    if (normalizedToken === "c#") variants.add("csharp");
    if (normalizedToken === "node.js") variants.add("nodejs");
    if (/[-_]/.test(normalizedToken)) {
      const parts = normalizedToken.split(/[-_]+/).filter(Boolean);
      if (parts.length > 1) {
        variants.add(parts.join(""));
        parts.forEach((part) => variants.add(part));
      }
    }
    const camelParts = splitCamelCase(rawToken);
    if (camelParts.length > 1) camelParts.forEach((part) => variants.add(part));
    variants.forEach((variant) => terms.push(variant));
  };

  const addQueryLatinTerms = (terms, normalizedToken) => {
    if (/[-_]/.test(normalizedToken)) {
      const parts = normalizedToken.split(/[-_]+/).filter(Boolean);
      if (parts.length > 1) {
        parts.forEach((part) => terms.push(part));
        return;
      }
    }
    terms.push(normalizedToken);
  };

  const shouldKeepLatinTerm = (term, fieldName, isQuery) => {
    if (PROTECTED_SHORT_TERMS.has(term)) {
      if (!isQuery && (term === "r" || term === "c" || term === "go")) {
        return STRICT_SHORT_FIELDS.has(fieldName);
      }
      return true;
    }
    if (STOP_WORDS.has(term)) return false;
    if (term.length > 1) return true;
    return isQuery || METADATA_FIELDS.has(fieldName);
  };

  const tokenize = (value, fieldName) => {
    const source = normalizeText(value);
    const isQuery = !fieldName;
    const terms = [];
    const matches = source.match(TOKEN_PATTERN) || [];

    matches.forEach((rawToken) => {
      if (HAN_SEQUENCE.test(rawToken)) {
        addChineseTerms(terms, rawToken, isQuery);
        return;
      }

      const normalizedToken = rawToken.toLocaleLowerCase();
      if (!shouldKeepLatinTerm(normalizedToken, fieldName, isQuery)) return;
      if (isQuery) addQueryLatinTerms(terms, normalizedToken);
      else addIndexedLatinTerms(terms, rawToken, normalizedToken);
    });

    return terms.filter((term) => {
      if (HAN_SEQUENCE.test(term)) return term.length >= 2;
      return shouldKeepLatinTerm(term, fieldName, isQuery);
    });
  };

  const uniqueQueryTerms = (value) => Array.from(new Set(tokenize(value)));

  const analyzeQuery = (value) => {
    const normalized = normalizeForMatch(value);
    const hanCharacters = normalized.match(HAN_PATTERN) || [];
    const latinContent = normalized.replace(HAN_PATTERN, "").replace(/[^a-z0-9+#.]+/g, "");
    const terms = uniqueQueryTerms(normalized);

    if (!normalized) return { valid: false, reason: "empty", normalized, terms };
    if (hanCharacters.length === 1 && !latinContent) {
      return { valid: false, reason: "short_chinese", normalized, terms };
    }
    if (terms.length === 0) return { valid: false, reason: "no_terms", normalized, terms };
    return { valid: true, reason: null, normalized, terms };
  };

  const isPrefixTerm = (term) => LATIN_TERM.test(term) && term.length >= 3;
  const isFuzzyTerm = (term) => LATIN_TERM.test(term) && term.length >= 5;

  return {
    CONTROL_CHARACTERS,
    PROTECTED_SHORT_TERMS,
    analyzeQuery,
    isFuzzyTerm,
    isPrefixTerm,
    normalizeForMatch,
    normalizeText,
    tokenize,
    uniqueQueryTerms,
  };
});
