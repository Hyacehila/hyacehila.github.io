"use strict";

importScripts(
  "../vendor/minisearch-7.2.0.js",
  "./search-tokenizer.js",
  "./search-engine.js",
);

let manifest = null;
let manifestUrl = null;
let corePromise = null;
let archivePromise = null;
let archiveReady = false;
let latestRequest = null;
const indexes = [];
const searchData = { posts: {}, records: {}, aliases: {} };

const fetchChecked = async (url, asText, cache) => {
  const response = await fetch(url, { cache });
  if (!response.ok) throw new Error(`Search asset request failed (${response.status}): ${url}`);
  return asText ? response.text() : response.json();
};

const resolveShardUrl = (fileName) => new URL(fileName, manifestUrl).href;

const loadShard = async (name) => {
  const shard = manifest.shards && manifest.shards[name];
  if (!shard) throw new Error(`Search manifest is missing the ${name} shard`);
  const [indexJson, documents] = await Promise.all([
    fetchChecked(resolveShardUrl(shard.index), true, "force-cache"),
    fetchChecked(resolveShardUrl(shard.documents), false, "force-cache"),
  ]);
  if (documents.schemaVersion !== SearchEngine.INDEX_SCHEMA_VERSION) {
    throw new Error(`Unsupported search document schema: ${documents.schemaVersion}`);
  }
  const index = await MiniSearch.loadJSONAsync(indexJson, SearchEngine.createMiniSearchOptions());
  indexes.push(index);
  Object.assign(searchData.posts, documents.posts || {});
  Object.assign(searchData.records, documents.records || {});
  Object.assign(searchData.aliases, documents.aliases || {});
  return shard;
};

const runLatestQuery = (refined) => {
  if (!latestRequest || indexes.length === 0) return;
  const startedAt = performance.now();
  const response = SearchEngine.search(indexes, searchData, latestRequest.query, {
    limit: latestRequest.limit,
  });
  postMessage({
    type: "results",
    id: latestRequest.id,
    partial: !archiveReady,
    refined: Boolean(refined),
    durationMs: performance.now() - startedAt,
    response,
  });
};

const beginArchiveLoad = () => {
  if (archivePromise) return archivePromise;
  archivePromise = loadShard("archive")
    .then((stats) => {
      archiveReady = true;
      postMessage({ type: "ready", stage: "complete", stats, manifest });
      runLatestQuery(true);
      return stats;
    })
    .catch((error) => {
      postMessage({ type: "archive-error", message: error.message });
      return null;
    });
  return archivePromise;
};

const initialize = async (url) => {
  if (corePromise) return corePromise;
  manifestUrl = url;
  corePromise = (async () => {
    manifest = await fetchChecked(manifestUrl, false, "no-cache");
    if (manifest.schemaVersion !== SearchEngine.INDEX_SCHEMA_VERSION) {
      throw new Error(`Unsupported search manifest schema: ${manifest.schemaVersion}`);
    }
    const stats = await loadShard("core");
    postMessage({ type: "ready", stage: "core", stats, manifest });
    beginArchiveLoad();
  })().catch((error) => {
    corePromise = null;
    postMessage({ type: "error", message: error.message });
    throw error;
  });
  return corePromise;
};

self.addEventListener("message", async (event) => {
  const message = event.data || {};
  if (message.type === "load") {
    try {
      await initialize(message.manifestUrl);
    } catch (error) {
    }
    return;
  }

  if (message.type === "query") {
    latestRequest = {
      id: message.id,
      query: String(message.query || ""),
      limit: Number.isFinite(message.limit) ? message.limit : 20,
    };
    try {
      await initialize(message.manifestUrl || manifestUrl);
      runLatestQuery(false);
    } catch (error) {
      postMessage({ type: "error", id: message.id, message: error.message });
    }
  }
});
