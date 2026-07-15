let didInit = false;
let worker = null;
let workerReadyPromise = null;
let resolveWorkerReady = null;
let rejectWorkerReady = null;
let debounceTimer = null;
let requestId = 0;
let activeIndex = -1;
let latestResponse = null;
let latestPartial = false;
let archiveFailed = false;

const COPY = {
  en: {
    idle: "Search titles, headings, tags, categories, and article text.",
    loading: "Loading the search index…",
    deepLoading: "Results shown; loading the historical full-text index…",
    ready: "Search index ready.",
    noResults: "No matching articles.",
    shortChinese: "Enter at least two Chinese characters.",
    error: "Search could not be loaded.",
    retry: "Retry",
    resultCount: (count) => `${count} article${count === 1 ? "" : "s"} found`,
    archiveWarning: "Historical full-text search is temporarily unavailable.",
  },
  zh: {
    idle: "可检索标题、章节、标签、分类与文章正文。",
    loading: "正在加载搜索索引…",
    deepLoading: "已显示结果，历史文章全文索引仍在后台加载…",
    ready: "搜索索引已就绪。",
    noResults: "没有找到匹配的文章。",
    shortChinese: "中文关键词至少输入两个汉字。",
    error: "搜索索引加载失败。",
    retry: "重试",
    resultCount: (count) => `找到 ${count} 篇文章`,
    archiveWarning: "历史文章全文索引暂时不可用。",
  },
};

const getLanguage = () => {
  const dataLanguage = document.documentElement.dataset.lang || "";
  if (dataLanguage.toLocaleLowerCase().startsWith("zh")) return "zh";
  try {
    const stored = localStorage.getItem("lang") || "";
    if (stored.toLocaleLowerCase().startsWith("zh")) return "zh";
  } catch (error) {
  }
  return "en";
};

const text = (key, ...args) => {
  const value = COPY[getLanguage()][key];
  return typeof value === "function" ? value(...args) : value;
};

const getSearchDom = () => ({
  input: document.querySelector(".search-input"),
  result: document.getElementById("search-result"),
  status: document.getElementById("search-status"),
  overlay: document.querySelector(".search-pop-overlay"),
});

const assetUrl = (assetPath) => {
  const root = String(window.config?.root || "/");
  const base = new URL(root.endsWith("/") ? root : `${root}/`, window.location.origin);
  return new URL(String(assetPath).replace(/^\/+/, ""), base).href;
};

const getManifestUrl = () => assetUrl("assets/search/manifest.json");

const setStatus = (message) => {
  const { status } = getSearchDom();
  if (status) status.textContent = message || "";
};

const setBusy = (busy) => {
  const { result } = getSearchDom();
  if (result) result.setAttribute("aria-busy", busy ? "true" : "false");
};

const createState = (icon, message, retry = false) => {
  const container = document.createElement("div");
  container.id = "no-result";
  container.className = "search-state";
  const iconElement = document.createElement("i");
  iconElement.className = `fa-solid ${icon} fa-3x`;
  iconElement.setAttribute("aria-hidden", "true");
  const label = document.createElement("p");
  label.textContent = message;
  container.append(iconElement, label);
  if (retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-retry";
    button.textContent = text("retry");
    container.append(button);
  }
  return container;
};

const renderState = (icon, message, options = {}) => {
  const { result, input } = getSearchDom();
  if (!result) return;
  result.replaceChildren(createState(icon, message, options.retry));
  setBusy(Boolean(options.busy));
  setStatus(message);
  if (input) input.setAttribute("aria-expanded", "false");
  activeIndex = -1;
};

const escapeIntervals = (source, terms) => {
  const lowerSource = source.toLocaleLowerCase();
  const intervals = [];
  terms.forEach((rawTerm) => {
    const term = String(rawTerm || "").toLocaleLowerCase().trim();
    if (term.length < 2) return;
    let position = 0;
    while ((position = lowerSource.indexOf(term, position)) >= 0) {
      intervals.push([position, position + term.length]);
      position += term.length;
    }
  });
  intervals.sort((left, right) => left[0] - right[0] || right[1] - left[1]);
  return intervals.reduce((merged, interval) => {
    const previous = merged[merged.length - 1];
    if (!previous || interval[0] > previous[1]) merged.push(interval);
    else previous[1] = Math.max(previous[1], interval[1]);
    return merged;
  }, []);
};

const highlightedText = (value, query, matchedTerms) => {
  const source = String(value || "");
  const fragment = document.createDocumentFragment();
  const intervals = escapeIntervals(source, [query, ...(matchedTerms || [])]);
  let cursor = 0;
  intervals.forEach(([start, end]) => {
    if (start > cursor) fragment.append(document.createTextNode(source.slice(cursor, start)));
    const mark = document.createElement("mark");
    mark.className = "search-keyword";
    mark.textContent = source.slice(start, end);
    fragment.append(mark);
    cursor = end;
  });
  if (cursor < source.length) fragment.append(document.createTextNode(source.slice(cursor)));
  return fragment;
};

const resultTitle = (result) => {
  if (getLanguage() === "zh") return result.titleZh || result.titleEn;
  return result.titleEn || result.titleZh;
};

const createResultItem = (result, query, index) => {
  const item = document.createElement("li");
  item.className = "search-result-item";
  const link = document.createElement("a");
  link.className = "search-result-link";
  link.href = result.url;
  link.id = `search-option-${index}`;
  link.setAttribute("role", "option");
  link.setAttribute("aria-selected", "false");

  const titleElement = document.createElement("span");
  titleElement.className = "search-result-title";
  titleElement.append(highlightedText(resultTitle(result), query, result.matchedTerms));
  link.append(titleElement);

  const metadata = document.createElement("span");
  metadata.className = "search-result-meta";
  const metadataItems = [
    result.date ? String(result.date).slice(0, 10) : "",
    ...(result.categories || []).slice(0, 2),
    ...(result.tags || []).slice(0, 2),
  ].filter(Boolean);
  metadata.textContent = metadataItems.join(" · ");
  if (metadata.textContent) link.append(metadata);

  if (result.heading) {
    const heading = document.createElement("span");
    heading.className = "search-result-heading";
    heading.append(highlightedText(result.heading, query, result.matchedTerms));
    link.append(heading);
  }

  if (result.snippet) {
    const snippet = document.createElement("span");
    snippet.className = "search-result-snippet";
    snippet.append(highlightedText(result.snippet, query, result.matchedTerms));
    link.append(snippet);
  }

  item.append(link);
  return item;
};

const renderResults = (response, partial) => {
  const { result, input } = getSearchDom();
  if (!result || !input) return;
  latestResponse = response;
  latestPartial = partial;
  activeIndex = -1;

  if (response.status === "short_chinese") {
    renderState("fa-language", text("shortChinese"));
    return;
  }
  if (response.status !== "ok") {
    renderState("fa-magnifying-glass", text("idle"));
    return;
  }
  if (response.results.length === 0) {
    if (partial) renderState("fa-spinner fa-spin-pulse", text("deepLoading"), { busy: true });
    else renderState("fa-box-open", text("noResults"));
    return;
  }

  const list = document.createElement("ul");
  list.className = "search-result-list";
  response.results.forEach((searchResult, index) => {
    list.append(createResultItem(searchResult, response.normalizedQuery, index));
  });
  result.replaceChildren(list);
  input.setAttribute("aria-expanded", "true");
  setBusy(false);
  const status = partial ? `${text("resultCount", response.results.length)}. ${text("deepLoading")}` : text("resultCount", response.results.length);
  setStatus(archiveFailed ? `${status}. ${text("archiveWarning")}` : status);
  window.pjax?.refresh(result);
};

const resetWorker = () => {
  worker?.terminate();
  worker = null;
  workerReadyPromise = null;
  resolveWorkerReady = null;
  rejectWorkerReady = null;
  archiveFailed = false;
};

const handleWorkerMessage = (event) => {
  const message = event.data || {};
  if (message.type === "ready") {
    if (message.stage === "core" && resolveWorkerReady) {
      resolveWorkerReady();
      resolveWorkerReady = null;
      rejectWorkerReady = null;
      const { input } = getSearchDom();
      if (!input?.value.trim()) renderState("fa-magnifying-glass", text("idle"));
    }
    if (message.stage === "complete" && !latestResponse) setStatus(text("ready"));
    return;
  }
  if (message.type === "results") {
    if (message.id !== requestId) return;
    renderResults(message.response, Boolean(message.partial));
    return;
  }
  if (message.type === "archive-error") {
    archiveFailed = true;
    if (latestResponse) renderResults(latestResponse, false);
    else setStatus(text("archiveWarning"));
    return;
  }
  if (message.type === "error") {
    if (rejectWorkerReady) rejectWorkerReady(new Error(message.message));
    renderState("fa-triangle-exclamation", text("error"), { retry: true });
  }
};

const ensureWorker = () => {
  if (workerReadyPromise) return workerReadyPromise;
  renderState("fa-spinner fa-spin-pulse", text("loading"), { busy: true });
  workerReadyPromise = new Promise((resolve, reject) => {
    resolveWorkerReady = resolve;
    rejectWorkerReady = reject;
    try {
      worker = new Worker(assetUrl("assets/js/search-worker.js"));
      worker.addEventListener("message", handleWorkerMessage);
      worker.addEventListener("error", () => {
        reject(new Error("Search worker failed"));
        renderState("fa-triangle-exclamation", text("error"), { retry: true });
      }, { once: true });
      worker.postMessage({ type: "load", manifestUrl: getManifestUrl() });
    } catch (error) {
      reject(error);
      renderState("fa-triangle-exclamation", text("error"), { retry: true });
    }
  });
  return workerReadyPromise;
};

const submitQuery = async (query) => {
  try {
    await ensureWorker();
    requestId += 1;
    worker.postMessage({
      type: "query",
      id: requestId,
      query,
      limit: 20,
      manifestUrl: getManifestUrl(),
    });
  } catch (error) {
    renderState("fa-triangle-exclamation", text("error"), { retry: true });
  }
};

const queueQuery = (input) => {
  const query = input.value.trim();
  clearTimeout(debounceTimer);
  latestResponse = null;
  if (!query) {
    renderState("fa-magnifying-glass", text("idle"));
    return;
  }
  renderState("fa-spinner fa-spin-pulse", text("loading"), { busy: true });
  debounceTimer = setTimeout(() => submitQuery(query), 90);
};

const closePopup = () => {
  const { overlay, input } = getSearchDom();
  if (!overlay) return;
  document.body.style.overflow = "";
  overlay.classList.remove("active");
  input?.setAttribute("aria-expanded", "false");
};

const openPopup = () => {
  const { overlay, input } = getSearchDom();
  if (!overlay || !input) return;
  document.body.style.overflow = "hidden";
  overlay.classList.add("active");
  setTimeout(() => input.focus(), 180);
  ensureWorker().catch(() => {});
};

const updateActiveResult = (nextIndex) => {
  const { result, input } = getSearchDom();
  const options = [...(result?.querySelectorAll(".search-result-link") || [])];
  if (!options.length || !input) return;
  activeIndex = (nextIndex + options.length) % options.length;
  options.forEach((option, index) => {
    const active = index === activeIndex;
    option.classList.toggle("active", active);
    option.setAttribute("aria-selected", active ? "true" : "false");
    if (active) option.scrollIntoView({ block: "nearest" });
  });
  input.setAttribute("aria-activedescendant", options[activeIndex].id);
};

const handleInput = (event) => {
  if (!event.target.matches(".search-input")) return;
  queueQuery(event.target);
};

const handleClick = (event) => {
  if (event.target.closest(".search-popup-trigger")) {
    openPopup();
    return;
  }
  const overlay = event.target.closest(".search-pop-overlay");
  if (overlay && event.target === overlay) {
    closePopup();
    return;
  }
  if (event.target.closest(".search-input-field-pre")) {
    const { input } = getSearchDom();
    if (input) {
      input.value = "";
      input.focus();
      queueQuery(input);
    }
    return;
  }
  if (event.target.closest(".popup-btn-close")) {
    closePopup();
    return;
  }
  if (event.target.closest(".search-retry")) {
    const { input } = getSearchDom();
    resetWorker();
    if (input?.value.trim()) submitQuery(input.value.trim());
    else ensureWorker().catch(() => {});
  }
};

const handleKeydown = (event) => {
  if (event.key === "Escape") {
    closePopup();
    return;
  }
  if (!event.target.matches(".search-input")) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    updateActiveResult(activeIndex + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    updateActiveResult(activeIndex - 1);
  } else if (event.key === "Enter" && activeIndex >= 0) {
    const { result } = getSearchDom();
    const active = result?.querySelectorAll(".search-result-link")[activeIndex];
    if (active) {
      event.preventDefault();
      active.click();
    }
  }
};

export const initLocalSearchGlobals = ({ signal } = {}) => {
  if (didInit) return;
  didInit = true;
  const listenerOptions = signal ? { signal } : undefined;
  document.addEventListener("input", handleInput, listenerOptions);
  document.addEventListener("click", handleClick, listenerOptions);
  window.addEventListener("keydown", handleKeydown, listenerOptions);
  const observer = new MutationObserver(() => {
    if (latestResponse) renderResults(latestResponse, latestPartial);
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-lang"] });
  signal?.addEventListener("abort", () => observer.disconnect(), { once: true });
};

export const initLocalSearchPage = () => {
  closePopup();
  const { input } = getSearchDom();
  if (input && !input.value.trim()) renderState("fa-magnifying-glass", text("idle"));
  if (window.theme?.navbar?.search?.preload) ensureWorker().catch(() => {});
};
