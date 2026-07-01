// EN/ZH language controls for the site chrome and hand-authored pages.
// English is the system default. Chinese-only post bodies and murmurs are
// treated as content islands; surrounding UI still follows the selected lang.
(function () {
  "use strict";

  var STORAGE_KEY = "lang";
  var DEFAULT_LANG = "en";
  var DEFAULT_LANG_MIGRATION_KEY = "lang-default-migration";
  var DEFAULT_LANG_MIGRATION = "en-gear-tools-2026-07";

  function getLang() {
    try {
      var v = localStorage.getItem(STORAGE_KEY);
      if (localStorage.getItem(DEFAULT_LANG_MIGRATION_KEY) !== DEFAULT_LANG_MIGRATION) {
        localStorage.setItem(DEFAULT_LANG_MIGRATION_KEY, DEFAULT_LANG_MIGRATION);
        if (v !== DEFAULT_LANG) {
          localStorage.setItem(STORAGE_KEY, DEFAULT_LANG);
          return DEFAULT_LANG;
        }
      }
      return v === "en" || v === "zh" ? v : DEFAULT_LANG;
    } catch (e) {
      return DEFAULT_LANG;
    }
  }

  function setLang(lang) {
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
  }

  function stampHtml(lang) {
    var el = document.documentElement;
    el.setAttribute("data-lang", lang);
    el.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
  }

  function safeDecode(s) {
    try { return decodeURIComponent(s); } catch (e) { return s; }
  }

  function normPath(href) {
    if (!href) return "";
    href = href.replace(/^https?:\/\/[^/]+/, "").split(/[?#]/)[0];
    if (href.charAt(0) !== "/") href = "/" + href;
    if (href.charAt(href.length - 1) !== "/") href += "/";
    return href;
  }

  function pathVariants(href) {
    var raw = normPath(href);
    var decoded = safeDecode(raw);
    var encoded = encodeURI(decoded);
    var variants = [raw, decoded, encoded];
    var out = [];
    variants.forEach(function (v) {
      if (v && out.indexOf(v) === -1) out.push(v);
    });
    return out;
  }

  function getCurrentPath() {
    return normPath(window.location.pathname || "/");
  }

  function setLeafText(el, text) {
    var done = false;
    for (var i = el.childNodes.length - 1; i >= 0; i--) {
      var n = el.childNodes[i];
      if (n.nodeType === 3 && n.textContent.trim()) {
        n.textContent = text;
        done = true;
        break;
      }
    }
    if (!done) {
      var span = el.querySelector("span:not(.lang-toggle-mark):not(.sr-only)");
      if (span) span.textContent = text; else el.textContent = text;
    }
  }

  function exactText(el, from, to) {
    if (!el) return;
    var cur = el.textContent.trim();
    if (cur === from) setLeafText(el, to);
  }

  var NAV_LABELS = {
    "/": { zh: "首页", en: "Home" },
    "/archives/": { zh: "归档", en: "Archives" },
    "/me/": { zh: "我", en: "Me" },
    "/projects/": { zh: "项目", en: "Project" },
    "/murmur/": { zh: "碎碎念", en: "Murmur" },
    "/footprints/": { zh: "Footprints", en: "Footprints" },
    "/friends/": { zh: "Friends", en: "Friends" },
    "/cv/": { zh: "CV", en: "CV" },
    "/categories/": { zh: "分类", en: "Categories" },
    "/tags/": { zh: "标签", en: "Tags" }
  };

  var PAGE_LABELS = {
    "/archives/": { zh: "归档", en: "Archive" },
    "/me/": { zh: "我", en: "Me" },
    "/projects/": { zh: "项目", en: "Project" },
    "/murmur/": { zh: "碎碎念", en: "Murmur" },
    "/footprints/": { zh: "Footprints", en: "Footprints" },
    "/friends/": { zh: "Friends", en: "Friends" },
    "/cv/": { zh: "CV", en: "CV" },
    "/categories/": { zh: "分类", en: "Categories" },
    "/tags/": { zh: "标签", en: "Tags" }
  };

  var TEXT_LABELS = [
    { zh: "关于", en: "About" },
    { zh: "关于", en: "ABOUT" },
    { zh: "归档", en: "Archive" },
    { zh: "归档", en: "Archives" },
    { zh: "分类", en: "Category" },
    { zh: "分类", en: "Categories" },
    { zh: "标签", en: "Tag" },
    { zh: "标签", en: "Tags" },
    { zh: "文章", en: "Post" },
    { zh: "文章", en: "Posts" },
    { zh: "访问人数", en: "VISITOR COUNT" },
    { zh: "总访问量", en: "TOTAL PAGE VIEWS" },
    { zh: "访问人数", en: "Visitors" },
    { zh: "总访问量", en: "Views" },
    { zh: "主题", en: "THEME" },
    { zh: "主题", en: "Theme" },
    { zh: "创建", en: "Created" },
    { zh: "更新", en: "Updated" },
    { zh: "博客已运行", en: "Blog up for" },
    { zh: "天", en: "days" },
    { zh: "小时", en: "hrs" },
    { zh: "分钟", en: "Min" },
    { zh: "秒", en: "Sec" },
    { zh: "阅读全文", en: "Read more" }
  ];

  function translateExactText(el, lang) {
    var cur = el.textContent.trim();
    for (var i = 0; i < TEXT_LABELS.length; i++) {
      var pair = TEXT_LABELS[i];
      if (cur === pair.zh || cur === pair.en) {
        if (cur !== pair[lang]) setLeafText(el, pair[lang]);
        break;
      }
    }
  }

  function applyChrome(lang) {
    var navSel = ".navbar-list a, .navbar-item a, .drawer-navbar-list a, .drawer-navbar-item a, .sidebar-links a";
    document.querySelectorAll(navSel).forEach(function (a) {
      var map = NAV_LABELS[normPath(a.getAttribute("href"))];
      if (map) setLeafText(a, map[lang]);
    });

    var textSel = [
      ".navbar-item > a",
      ".drawer-navbar-item",
      ".drawer-navbar-item-sub [navbar-data-toggle] > span",
      ".statistics .label",
      ".sidebar-links .link-name",
      ".footer span",
      "footer span",
      ".article-meta-info .hover-info"
    ].join(", ");
    document.querySelectorAll(textSel).forEach(function (el) {
      translateExactText(el, lang);
    });

    document.querySelectorAll(".post-count span").forEach(function (el) {
      var text = el.textContent.trim();
      var m;
      if (lang === "en") {
        m = text.match(/^共撰写了\s*(\d+)\s*篇文章$/);
        if (m) el.textContent = m[1] + " posts in total";
      } else {
        m = text.match(/^(\d+)\s+posts in total$/i);
        if (m) el.textContent = "共撰写了 " + m[1] + " 篇文章";
      }
    });
  }

  function setDocumentTitle(target, knownLabels) {
    if (!target) return;
    var current = document.title || "";
    var suffix = "";
    var sep = " | ";
    var idx = current.indexOf(sep);
    if (idx !== -1) suffix = current.slice(idx);
    if (!suffix) suffix = " | Hyacehila's Blog";
    document.title = target + suffix;

    document.querySelectorAll('meta[property="og:title"], meta[name="twitter:title"]').forEach(function (meta) {
      var content = meta.getAttribute("content") || "";
      if (!knownLabels || knownLabels.indexOf(content) !== -1) meta.setAttribute("content", target);
    });
  }

  function applyPageLabel(lang) {
    var map = PAGE_LABELS[getCurrentPath()];
    if (!map) return;
    var target = map[lang];
    var labels = [map.zh, map.en];

    document.querySelectorAll(".page-template-content > h1, .page-title-header").forEach(function (h1) {
      var cur = h1.textContent.trim();
      if (labels.indexOf(cur) !== -1) h1.textContent = target;
    });
    setDocumentTitle(target, labels);
  }

  var POST_I18N = null;
  var POST_I18N_PENDING = false;
  var READ_MORE_LABELS = {
    zh: "阅读全文",
    en: "Read more"
  };

  function postRecordFor(href) {
    if (!POST_I18N) return null;
    var keys = pathVariants(href);
    for (var i = 0; i < keys.length; i++) {
      if (POST_I18N[keys[i]]) return POST_I18N[keys[i]];
    }
    return null;
  }

  function swapPostTitles(lang) {
    if (!POST_I18N) return;
    document.querySelectorAll('a[href*="/blog/"]').forEach(function (a) {
      var rec = postRecordFor(a.getAttribute("href"));
      if (!rec || !rec.title_en) return;
      if (a.querySelector(".seo-reader-text")) return;

      var el = a.querySelector(".article-title") || a;
      if (el === a && (!a.textContent.trim() || a.querySelector("img"))) return;
      if (el.dataset.zhTitle === undefined) el.dataset.zhTitle = rec.title_zh || el.textContent.trim();
      var target = lang === "en" ? rec.title_en : el.dataset.zhTitle;
      if (target && el.textContent.trim() !== target) el.textContent = target;
    });

    document.querySelectorAll(".home-article-item").forEach(function (item) {
      var titleLink = item.querySelector(".home-article-title a");
      var exc = item.querySelector(".home-article-content");
      if (!titleLink || !exc) return;
      var rec = postRecordFor(titleLink.getAttribute("href"));
      if (!rec || !rec.excerpt_en) return;
      if (exc.dataset.zhHtml === undefined) exc.dataset.zhHtml = exc.innerHTML;
      if (lang === "en") {
        exc.innerHTML = "<p>" + rec.excerpt_en + "</p>";
      } else if (rec.excerpt_zh) {
        exc.innerHTML = "<p>" + rec.excerpt_zh + "</p>";
      } else {
        exc.innerHTML = exc.dataset.zhHtml;
      }
    });
  }

  function applyPostPageI18n(lang) {
    if (!POST_I18N || getCurrentPath().indexOf("/blog/") !== 0) return;
    var rec = postRecordFor(window.location.pathname);
    if (!rec || !rec.title_en) return;
    var titleNode = document.querySelector(".article-title-regular");
    if (titleNode) {
      if (titleNode.dataset.zhTitle === undefined) titleNode.dataset.zhTitle = rec.title_zh || titleNode.textContent.trim();
      titleNode.textContent = lang === "en" ? rec.title_en : titleNode.dataset.zhTitle;
    }
    setDocumentTitle(lang === "en" ? rec.title_en : (rec.title_zh || rec.title_en), [rec.title_en, rec.title_zh]);
  }

  function setReadMoreText(a, text) {
    var seo = a.querySelector(".seo-reader-text");
    if (!seo) return;
    for (var i = 0; i < a.childNodes.length; i++) {
      var n = a.childNodes[i];
      if (n === seo) break;
      if (n.nodeType === 3) {
        n.textContent = text;
        return;
      }
    }
    a.insertBefore(document.createTextNode(text), seo);
  }

  function applyReadMoreLinks(lang) {
    var label = READ_MORE_LABELS[lang] || READ_MORE_LABELS.en;
    document.querySelectorAll(".home-article-meta-info-container > a").forEach(function (a) {
      if (a.querySelector(".seo-reader-text")) setReadMoreText(a, label);
    });
  }

  function applyPostI18n(lang) {
    if (POST_I18N) {
      swapPostTitles(lang);
      applyPostPageI18n(lang);
      return;
    }
    if (POST_I18N_PENDING) return;
    POST_I18N_PENDING = true;
    fetch("/assets/data/post-i18n.json")
      .then(function (r) { return r.json(); })
      .then(function (m) {
        POST_I18N = m || {};
        var current = getLang();
        swapPostTitles(current);
        applyPostPageI18n(current);
      })
      .catch(function () { POST_I18N = {}; })
      .then(function () { POST_I18N_PENDING = false; });
  }

  function ensureToggleButton() {
    if (getCurrentPath() === "/me/") {
      var existing = document.getElementById("language-toggle");
      if (existing) existing.remove();
      return null;
    }

    var list = document.querySelector(".hidden-tools-list");
    if (!list) return null;

    var btn = document.getElementById("language-toggle");
    if (!btn || btn.parentElement !== list) {
      if (btn) btn.remove();
      btn = document.createElement("li");
      btn.id = "language-toggle";
      btn.className = "right-bottom-tools tool-language-toggle flex justify-center items-center";
      btn.setAttribute("role", "button");
      btn.setAttribute("tabindex", "0");
      btn.innerHTML = '<i class="fa-solid fa-globe" aria-hidden="true"></i><span class="lang-toggle-mark" aria-hidden="true"></span><span class="sr-only">Switch language</span>';

      var dark = list.querySelector(".tool-dark-light-toggle");
      if (dark && dark.nextSibling) list.insertBefore(btn, dark.nextSibling);
      else list.appendChild(btn);
    }
    return btn;
  }

  function updateToggleButton(lang) {
    var btn = ensureToggleButton();
    if (!btn) return;
    var label = lang === "en" ? "Switch to Chinese" : "Switch to English";
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
    btn.setAttribute("data-current-lang", lang);
    var mark = btn.querySelector(".lang-toggle-mark");
    if (mark) mark.textContent = lang === "en" ? "EN" : "中";
    var hiddenText = btn.querySelector(".sr-only");
    if (hiddenText) hiddenText.textContent = label;
  }

  function applyDataI18n(lang) {
    var dict = window.I18N;
    if (!dict) return;
    document.querySelectorAll("[data-i18n]").forEach(function (node) {
      var key = node.getAttribute("data-i18n");
      if (!key) return;
      var forced = node.getAttribute("data-i18n-force-lang");
      var useLang = forced === "en" || forced === "zh" ? forced : lang;
      var table = dict[useLang] || {};
      var fallback = dict.en || {};
      var val = table[key];
      if (val == null) val = fallback[key];
      if (val != null) node.innerHTML = val;
    });
  }

  function applyI18n() {
    var lang = getCurrentPath() === "/me/" ? DEFAULT_LANG : getLang();
    stampHtml(lang);
    applyDataI18n(lang);
    updateToggleButton(lang);
    applyChrome(lang);
    applyPageLabel(lang);
    applyPostI18n(lang);
    applyReadMoreLinks(lang);

    if (typeof window.updateGlobeLanguage === "function") {
      try { window.updateGlobeLanguage(lang); } catch (e) {}
    }
    document.dispatchEvent(new CustomEvent("i18n:applied", { detail: { lang: lang } }));
  }

  function toggleLang() {
    setLang(getLang() === "en" ? "zh" : "en");
    applyI18n();
  }

  function bindButton() {
    var btn = ensureToggleButton();
    if (!btn || btn.dataset.bound === "1") return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", toggleLang);
    btn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleLang();
      }
    });
  }

  function init() {
    bindButton();
    applyI18n();
  }

  stampHtml(getLang());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  function hookSwup(swup) {
    if (!swup || !swup.hooks || swup.__i18nHooked) return;
    swup.__i18nHooked = true;
    swup.hooks.on("page:view", function () { init(); });
  }

  if (window.swup) {
    hookSwup(window.swup);
  } else {
    window.addEventListener(
      "redefine:swup:ready",
      function (e) { hookSwup(e.detail && e.detail.swup ? e.detail.swup : window.swup); },
      { once: true }
    );
  }
})();
