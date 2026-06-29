// EN/ZH language toggle for the personal pages (Me / Projects / About).
// Self-contained, no theme-template edits. Loaded via inject.head so it lives
// in <head> and survives swup (single_page) navigations.
//
// Mechanism:
//   - window.I18N (from i18n.js) holds { en:{key:val}, zh:{key:val} }.
//   - Elements carry data-i18n="key"; we swap their innerHTML on apply.
//   - data-i18n-force-lang="en" pins an element to English regardless of mode.
//   - Choice persists in localStorage['lang'] ('en' | 'zh'); default 'zh'.
//   - Re-applied on DOMContentLoaded and on every swup page:view.
//
// The toggle only affects hand-authored [data-i18n] nodes (Me/Projects/About).
// It does NOT translate post bodies or the theme's own chrome.
(function () {
  "use strict";

  var STORAGE_KEY = "lang";
  var DEFAULT_LANG = "zh";

  function getLang() {
    try {
      var v = localStorage.getItem(STORAGE_KEY);
      return v === "en" || v === "zh" ? v : DEFAULT_LANG;
    } catch (e) {
      return DEFAULT_LANG;
    }
  }

  function setLang(lang) {
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
  }

  // Anti-FOUC: stamp the chosen language onto <html> as early as possible.
  function stampHtml(lang) {
    var el = document.documentElement;
    el.setAttribute("data-lang", lang);
  }

  // Theme chrome translation: the navbar/sidebar/footer labels are baked at
  // build time in zh-CN. We map them at runtime so the toggle is visible on
  // Home (which has no [data-i18n] nodes). Navbar items are matched by href
  // (robust); sidebar/footer by their canonical zh<->en text pairs.
  var NAV_LABELS = {
    "/": { zh: "首页", en: "Home" },
    "/archives/": { zh: "归档", en: "Archives" },
    "/me/": { zh: "Me", en: "Me" },
    "/projects/": { zh: "Project", en: "Project" },
    "/murmur/": { zh: "碎碎念", en: "Murmurs" },
    "/footprints/": { zh: "Footprints", en: "Footprints" },
    "/friends/": { zh: "Friends", en: "Friends" },
    "/cv/": { zh: "CV", en: "CV" },
    "/contact/": { zh: "Contact", en: "Contact" }
  };
  // Dropdown parent "About/关于" has no single href; match by text.
  var TEXT_LABELS = [
    { zh: "关于", en: "About" },
    { zh: "归档", en: "Archives" },
    { zh: "分类", en: "Categories" },
    { zh: "标签", en: "Tags" },
    { zh: "文章", en: "Posts" },
    { zh: "访问人数", en: "Visitors" },
    { zh: "总访问量", en: "Views" },
    { zh: "主题", en: "Theme" }
  ];

  function setLeafText(el, text) {
    // replace only the trailing text node (preserve any leading <i> icon)
    var done = false;
    for (var i = el.childNodes.length - 1; i >= 0; i--) {
      var n = el.childNodes[i];
      if (n.nodeType === 3 && n.textContent.trim()) { n.textContent = text; done = true; break; }
    }
    if (!done) {
      // element wraps text in a <span> with no own text node; try a child span
      var span = el.querySelector("span");
      if (span) span.textContent = text; else el.textContent = text;
    }
  }

  function applyChrome(lang) {
    // 1) navbar links by href (desktop + mobile drawer)
    var navSel = ".navbar-list a, .navbar-item a, .drawer-navbar-list a, .drawer-navbar-item a, .sidebar-links a";
    document.querySelectorAll(navSel).forEach(function (a) {
      var href = a.getAttribute("href");
      if (!href) return;
      var map = NAV_LABELS[href];
      if (map) setLeafText(a, map[lang]);
    });
    // 2) generic text labels (dropdown parent, sidebar stat labels, footer)
    //    Walk small text-bearing nodes and swap canonical pairs.
    var textSel = ".navbar-item > a, .drawer-navbar-item, .statistics .label, .sidebar-links .link-name, .footer span, footer span";
    document.querySelectorAll(textSel).forEach(function (el) {
      var cur = el.textContent.trim();
      for (var i = 0; i < TEXT_LABELS.length; i++) {
        var pair = TEXT_LABELS[i];
        if (cur === pair.zh || cur === pair.en) {
          if (cur !== pair[lang]) setLeafText(el, pair[lang]);
          break;
        }
      }
    });
  }

  // Blog list i18n: swap post titles/excerpts to their English fields in EN
  // mode (Home feed, archives, categories, tags). Data comes from a generated
  // map {url: {title_en, excerpt_en}} so no theme-template edits are needed.
  var POST_I18N = null;
  var POST_I18N_PENDING = false;

  function normUrl(href) {
    if (!href) return "";
    // strip origin, query/hash; ensure single leading slash + trailing slash
    href = href.replace(/^https?:\/\/[^/]+/, "").split(/[?#]/)[0];
    if (href.charAt(0) !== "/") href = "/" + href;
    if (href.charAt(href.length - 1) !== "/") href += "/";
    return href;
  }

  function swapPostTitles(lang) {
    if (!POST_I18N) return;
    var links = document.querySelectorAll('a[href*="/blog/"]');
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var rec = POST_I18N[normUrl(a.getAttribute("href"))];
      if (!rec || !rec.title_en) continue;
      // skip "阅读全文"/"read more" links (they carry a hidden seo span)
      if (a.querySelector(".seo-reader-text")) continue;
      // The title text may live directly in the <a> (home cards) or in a child
      // .article-title span (archives / category / tag list). Pick the element
      // that actually holds the title text and swap only that.
      var el = a.querySelector(".article-title") || a;
      // skip links that wrap non-title content (e.g. a thumbnail image only)
      if (el === a && (!a.textContent.trim() || a.querySelector("img"))) continue;
      if (el.dataset.zhTitle === undefined) el.dataset.zhTitle = el.textContent.trim();
      var target = lang === "en" ? rec.title_en : el.dataset.zhTitle;
      if (target && el.textContent.trim() !== target) el.textContent = target;
    }
    // Home card excerpts: each card's title link gives the URL; the excerpt is
    // the sibling .home-article-content in the same article item.
    var items = document.querySelectorAll(".home-article-item");
    for (var j = 0; j < items.length; j++) {
      var item = items[j];
      var titleLink = item.querySelector(".home-article-title a");
      var exc = item.querySelector(".home-article-content");
      if (!titleLink || !exc) continue;
      var r = POST_I18N[normUrl(titleLink.getAttribute("href"))];
      if (!r || !r.excerpt_en) continue;
      if (exc.dataset.zhHtml === undefined) exc.dataset.zhHtml = exc.innerHTML;
      if (lang === "en") {
        exc.innerHTML = "<p>" + r.excerpt_en + "</p>";
      } else if (exc.dataset.zhHtml !== undefined) {
        exc.innerHTML = exc.dataset.zhHtml;
      }
    }
  }

  function applyPostI18n(lang) {
    if (POST_I18N) { swapPostTitles(lang); return; }
    if (POST_I18N_PENDING) return;
    POST_I18N_PENDING = true;
    fetch("/assets/data/post-i18n.json")
      .then(function (r) { return r.json(); })
      .then(function (m) { POST_I18N = m || {}; swapPostTitles(getLang()); })
      .catch(function () { POST_I18N = {}; })
      .then(function () { POST_I18N_PENDING = false; });
  }

  function applyI18n() {
    var dict = window.I18N;
    if (!dict) return;
    var lang = getLang();
    stampHtml(lang);

    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      var key = node.getAttribute("data-i18n");
      if (!key) continue;
      var forced = node.getAttribute("data-i18n-force-lang");
      var useLang = forced === "en" || forced === "zh" ? forced : lang;
      var table = dict[useLang] || {};
      var fallback = dict.zh || {};
      var val = table[key];
      if (val == null) val = fallback[key];
      if (val != null) node.innerHTML = val;
    }

    // Update the toggle button label to show the OTHER language as the action.
    var btn = document.getElementById("language-toggle");
    if (btn) btn.textContent = lang === "en" ? "中文" : "EN";

    // Translate the theme chrome (navbar / sidebar / footer) so the toggle has a
    // visible effect on Home and every page, where there are no [data-i18n] nodes.
    applyChrome(lang);

    // Swap blog titles/excerpts in lists (Home / archives / categories / tags)
    // to their English fields when in EN mode. Post bodies stay Chinese.
    applyPostI18n(lang);

    // Notify dependent modules (e.g. the globe tooltips).
    if (typeof window.updateGlobeLanguage === "function") {
      try { window.updateGlobeLanguage(lang); } catch (e) {}
    }
    if (typeof window.renderMurmur === "function") {
      try { window.renderMurmur(); } catch (e) {}
    }
    document.dispatchEvent(new CustomEvent("i18n:applied", { detail: { lang: lang } }));
  }

  function toggleLang() {
    setLang(getLang() === "en" ? "zh" : "en");
    applyI18n();
  }

  // Bind the toggle button exactly once (button lives in inject.footer,
  // outside #swup, so it persists; binding once is enough).
  function bindButton() {
    var btn = document.getElementById("language-toggle");
    if (!btn || btn.dataset.bound === "1") return;
    btn.dataset.bound = "1";
    btn.addEventListener("click", toggleLang);
  }

  function init() {
    bindButton();
    applyI18n();
  }

  // Stamp ASAP to reduce flash.
  stampHtml(getLang());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // swup re-init: content is swapped without a reload, so re-apply on every view.
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
