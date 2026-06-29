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
