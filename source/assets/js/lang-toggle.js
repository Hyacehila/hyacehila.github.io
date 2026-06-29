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
