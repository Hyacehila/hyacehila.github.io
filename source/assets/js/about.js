// Interactive modules for the About sub-pages: Globe.gl footprints (/footprints/)
// and the CV placeholder check (/cv/). Loaded via inject.head (persistent in
// <head>), swup-safe: re-init on page:view, teardown WebGL on visit:start.
// Each module is guarded by the existence of its container, so it runs only on
// the page that needs it — no path matching required.
(function () {
  "use strict";

  var GLOBE_SRC = "https://cdn.jsdelivr.net/npm/globe.gl@2";
  var EARTH_TEX = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-dark.jpg";

  function currentLang() {
    try {
      var v = localStorage.getItem("lang");
      return v === "en" ? "en" : "zh";
    } catch (e) { return "zh"; }
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ---- CV: gracefully handle a missing placeholder PDF -------------------
  function checkCv() {
    var frame = document.getElementById("cv-frame");
    var ph = document.getElementById("cv-placeholder");
    if (!frame || !ph || frame.dataset.checked === "1") return;
    frame.dataset.checked = "1";
    fetch("/assets/cv.pdf", { method: "HEAD" })
      .then(function (r) { if (!r.ok) { frame.hidden = true; ph.hidden = false; } })
      .catch(function () { frame.hidden = true; ph.hidden = false; });
  }

  // ---- Globe.gl footprints -------------------------------------------------
  var TYPE_COLORS = {
    "学习": "#5b8cff", "Study": "#5b8cff",
    "旅游": "#ffb347", "Travel": "#ffb347",
    "家": "#f54545", "Home": "#f54545"
  };

  function loadGlobeLib(cb) {
    if (window.Globe) { cb(); return; }
    if (window.__globeLibLoading) {
      var t = setInterval(function () { if (window.Globe) { clearInterval(t); cb(); } }, 60);
      return;
    }
    window.__globeLibLoading = true;
    var s = document.createElement("script");
    s.src = GLOBE_SRC;
    s.onload = function () { cb(); };
    s.onerror = function () { window.__globeLibLoading = false; };
    document.head.appendChild(s);
  }

  function buildPoints(cities, lang) {
    return cities.map(function (c) {
      var v = c.value || [0, 0];
      var label = lang === "en" ? (c.nameEn || c.name) : (c.name || c.nameEn);
      var typ = lang === "en" ? (c.typeEn || c.type) : (c.type || c.typeEn);
      var detail = lang === "en" ? (c.detailsEn || c.details) : (c.details || c.detailsEn);
      var visits = c.visits;
      return {
        lat: v[1], lng: v[0],
        size: typeof visits === "number" ? Math.min(0.25, 0.07 + visits * 0.03) : 0.12,
        color: TYPE_COLORS[c.type] || TYPE_COLORS[c.typeEn] || "#5b8cff",
        label: label, typ: typ, detail: detail
      };
    });
  }

  function initGlobe() {
    var el = document.getElementById("globe-container");
    if (!el) return;
    if (el.dataset.globeInit === "1") return;
    el.dataset.globeInit = "1";

    loadGlobeLib(function () {
      if (!window.Globe || !document.getElementById("globe-container")) return;
      fetch("/assets/data/cities.json")
        .then(function (r) { return r.json(); })
        .then(function (cities) {
          window.__globeCities = cities;
          var lang = currentLang();
          var g = window.Globe()(el)
            .globeImageUrl(EARTH_TEX)
            .backgroundColor("rgba(0,0,0,0)")
            .pointsData(buildPoints(cities, lang))
            .pointAltitude("size")
            .pointColor("color")
            .pointRadius(0.32)
            .pointLabel(function (d) {
              return '<div style="text-align:left;font-size:12px;line-height:1.5">' +
                '<b>' + escapeHtml(d.label) + '</b> · ' + escapeHtml(d.typ || "") +
                (d.detail ? '<br>' + escapeHtml(d.detail) : '') + '</div>';
            });
          try {
            g.controls().autoRotate = true;
            g.controls().autoRotateSpeed = 0.6;
          } catch (e) {}
          var w = el.clientWidth || 600;
          g.width(w).height(Math.min(480, Math.max(320, w * 0.7)));
          g.pointOfView({ lat: 30, lng: 110, altitude: 2.2 }, 0);
          window.__globe = g;
        })
        .catch(function () {});
    });
  }

  function updateGlobeLanguage(lang) {
    if (!window.__globe || !window.__globeCities) return;
    window.__globe.pointsData(buildPoints(window.__globeCities, lang || currentLang()));
  }
  window.updateGlobeLanguage = updateGlobeLanguage;

  function teardownGlobe() {
    if (window.__globe) {
      try { if (window.__globe._destructor) window.__globe._destructor(); } catch (e) {}
      window.__globe = null;
    }
    var el = document.getElementById("globe-container");
    if (el) { el.innerHTML = ""; el.dataset.globeInit = "0"; }
  }

  // ---- lifecycle -----------------------------------------------------------
  function initPage() {
    checkCv();
    initGlobe();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPage);
  } else {
    initPage();
  }

  function hookSwup(swup) {
    if (!swup || !swup.hooks || swup.__aboutHooked) return;
    swup.__aboutHooked = true;
    swup.hooks.on("visit:start", teardownGlobe);
    swup.hooks.on("page:view", initPage);
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
