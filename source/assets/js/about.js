// About page interactive modules: murmur timeline + Globe.gl footprints.
// Loaded via inject.head (persistent in <head>), but all work is guarded to
// run ONLY on /about/ and is swup-safe:
//   - re-init on swup page:view, teardown WebGL on visit:start (no context leak)
//   - globe.gl library injected once into <head>; only init() re-runs
(function () {
  "use strict";

  var GLOBE_SRC = "https://cdn.jsdelivr.net/npm/globe.gl@2";
  var EARTH_TEX = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-dark.jpg";

  function onAbout() {
    return location.pathname.replace(/\/+$/, "") === "/about" ||
           location.pathname.indexOf("/about/") === 0;
  }

  function currentLang() {
    try {
      var v = localStorage.getItem("lang");
      return v === "en" ? "en" : "zh";
    } catch (e) { return "zh"; }
  }

  // ---- Murmur timeline -----------------------------------------------------
  function renderMurmur() {
    var box = document.getElementById("murmur-timeline");
    if (!box) return;
    if (box.dataset.loading === "1") return;
    if (window.__murmurData) {
      paintMurmur(box, window.__murmurData);
      return;
    }
    box.dataset.loading = "1";
    fetch("/assets/data/murmur.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        window.__murmurData = data;
        box.dataset.loading = "0";
        paintMurmur(box, data);
      })
      .catch(function () {
        box.dataset.loading = "0";
        box.innerHTML = '<p class="murmur-empty">…</p>';
      });
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function paintMurmur(box, data) {
    var items = (data || []).slice().sort(function (a, b) {
      return (b.date || "").localeCompare(a.date || "");
    });
    var html = items.map(function (m) {
      var meta = [];
      if (m.author) meta.push(escapeHtml(m.author));
      if (m.source) meta.push("《" + escapeHtml(m.source) + "》");
      if (m.date) meta.push(escapeHtml(m.date));
      var moodTag = m.mood ? '<span class="murmur-mood">' + escapeHtml(m.mood) + "</span>" : "";
      return (
        '<li class="murmur-item">' +
          '<div class="murmur-text">' + escapeHtml(m.text) + "</div>" +
          '<div class="murmur-meta">' + meta.join(" · ") + " " + moodTag + "</div>" +
        "</li>"
      );
    }).join("");
    box.innerHTML = '<ul class="murmur-list">' + html + "</ul>";
  }
  // expose so lang-toggle can re-render (meta order is language-agnostic, but
  // keeping the hook makes future localization easy)
  window.renderMurmur = renderMurmur;

  // ---- Globe.gl footprints -------------------------------------------------
  var TYPE_COLORS = {
    "学习": "#5b8cff", "Study": "#5b8cff",
    "旅游": "#ffb347", "Travel": "#ffb347",
    "家": "#f54545", "Home": "#f54545"
  };

  function loadGlobeLib(cb) {
    if (window.Globe) { cb(); return; }
    if (window.__globeLibLoading) {
      var t = setInterval(function () {
        if (window.Globe) { clearInterval(t); cb(); }
      }, 60);
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
        lat: v[1],
        lng: v[0],
        size: typeof visits === "number" ? Math.min(0.25, 0.07 + visits * 0.03) : 0.12,
        color: TYPE_COLORS[c.type] || TYPE_COLORS[c.typeEn] || "#5b8cff",
        label: label,
        typ: typ,
        detail: detail
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
          // gentle auto-rotate
          try {
            g.controls().autoRotate = true;
            g.controls().autoRotateSpeed = 0.6;
          } catch (e) {}
          // size to container
          var w = el.clientWidth || 600;
          g.width(w).height(Math.min(480, Math.max(320, w * 0.7)));
          // focus on China-ish view
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

  // ---- CV: gracefully handle a missing placeholder PDF -------------------
  function checkCv() {
    var frame = document.getElementById("cv-frame");
    var ph = document.getElementById("cv-placeholder");
    if (!frame || !ph || frame.dataset.checked === "1") return;
    frame.dataset.checked = "1";
    fetch("/assets/cv.pdf", { method: "HEAD" })
      .then(function (r) {
        if (!r.ok) {
          frame.hidden = true;
          ph.hidden = false;
        }
      })
      .catch(function () {
        frame.hidden = true;
        ph.hidden = false;
      });
  }

  // ---- lifecycle -----------------------------------------------------------
  function initAbout() {
    if (!onAbout()) return;
    renderMurmur();
    checkCv();
    initGlobe();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAbout);
  } else {
    initAbout();
  }

  function hookSwup(swup) {
    if (!swup || !swup.hooks || swup.__aboutHooked) return;
    swup.__aboutHooked = true;
    swup.hooks.on("visit:start", teardownGlobe);
    swup.hooks.on("page:view", initAbout);
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
