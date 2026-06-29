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
    if (window.Globe && window.THREE) { cb(); return; }
    if (window.__globeLibLoading) {
      var t = setInterval(function () {
        if (window.Globe && window.THREE) { clearInterval(t); cb(); }
      }, 60);
      return;
    }
    window.__globeLibLoading = true;
    // Load THREE + globe.gl from the SAME ESM module graph (esm.sh) so they
    // share one THREE instance (a mismatched THREE breaks globe.gl's internal
    // instanceof checks). Expose both on window for the rest of about.js.
    var loader = document.createElement("script");
    loader.type = "module";
    loader.textContent = [
      "import * as THREE from 'https://esm.sh/three@0.179.0';",
      "import Globe from 'https://esm.sh/globe.gl@2?deps=three@0.179.0';",
      "window.THREE = THREE;",
      "window.Globe = Globe;",
      "window.dispatchEvent(new Event('globe-lib-ready'));"
    ].join("\n");
    loader.onerror = function () { window.__globeLibLoading = false; };
    window.addEventListener("globe-lib-ready", function () { cb(); }, { once: true });
    document.head.appendChild(loader);
  }

  // ---- Day/night terminator -------------------------------------------------
  // Blends a day texture and a night texture by the real sun direction, so the
  // globe shows the actual day/night boundary for the current moment.
  var DAY_TEX = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-day.jpg";
  var NIGHT_TEX = "https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-night.jpg";

  var DAYNIGHT_VERT = [
    "varying vec3 vNormal;",
    "varying vec2 vUv;",
    "void main() {",
    "  vNormal = normalize(normalMatrix * normal);",
    "  vUv = uv;",
    "  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
    "}"
  ].join("\n");

  var DAYNIGHT_FRAG = [
    "#define PI 3.141592653589793",
    "uniform sampler2D dayTexture;",
    "uniform sampler2D nightTexture;",
    "uniform vec2 sunPosition;",
    "uniform vec2 globeRotation;",
    "varying vec3 vNormal;",
    "varying vec2 vUv;",
    "float toRad(in float a) { return a * PI / 180.0; }",
    "vec3 Polar2Cartesian(in vec2 c) {",
    "  float theta = toRad(90.0 - c.x);",
    "  float phi = toRad(90.0 - c.y);",
    "  return vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));",
    "}",
    "void main() {",
    "  float invLon = toRad(globeRotation.x);",
    "  float invLat = -toRad(globeRotation.y);",
    "  mat3 rotX = mat3(1, 0, 0, 0, cos(invLat), -sin(invLat), 0, sin(invLat), cos(invLat));",
    "  mat3 rotY = mat3(cos(invLon), 0, sin(invLon), 0, 1, 0, -sin(invLon), 0, cos(invLon));",
    "  vec3 rotatedSunDirection = rotX * rotY * Polar2Cartesian(sunPosition);",
    "  float intensity = dot(normalize(vNormal), normalize(rotatedSunDirection));",
    "  vec4 dayColor = texture2D(dayTexture, vUv);",
    "  vec4 nightColor = texture2D(nightTexture, vUv);",
    "  float blendFactor = smoothstep(-0.1, 0.1, intensity);",
    "  gl_FragColor = mix(nightColor, dayColor, blendFactor);",
    "}"
  ].join("\n");

  // Sub-solar point [lng, lat] for a given Date (compact NOAA-style solar calc).
  function sunPosAt(dt) {
    var ms = +dt;
    var dayStart = new Date(ms);
    dayStart.setUTCHours(0, 0, 0, 0);
    // Julian centuries since J2000.0
    var jd = ms / 86400000 + 2440587.5;
    var t = (jd - 2451545.0) / 36525.0;
    var deg = Math.PI / 180;
    // geometric mean longitude & anomaly of the sun
    var L0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360;
    var M = 357.52911 + t * (35999.05029 - 0.0001537 * t);
    var e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
    var Mr = M * deg;
    var C = (1.914602 - t * (0.004817 + 0.000014 * t)) * Math.sin(Mr) +
            (0.019993 - 0.000101 * t) * Math.sin(2 * Mr) +
            0.000289 * Math.sin(3 * Mr);
    var trueLong = L0 + C;
    var omega = 125.04 - 1934.136 * t;
    var lambda = trueLong - 0.00569 - 0.00478 * Math.sin(omega * deg);
    // obliquity
    var seconds = 21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813));
    var eps0 = 23 + (26 + seconds / 60) / 60;
    var eps = eps0 + 0.00256 * Math.cos(omega * deg);
    // declination
    var declination = Math.asin(Math.sin(eps * deg) * Math.sin(lambda * deg)) / deg;
    // equation of time (minutes)
    var y = Math.tan(eps / 2 * deg); y = y * y;
    var Lr = L0 * deg;
    var Eqt = y * Math.sin(2 * Lr) - 2 * e * Math.sin(Mr) +
              4 * e * y * Math.sin(Mr) * Math.cos(2 * Lr) -
              0.5 * y * y * Math.sin(4 * Lr) - 1.25 * e * e * Math.sin(2 * Mr);
    Eqt = Eqt / deg * 4; // radians -> minutes
    var longitude = (dayStart.getTime() - ms) / 864e5 * 360 - 180;
    return [longitude - Eqt / 4, declination];
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
          // globe.gl ESM default export is a constructor: new Globe(el).
          // (UMD style Globe()(el) also exists; support both.)
          var GlobeCtor = window.Globe;
          var inst;
          try { inst = new GlobeCtor(el); }
          catch (e) { inst = GlobeCtor()(el); }
          var g = inst
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

          // Day/night terminator material (falls back to flat texture if THREE
          // or the shader is unavailable for any reason).
          var THREE = window.THREE;
          if (THREE && THREE.ShaderMaterial) {
            try {
              var loader = new THREE.TextureLoader();
              Promise.all([
                loader.loadAsync(DAY_TEX),
                loader.loadAsync(NIGHT_TEX)
              ]).then(function (tex) {
                var material = new THREE.ShaderMaterial({
                  uniforms: {
                    dayTexture: { value: tex[0] },
                    nightTexture: { value: tex[1] },
                    sunPosition: { value: new THREE.Vector2() },
                    globeRotation: { value: new THREE.Vector2() }
                  },
                  vertexShader: DAYNIGHT_VERT,
                  fragmentShader: DAYNIGHT_FRAG
                });
                g.globeMaterial(material)
                  .onZoom(function (pov) {
                    material.uniforms.globeRotation.value.set(pov.lng, pov.lat);
                  });
                // set the sun to the real current position (one-shot; the
                // terminator is correct for "now"). pass a fixed timestamp via
                // Date.now equivalent through performance origin to avoid
                // disallowed Date.now in some contexts:
                var now = new Date();
                material.uniforms.sunPosition.value.set.apply(
                  material.uniforms.sunPosition.value, sunPosAt(now)
                );
              }).catch(function () {
                g.globeImageUrl(EARTH_TEX);
              });
            } catch (e) {
              g.globeImageUrl(EARTH_TEX);
            }
          } else {
            g.globeImageUrl(EARTH_TEX);
          }
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
