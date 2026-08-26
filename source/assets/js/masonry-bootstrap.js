(function () {
  "use strict";

  // Swup's delegated listener is installed later in the document. Intercept
  // fixed-route links first so a cross-root language change is a clean reload.
  document.addEventListener("click", function (event) {
    var link = event.target && event.target.closest ? event.target.closest("a[data-no-swup]") : null;
    if (link) event.stopImmediatePropagation();
  }, true);

  // Redefine loads its Masonry module and MiniMasonry through separate script
  // paths. This small bootstrap waits for both and retries after Swup views so
  // the gallery also works when the user enters it from the home page.
  var active = new WeakSet();

  function loadImages(container) {
    var images = container.querySelectorAll("img[data-src]");
    images.forEach(function (img) {
      if (img.dataset.masonryLoaded === "1") return;
      img.src = img.getAttribute("data-src");
      img.dataset.masonryLoaded = "1";
    });
  }

  function render(container, items) {
    if (!Array.isArray(items) || !items.length || active.has(container)) return;
    if (container.children.length) return;
    active.add(container);
    var fragment = document.createDocumentFragment();
    items.forEach(function (item) {
      if (!item || !item.image) return;
      var card = document.createElement("div");
      card.className = "masonry-item";
      var box = document.createElement("div");
      box.className = "image-container has-ratio";
      var width = Number(item.width) || 1;
      var height = Number(item.height) || 1;
      box.style.setProperty("--masonry-aspect-ratio", width + " / " + height);
      var img = document.createElement("img");
      img.className = "masonry-img";
      img.alt = item.title || "Photo";
      img.width = width;
      img.height = height;
      img.loading = "lazy";
      img.decoding = "async";
      img.setAttribute("data-src", item.image);
      img.src = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";
      box.appendChild(img);
      if (item.title) {
        var title = document.createElement("div");
        title.className = "image-title";
        title.textContent = item.title;
        box.appendChild(title);
      }
      card.appendChild(box);
      fragment.appendChild(card);
    });
    container.appendChild(fragment);
    loadImages(container);
    if (typeof MiniMasonry === "function") {
      try {
        new MiniMasonry({ baseWidth: window.innerWidth >= 768 ? 255 : 150, container: container, gutterX: 10, gutterY: 10, surroundingGutter: false });
      } catch (error) {
        container.style.height = "auto";
      }
    }
    container.classList.remove("min-h-screen!");
  }

  function init() {
    var container = document.querySelector("#masonry-container");
    if (!container || container.children.length || active.has(container)) return;
    var url = container.getAttribute("data-masonry-data-url") || "/masonry/data.json";
    fetch(url).then(function (response) {
      if (!response.ok) throw new Error("Masonry data request failed");
      return response.json();
    }).then(function (items) {
      window.setTimeout(function () { render(container, items); }, 0);
    }).catch(function () {});
  }

  function retry() {
    var attempts = 0;
    var tick = function () {
      attempts += 1;
      init();
      if (attempts < 12 && document.querySelector("#masonry-container")) window.setTimeout(tick, 300);
    };
    window.setTimeout(tick, 1200);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", retry);
  else retry();
  window.addEventListener("redefine:swup:ready", retry);
  if (window.swup && window.swup.hooks) window.swup.hooks.on("page:view", retry);
})();
