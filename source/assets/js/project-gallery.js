/* Progressive enhancement: full descriptions remain readable without JS. */
(function () {
  'use strict';
  var PREVIEW_LINES = 4;
  var layoutFrame;
  function fitPreview(button) {
    var paragraph = button.closest('.gallery-description');
    var full = paragraph.querySelector('.gallery-full');
    var preview = paragraph.querySelector('.gallery-preview');
    var expanded = !button.hidden && button.getAttribute('aria-expanded') === 'true';
    var characters = Array.from(full.textContent.trim());
    // Use the actual font, available width and translated button in normal flow.
    var measure = paragraph.cloneNode(true);
    measure.removeAttribute('id');
    measure.classList.remove('gallery-collapsed');
    measure.classList.add('gallery-text-measure');
    measure.setAttribute('aria-hidden', 'true');
    measure.setAttribute('inert', '');
    measure.style.width = getComputedStyle(paragraph).width;
    var sample = measure.querySelector('.gallery-preview');
    var control = measure.querySelector('.gallery-toggle');
    measure.querySelector('.gallery-full').hidden = true;
    sample.hidden = false;
    control.hidden = true;
    control.removeAttribute('aria-controls');
    control.setAttribute('aria-expanded', 'false');
    paragraph.parentElement.appendChild(measure);
    var height = parseFloat(getComputedStyle(measure).lineHeight) * PREVIEW_LINES;
    sample.textContent = characters.join('');
    var truncated = parseFloat(getComputedStyle(measure).height) > height + 1;
    if (truncated) {
      control.hidden = false;
      var low = 0, high = characters.length;
      while (low < high) {
        var mid = Math.ceil((low + high) / 2);
        sample.textContent = characters.slice(0, mid).join('').trimEnd() + '…';
        if (parseFloat(getComputedStyle(measure).height) <= height + 1) low = mid;
        else high = mid - 1;
      }
      var text = characters.slice(0, low).join('');
      // Keep Latin words intact when the measured cut falls inside a word.
      if (/[a-zA-Z]$/.test(text) && /^[a-zA-Z]/.test(characters[low] || '')) text = text.replace(/[a-zA-Z]+$/, '');
      preview.textContent = text.replace(/[，。；！？、,.!?;:\s]+$/u, '') + '…';
    }
    measure.remove();
    preview.hidden = !truncated || expanded;
    full.hidden = truncated && !expanded;
    button.hidden = !truncated;
    button.setAttribute('aria-expanded', String(!truncated || expanded));
    paragraph.classList.toggle('gallery-collapsed', truncated && !expanded);
    paragraph.style.setProperty('--gallery-preview-height', height + 'px');
  }
  function equalizeCards() {
    document.querySelectorAll('.project-gallery').forEach(function (gallery) {
      var tallest = 0;
      gallery.querySelectorAll('.gallery-body').forEach(function (body) {
        // Measure the collapsed content even while its real card is expanded.
        var measure = body.cloneNode(true);
        measure.classList.add('gallery-measure');
        measure.setAttribute('aria-hidden', 'true');
        measure.setAttribute('inert', '');
        measure.querySelectorAll('[id]').forEach(function (node) { node.removeAttribute('id'); });
        var button = measure.querySelector('.gallery-toggle');
        if (!button.hidden) {
          measure.querySelector('.gallery-preview').hidden = false;
          measure.querySelector('.gallery-full').hidden = true;
          button.setAttribute('aria-expanded', 'false');
          measure.querySelector('.gallery-description').classList.add('gallery-collapsed');
        }
        body.parentElement.appendChild(measure);
        tallest = Math.max(tallest, measure.offsetHeight);
        measure.remove();
      });
      gallery.style.setProperty('--gallery-body-height', Math.ceil(tallest) + 'px');
    });
  }
  function scheduleLayout() {
    cancelAnimationFrame(layoutFrame);
    layoutFrame = requestAnimationFrame(function () {
      document.querySelectorAll('.gallery-toggle').forEach(fitPreview);
      equalizeCards();
    });
  }
  function initialize() {
    scheduleLayout();
  }
  if (!window.projectGalleryBound) {
    document.addEventListener('click', function (event) {
      var button = event.target.closest('.gallery-toggle');
      if (!button) return;
      var paragraph = button.closest('.gallery-description');
      var expanded = button.getAttribute('aria-expanded') !== 'true';
      paragraph.querySelector('.gallery-preview').hidden = expanded;
      paragraph.querySelector('.gallery-full').hidden = !expanded;
      button.setAttribute('aria-expanded', String(expanded));
      paragraph.classList.toggle('gallery-collapsed', !expanded);
    });
    document.addEventListener('swup:page:view', initialize);
    document.addEventListener('i18n:applied', initialize);
    window.addEventListener('resize', scheduleLayout);
    if (document.fonts) document.fonts.ready.then(scheduleLayout);
    if (window.ResizeObserver) {
      var observer = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.target.galleryWidth !== entry.contentRect.width) {
            entry.target.galleryWidth = entry.contentRect.width;
            scheduleLayout();
          }
        });
      });
      document.querySelectorAll('.project-gallery').forEach(function (gallery) { observer.observe(gallery); });
    }
    window.projectGalleryBound = true;
  }
  initialize();
})();
