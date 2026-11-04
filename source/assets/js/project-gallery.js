/* Progressive enhancement: full descriptions remain readable without JS. */
(function () {
  'use strict';
  var PREVIEW_LENGTH = 92;
  var layoutFrame;
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
        }
        body.parentElement.appendChild(measure);
        tallest = Math.max(tallest, measure.getBoundingClientRect().height);
        measure.remove();
      });
      gallery.style.setProperty('--gallery-body-height', Math.ceil(tallest) + 'px');
    });
  }
  function scheduleLayout() {
    cancelAnimationFrame(layoutFrame);
    layoutFrame = requestAnimationFrame(equalizeCards);
  }
  function initialize() {
    document.querySelectorAll('.gallery-toggle').forEach(function (button) {
      var paragraph = button.closest('.gallery-description');
      var full = paragraph.querySelector('.gallery-full');
      var preview = paragraph.querySelector('.gallery-preview');
      var characters = Array.from(full.textContent.trim());
      if (characters.length <= PREVIEW_LENGTH) {
        full.hidden = false;
        preview.hidden = true;
        button.hidden = true;
        button.setAttribute('aria-expanded', 'true');
        return;
      }
      var end = PREVIEW_LENGTH;
      // If the limit lands on a sentence boundary, leave that sentence unfinished.
      if (/[。！？.!?]/u.test(characters[end - 1]) || /[。！？.!?]/u.test(characters[end])) end -= 6;
      preview.textContent = characters.slice(0, end).join('').replace(/[，。；！？、,.!?;:\s]+$/u, '') + '…';
      var expanded = !button.hidden && button.getAttribute('aria-expanded') === 'true';
      preview.hidden = expanded;
      full.hidden = !expanded;
      button.setAttribute('aria-expanded', String(expanded));
      button.hidden = false;
    });
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
