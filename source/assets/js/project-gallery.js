/* Progressive enhancement: full descriptions remain readable without JS. */
(function () {
  'use strict';
  var PREVIEW_LENGTH = 92;
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
    window.projectGalleryBound = true;
  }
  initialize();
})();
