/* Progressive enhancement: full descriptions remain readable without JS. */
(function () {
  'use strict';
  function initialize() {
    document.querySelectorAll('.gallery-toggle[hidden]').forEach(function (button) {
      var paragraph = button.closest('.gallery-description');
      paragraph.querySelector('.gallery-preview').hidden = false;
      paragraph.querySelector('.gallery-full').hidden = true;
      button.setAttribute('aria-expanded', 'false');
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
    window.projectGalleryBound = true;
  }
  initialize();
})();
