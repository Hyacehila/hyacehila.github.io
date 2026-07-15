/* global hexo */
'use strict';

hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('utils/local-search.ejs', `
<div class="search-pop-overlay" role="dialog" aria-modal="true" aria-labelledby="search-dialog-title">
  <div class="popup search-popup">
    <h2 id="search-dialog-title" class="sr-only"><%= __('search') %></h2>
    <div class="search-header">
      <button type="button" class="search-input-field-pre" aria-label="Clear search">
        <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
      </button>
      <div class="search-input-container">
        <input autocomplete="off" autocorrect="off" autocapitalize="off" placeholder="<%= __('search') %>" spellcheck="false" type="search" class="search-input" role="combobox" aria-autocomplete="list" aria-controls="search-result" aria-expanded="false">
      </div>
      <button type="button" class="popup-btn-close" aria-label="Close search">
        <i class="fa-solid fa-times" aria-hidden="true"></i>
      </button>
    </div>
    <div id="search-status" class="search-status" aria-live="polite"></div>
    <div id="search-result" role="listbox" aria-busy="true">
      <div id="no-result" class="search-state search-state-loading">
        <i class="fa-solid fa-spinner fa-spin-pulse fa-3x fa-fw" aria-hidden="true"></i>
      </div>
    </div>
  </div>
</div>
`);
});
