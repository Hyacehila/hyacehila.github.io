/* global hexo */
'use strict';

hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('components/comments/utterances.ejs', `
<% if(theme.comment.system === 'utterances' && theme.comment.config.utterances && theme.comment.config.utterances.repo) { %>
    <% const utterancesConfig = theme.comment.config.utterances || {}; %>
    <% const issueNumber = utterancesConfig.issue_number; %>
    <% const issueTerm = utterancesConfig.issue_term || 'pathname'; %>
    <% const themeLight = utterancesConfig.theme_light || 'github-light'; %>
    <% const themeDark = utterancesConfig.theme_dark || 'github-dark'; %>
    <% const safeJson = value => JSON.stringify(value || '').replace(/</g, '\\\\u003c'); %>
    <div id="utterances-container"
         data-utterances-site-url="<%= config.url || '' %>"
         data-utterances-theme-light="<%= themeLight %>"
         data-utterances-theme-dark="<%= themeDark %>">
    </div>
    <script <%- theme.global.single_page === true ? 'data-swup-reload-script' : '' %>>
        (function() {
            const currentScript = document.currentScript;
            const container = currentScript &&
                currentScript.previousElementSibling &&
                currentScript.previousElementSibling.id === "utterances-container"
                ? currentScript.previousElementSibling
                : document.getElementById("utterances-container");

            if (!container) {
                return;
            }

            const repo = <%- safeJson(utterancesConfig.repo) %>;
            const issueNumber = <%- safeJson(issueNumber) %>;
            const issueTerm = <%- safeJson(issueTerm) %>;
            const label = <%- safeJson(utterancesConfig.label) %>;

            if (!repo || (!issueNumber && !issueTerm)) {
                return;
            }

            const themeLight = container.dataset.utterancesThemeLight || "github-light";
            const themeDark = container.dataset.utterancesThemeDark || "github-dark";
            const isDark = document.body.classList.contains("dark-mode");
            const theme = isDark ? themeDark : themeLight;

            const canonicalUrl = (function() {
                const siteUrl = container.dataset.utterancesSiteUrl || window.location.origin;
                const base = new URL(siteUrl, window.location.href);
                const canonical = new URL(window.location.pathname, base);
                // Let utterances read the OAuth query from location; only normalize canonical.
                canonical.search = "";
                canonical.hash = "";
                return canonical.href;
            })();

            let canonicalLink = document.querySelector('link[rel="canonical"]');
            if (!canonicalLink) {
                canonicalLink = document.createElement("link");
                canonicalLink.rel = "canonical";
                document.head.appendChild(canonicalLink);
            }
            canonicalLink.href = canonicalUrl;

            const key = [
                repo,
                issueNumber ? "number:" + issueNumber : "term:" + issueTerm,
                label,
                canonicalUrl,
                theme
            ].join("|");

            if (container.dataset.utterancesKey === key) {
                const existingFrame = container.querySelector("iframe.utterances-frame");
                const pendingScript = container.querySelector('script[src^="https://utteranc.es/client.js"]');
                if (existingFrame || pendingScript) {
                    return;
                }
            }

            container.dataset.utterancesKey = key;
            container.innerHTML = "";

            const script = document.createElement("script");
            script.src = "https://utteranc.es/client.js";
            script.async = true;
            script.crossOrigin = "anonymous";
            script.setAttribute("repo", repo);

            if (issueNumber) {
                script.setAttribute("issue-number", issueNumber);
            } else {
                script.setAttribute("issue-term", issueTerm);
            }

            if (label) {
                script.setAttribute("label", label);
            }

            script.setAttribute("theme", theme);
            container.appendChild(script);
        })();
    </script>
<% } %>
`);
});
