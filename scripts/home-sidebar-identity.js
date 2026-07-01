/* global hexo */
'use strict';

hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('pages/home/home-sidebar.ejs', `
<div class="home-sidebar-container home-identity-sidebar">
    <div class="sticky-container sticky">
        <%
            const hasSidebarLinks = theme.home.sidebar.links !== null;
            const hasSidebarAnnouncement = theme.home.sidebar.announcement !== null;
        %>
        <% if (hasSidebarLinks || hasSidebarAnnouncement) {%>
            <div class="sidebar-links">
                <div class="site-info">
                    <div class="site-name"><%= theme.info.title || config.title %></div>
                    <% if (hasSidebarAnnouncement && hasSidebarLinks) { %>
                        <div class="announcement">
                            <%- theme.home.sidebar.announcement %>
                        </div>
                    <% } %>
                </div>
                <% if (hasSidebarLinks) {%>
                    <% for (let i in theme.home.sidebar.links) { %>
                        <% if (theme.home.sidebar.links[i].path === 'none') {} else {%>
                            <a class="links" href="<%= url_for(theme.home.sidebar.links[i].path) %>">
                                <% if (theme.home.sidebar.links[i].icon) { %>
                                    <i class="<%- theme.home.sidebar.links[i].icon %> icon-space"></i>
                                <% } %>
                                <span class="link-name"><%= __(i) %></span>
                            </a>
                        <% } %>
                    <% } %>
                <% } else {%>
                    <div class="announcement-outside">
                        <%- theme.home.sidebar.announcement %>
                    </div>
                <% } %>
            </div>
        <% } %>

        <div class="sidebar-content home-identity-card">
            <%- partial("components/sidebar/avatar") %>
            <%- partial("components/sidebar/author") %>

            <div class="home-identity-focus" aria-label="Research interests">
                <div class="home-identity-focus-list">
                    <div class="home-identity-focus-item">
                        <i class="fa-solid fa-chart-pie" aria-hidden="true"></i>
                        <span data-i18n="service-data-science-title">Data Science</span>
                    </div>
                    <div class="home-identity-focus-item">
                        <i class="fa-solid fa-robot" aria-hidden="true"></i>
                        <span data-i18n="service-llm-agents-title">LLM Agents</span>
                    </div>
                    <div class="home-identity-focus-item">
                        <i class="fa-solid fa-clipboard-check" aria-hidden="true"></i>
                        <span data-i18n="service-llm-eval-title">Evaluation/Benchmark</span>
                    </div>
                    <div class="home-identity-focus-item">
                        <i class="fa-solid fa-mobile-screen-button" aria-hidden="true"></i>
                        <span data-i18n="service-ios-dev-title">iOS Dev</span>
                    </div>
                    <div class="home-identity-focus-item">
                        <i class="fa-solid fa-gamepad" aria-hidden="true"></i>
                        <span data-i18n="service-game-client-title">Game Client Dev</span>
                    </div>
                </div>
            </div>

            <div class="home-identity-statistics" aria-label="Site statistics">
                <div class="statistics static-sidebar-statistics flex justify-around my-2.5">
                    <div class="item tag-count-item flex flex-col justify-center items-center w-20">
                        <div class="number text-2xl sm:text-xl text-second-text-color font-semibold"><%= site.tags.length %></div>
                        <div class="label text-third-text-color text-sm"><%- __('tags') %></div>
                    </div>
                    <div class="item tag-count-item flex flex-col justify-center items-center w-20">
                        <div class="number text-2xl sm:text-xl text-second-text-color font-semibold"><%= site.categories.length %></div>
                        <div class="label text-third-text-color text-sm"><%- __('categories') %></div>
                    </div>
                    <div class="item tag-count-item flex flex-col justify-center items-center w-20">
                        <div class="number text-2xl sm:text-xl text-second-text-color font-semibold"><%= site.posts.length %></div>
                        <div class="label text-third-text-color text-sm"><%- __('posts') %></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
`);
});
