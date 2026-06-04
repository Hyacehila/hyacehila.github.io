'use strict';

/*====================
  NAVIGATION HANDLING
====================*/
const pageContainers = document.querySelectorAll('[data-page]');
const navLinks = document.querySelectorAll('[data-nav-link]');

// 根据目标页名切换导航高亮与页面显隐
const activatePage = (targetPage) => {
  navLinks.forEach(link => {
    link.classList[link.dataset.target === targetPage ? 'add' : 'remove']('active');
  });

  pageContainers.forEach(container => {
    container.classList[container.dataset.page === targetPage ? 'add' : 'remove']('active');
  });
};

// 绑定导航点击事件（forEach 对空列表天然安全）
navLinks.forEach(link => {
  link.addEventListener('click', () => {
    const target = link.dataset.target;
    if (target) {
      activatePage(target);
    }
  });
});
