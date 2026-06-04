/* 文章页渲染脚本：从 _layouts/blog-post.html 的内联 <script> 抽离而来。
   依赖全局 mermaid（由 CDN 引入，必须在本文件之前加载）与 MathJax。
   行为与原内联脚本保持一致：mermaid 暗色主题渲染 + MathJax 公式着色。 */

/* ===== Mermaid 图表渲染 ===== */
document.addEventListener('DOMContentLoaded', function () {
    // 初始化mermaid配置
    mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        themeVariables: {
            primaryColor: '#3b82f6',
            primaryTextColor: '#f8f8f2',
            primaryBorderColor: '#555',
            lineColor: '#888',
            secondaryColor: '#2d2d2d',
            tertiaryColor: '#1e1e1e',
            background: '#272822',
            mainBkg: '#2d2d2d',
            nodeBorder: '#555',
            clusterBkg: '#1e1e1e',
            clusterBorder: '#555',
            titleColor: '#f8f8f2',
            edgeLabelBackground: '#2d2d2d'
        }
    });

    // 查找所有mermaid代码块并渲染
    const mermaidBlocks = document.querySelectorAll('pre code.language-mermaid, code.language-mermaid');
    mermaidBlocks.forEach(function (block, index) {
        const pre = block.closest('pre') || block.parentElement;
        const code = block.textContent;

        // 创建mermaid容器
        const container = document.createElement('div');
        container.className = 'mermaid';
        container.textContent = code;

        // 替换原始代码块
        pre.parentNode.replaceChild(container, pre);
    });

    // 同时处理rouge高亮后的mermaid块
    const highlightedMermaidBlocks = document.querySelectorAll('.highlight-mermaid, div.language-mermaid');
    highlightedMermaidBlocks.forEach(function (block, index) {
        const code = block.textContent;

        // 创建mermaid容器
        const container = document.createElement('div');
        container.className = 'mermaid';
        container.textContent = code;

        // 替换原始代码块
        block.parentNode.replaceChild(container, block);
    });

    // 运行mermaid渲染
    if (document.querySelectorAll('.mermaid').length > 0) {
        mermaid.run();
    }
});

/* ===== MathJax 渲染后处理（着色 / 字体） ===== */
document.addEventListener('DOMContentLoaded', function () {
    // 等待 MathJax 加载完成
    function waitForMathJax(callback) {
        if (window.MathJax && MathJax.typesetPromise) {
            callback();
        } else {
            setTimeout(function () {
                waitForMathJax(callback);
            }, 100);
        }
    }

    waitForMathJax(function () {
        // 使用 MutationObserver 监听 DOM 变化，确保新渲染的公式也有正确的颜色
        const observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
                if (mutation.addedNodes.length > 0) {
                    applyMathJaxColors();
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // 初始应用颜色
        applyMathJaxColors();

        // MathJax 渲染完成后再次应用
        MathJax.typesetPromise().then(function () {
            applyMathJaxColors();
        });
    });

    function applyMathJaxColors() {
        // 所有 MathJax 元素
        const mathElements = document.querySelectorAll('.mjx-chtml, mjx-container, .MathJax, .mjx-container');

        mathElements.forEach(function (el) {
            // 检查是否为行内公式
            const isInline = el.getAttribute('display') === 'false' ||
                el.classList.contains('mjx-chtml') && el.getAttribute('jax') === 'CHTML' && !el.hasAttribute('display');

            // 检查是否在表格中
            const isInTable = el.closest('table') !== null;

            if (isInline) {
                // 行内公式 - 橙色
                el.style.setProperty('color', '#ffa500', 'important');
                el.style.setProperty('display', 'inline', 'important');
                el.style.setProperty('vertical-align', 'middle', 'important');
            } else {
                // 行间公式 - 浅灰色
                el.style.setProperty('color', '#f8f8f2', 'important');
                el.style.setProperty('font-family', "'Latin-Modern Math', 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif", 'important');
            }

            // 表格中的特殊处理
            if (isInTable && isInline) {
                el.style.setProperty('font-size', '0.9em', 'important');
            }

            // 设置所有子元素的样式
            const allChildren = el.querySelectorAll('*');
            allChildren.forEach(function (child) {
                if (child.classList.contains('mjx-mo')) {
                    // 运算符用橙色
                    child.style.setProperty('color', '#ffa500', 'important');
                } else if (child.classList.contains('mjx-mi') ||
                    child.classList.contains('mjx-mn') ||
                    child.classList.contains('mjx-mtext') ||
                    child.classList.contains('mjx-mspace')) {
                    // 其他元素用浅灰色
                    child.style.setProperty('color', '#f8f8f2', 'important');
                }

                // 为所有数学元素设置字体
                if (child.classList.contains('mjx-mi') ||
                    child.classList.contains('mjx-mo') ||
                    child.classList.contains('mjx-mn') ||
                    child.classList.contains('mjx-mtext') ||
                    child.classList.contains('mjx-mrow')) {
                    child.style.setProperty('font-family', "'Latin-Modern Math', 'STIX Two Math', serif", 'important');
                }
            });
        });
    }
});
