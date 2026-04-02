'use strict';

(function () {
  const groups = Array.from(document.querySelectorAll('[data-busuanzi-group]'));
  if (groups.length === 0) {
    return;
  }

  const metrics = groups.flatMap(group => Array.from(group.querySelectorAll('[data-busuanzi-metric]')));
  const configGroup = groups.find(group => group.dataset.counterApi);
  const apiUrl = configGroup ? configGroup.dataset.counterApi : '';
  const timeoutMs = Number(configGroup && configGroup.dataset.counterTimeout) || 4500;

  const getMetricKey = (metric) => metric.dataset.counterKey || '';

  const isValidMetricValue = (value) => {
    if (value === null || value === undefined) {
      return false;
    }

    const text = String(value).trim();
    if (!text) {
      return false;
    }

    const normalized = text.toLowerCase();
    return text !== '--' && text !== '加载中...' && normalized !== 'loading...' && normalized !== 'nan';
  };

  const setValue = (metric, value) => {
    const valueNode = metric.querySelector('[data-busuanzi-value]');
    if (!valueNode) {
      return;
    }

    valueNode.textContent = String(value);
  };

  const setMetricReady = (metric) => {
    metric.classList.remove('is-hidden');
    metric.classList.add('is-ready');
    metric.setAttribute('aria-hidden', 'false');
  };

  const setMetricHidden = (metric) => {
    metric.classList.remove('is-ready');
    metric.classList.add('is-hidden');
    metric.setAttribute('aria-hidden', 'true');
  };

  const setGroupVisible = (group) => {
    group.classList.remove('is-hidden');
    group.classList.add('is-visible');
    group.setAttribute('aria-hidden', 'false');
  };

  const setGroupHidden = (group) => {
    group.classList.remove('is-visible');
    group.classList.add('is-hidden');
    group.setAttribute('aria-hidden', 'true');
  };

  const renderMetrics = (payload) => {
    groups.forEach(group => {
      let groupHasReadyMetric = false;

      group.querySelectorAll('[data-busuanzi-metric]').forEach(metric => {
        const key = getMetricKey(metric);
        const value = key ? payload[key] : undefined;

        if (isValidMetricValue(value)) {
          setValue(metric, value);
          setMetricReady(metric);
          groupHasReadyMetric = true;
        } else {
          setMetricHidden(metric);
        }
      });

      if (groupHasReadyMetric) {
        setGroupVisible(group);
      } else {
        setGroupHidden(group);
      }
    });
  };

  const hideAll = () => {
    renderMetrics({});
  };

  if (!apiUrl || metrics.length === 0) {
    hideAll();
    return;
  }

  if (typeof window.fetch !== 'function') {
    hideAll();
    return;
  }

  const controller = typeof AbortController === 'function' ? new AbortController() : null;
  const timeoutId = window.setTimeout(() => {
    if (controller) {
      controller.abort();
    }
  }, timeoutMs);

  const requestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      url: window.location.origin + window.location.pathname + window.location.search,
      referrer: document.referrer
    })
  };

  if (controller) {
    requestInit.signal = controller.signal;
  }

  window.fetch(apiUrl, requestInit)
    .then(response => {
      if (!response.ok) {
        throw new Error('counter_request_failed');
      }

      return response.json();
    })
    .then(payload => {
      if (!payload || typeof payload !== 'object') {
        hideAll();
        return;
      }

      renderMetrics(payload);
    })
    .catch(() => {
      hideAll();
    })
    .finally(() => {
      window.clearTimeout(timeoutId);
    });
})();
