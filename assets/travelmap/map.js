/**
 * Travel Footprint Map
 * Interactive 2D map with China/World toggle
 */

(function () {
    'use strict';

    // Current map mode: 'china' or 'world'
    let currentMapMode = 'china';
    let myChart = null;
    let citiesData = [];
    let chinaGeoJSON = null;
    let worldGeoJSON = null;

    // Get current language from page
    function getCurrentLang() {
        return localStorage.getItem('language') || 'en';
    }

    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', async function () {
        const mapContainer = document.getElementById('travel-map');
        if (!mapContainer) return;

        // Initialize ECharts instance
        myChart = echarts.init(mapContainer);

        // Setup toggle buttons (moved outside try block to ensure they always work)
        setupToggleButtons();

        // Show loading state
        myChart.showLoading({
            text: getCurrentLang() === 'zh' ? '加载中...' : 'Loading...',
            color: '#ffdb70',
            textColor: 'rgba(255, 255, 255, 0.7)',
            maskColor: 'rgba(0, 0, 0, 0.3)'
        });

        try {
            // Load all data in parallel with individual error handling
            const [citiesResponse, chinaResponse, worldResponse] = await Promise.all([
                fetch('/assets/travelmap/cities.json'),
                fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/china.json'),
                fetch('https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/json/world.json')
            ]);

            // Check each response
            if (!citiesResponse.ok) {
                throw new Error(`Cities data failed: ${citiesResponse.status} ${citiesResponse.statusText}`);
            }
            if (!chinaResponse.ok) {
                throw new Error(`China map failed: ${chinaResponse.status} ${chinaResponse.statusText}`);
            }
            if (!worldResponse.ok) {
                throw new Error(`World map failed: ${worldResponse.status} ${worldResponse.statusText}`);
            }

            const [cities, china, world] = await Promise.all([
                citiesResponse.json(),
                chinaResponse.json(),
                worldResponse.json()
            ]);

            citiesData = cities;
            chinaGeoJSON = china;
            worldGeoJSON = world;

            // Register maps
            echarts.registerMap('china', chinaGeoJSON);
            echarts.registerMap('world', worldGeoJSON);

            // Render initial map
            renderMap(currentMapMode);

            myChart.hideLoading();
        } catch (error) {
            console.error('Failed to load map data:', error);
            myChart.hideLoading();
            mapContainer.innerHTML = '<div style="color: rgba(255,255,255,0.6); text-align: center; padding: 40px;">Failed to load map / 加载地图失败<br><small style="color: rgba(255,255,255,0.4);">' + error.message + '</small></div>';
        }

        // Handle window resize
        window.addEventListener('resize', function () {
            if (myChart) {
                myChart.resize();
            }
        });
    });

    // Setup toggle buttons
    function setupToggleButtons() {
        const toggleBtns = document.querySelectorAll('.map-toggle-btn');
        toggleBtns.forEach(btn => {
            btn.addEventListener('click', function () {
                const mode = this.getAttribute('data-map-mode');
                if (mode !== currentMapMode) {
                    currentMapMode = mode;
                    // Update active button state
                    toggleBtns.forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    // Re-render map
                    renderMap(mode);
                }
            });
        });
    }

    // Render map with specified mode
    function renderMap(mode) {
        const lang = getCurrentLang();
        const mapName = mode === 'china' ? 'china' : 'world';

        // Process cities data for current language
        const processedData = citiesData.map(city => ({
            name: lang === 'zh' ? city.name : city.nameEn,
            value: city.value,
            visits: city.visits,
            type: lang === 'zh' ? city.type : city.typeEn,
            details: lang === 'zh' ? city.details : city.detailsEn,
            // Keep original data for reference
            originalName: city.name,
            originalNameEn: city.nameEn
        }));

        const option = {
            backgroundColor: 'transparent',

            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(30, 30, 30, 0.9)',
                borderColor: 'rgba(255, 219, 112, 0.3)',
                borderWidth: 1,
                textStyle: {
                    color: '#fff',
                    fontSize: 13
                },
                formatter: function (params) {
                    if (params.componentType === 'series' && params.data) {
                        const city = params.data;
                        const visitLabel = lang === 'zh' ? '到访次数' : 'Visits';
                        const typeLabel = lang === 'zh' ? '类型' : 'Type';
                        const detailsLabel = lang === 'zh' ? '备注' : 'Notes';

                        return `
              <div style="padding: 4px 8px;">
                <strong style="color: #ffdb70; font-size: 14px;">📍 ${city.name}</strong><br/>
                <span style="color: rgba(255,255,255,0.7);">${typeLabel}: ${city.type || '-'}</span><br/>
                <span style="color: rgba(255,255,255,0.7);">${visitLabel}: ${city.visits || 1}</span><br/>
                <span style="color: rgba(255,255,255,0.6); font-size: 12px;">${city.details || ''}</span>
              </div>
            `;
                    }
                    return '';
                }
            },

            geo: {
                map: mapName,
                roam: false, // Disable zoom and pan as per user request
                // Use layoutCenter and layoutSize for proper centering and sizing
                layoutCenter: ['50%', '50%'],
                layoutSize: mode === 'china' ? '100%' : '95%',
                // Aspect ratio adjustment
                aspectScale: mode === 'china' ? 0.85 : 0.75,
                label: {
                    show: false
                },
                itemStyle: {
                    areaColor: 'rgba(40, 40, 45, 0.8)',
                    borderColor: 'rgba(100, 100, 110, 0.6)',
                    borderWidth: 0.5
                },
                emphasis: {
                    disabled: true // Disable region hover effect
                },
                silent: true // Make geo component non-interactive
            },

            series: [{
                name: lang === 'zh' ? '足迹' : 'Footprints',
                type: 'scatter',
                coordinateSystem: 'geo',
                symbol: 'pin',
                symbolSize: 30,
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{b}',
                    fontSize: 11,
                    color: 'rgba(255, 255, 255, 0.9)',
                    textBorderColor: 'rgba(0, 0, 0, 0.5)',
                    textBorderWidth: 2
                },
                itemStyle: {
                    color: '#ffdb70', // Vegas gold - matching site theme
                    shadowBlur: 8,
                    shadowColor: 'rgba(255, 219, 112, 0.5)'
                },
                emphasis: {
                    scale: 1.2,
                    itemStyle: {
                        color: '#ffd700'
                    }
                },
                data: processedData
            }]
        };

        myChart.setOption(option, true); // true = notMerge, clear previous options
    }

    // Expose function for language change callback
    window.updateTravelMapLanguage = function () {
        if (myChart && citiesData.length > 0) {
            renderMap(currentMapMode);
        }
    };

})();
