document.addEventListener('DOMContentLoaded', () => {
    
    // --- Scroll Timeline Line Logic ---
    const scrollLine = document.getElementById('scroll-line');
    
    window.addEventListener('scroll', () => {
        // Calculate scroll percentage
        const scrollTop = window.scrollY;
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPercent = scrollTop / docHeight;
        
        // Update SVG line dashoffset
        // The stroke-dasharray is 1000, so setting offset from 1000 down to 0
        if (scrollLine) {
            const offset = 1000 - (scrollPercent * 1000);
            scrollLine.style.strokeDashoffset = offset;
        }

        // Sticky Navbar
        const navbar = document.getElementById('navbar');
        if (scrollTop > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // --- Custom Masking Reveal Animation ---
    const revealElements = document.querySelectorAll('.reveal-mask');
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
    };

    const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                // Unobserve after revealing for performance and static feel
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    revealElements.forEach(el => {
        revealObserver.observe(el);
    });

    // --- Presentation (Viva) Mode Toggle ---
    const vivaBtn = document.getElementById('viva-mode-btn');
    let isVivaMode = false;
    
    const vivaControls = document.createElement('div');
    vivaControls.className = 'viva-controls hidden';
    vivaControls.innerHTML = `<button id="exit-viva" class="btn btn-outline" style="background: #fff;"><i class="fa-solid fa-xmark"></i> Exit Presentation</button>`;
    document.body.appendChild(vivaControls);

    const toggleVivaMode = () => {
        isVivaMode = !isVivaMode;
        if (isVivaMode) {
            document.body.classList.add('viva-mode');
            vivaControls.classList.remove('hidden');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            document.body.classList.remove('viva-mode');
            vivaControls.classList.add('hidden');
        }
    };

    vivaBtn.addEventListener('click', (e) => {
        e.preventDefault();
        toggleVivaMode();
    });
    
    document.getElementById('exit-viva').addEventListener('click', toggleVivaMode);

    // --- State & District Autocomplete Data ---
    let statesDistrictsMap = {};
    const stateInput = document.getElementById('m3-state');
    const districtInput = document.getElementById('m3-district');
    const statesDatalist = document.getElementById('m3-states-list');
    const districtsDatalist = document.getElementById('m3-districts-list');

    fetch("http://127.0.0.1:8000/api/v3/states_districts")
        .then(res => res.json())
        .then(data => {
            statesDistrictsMap = data;
            if (statesDatalist) {
                statesDatalist.innerHTML = Object.keys(data)
                    .map(st => `<option value="${st}">`)
                    .join('');
            }
            updateDistrictsDatalist();
        })
        .catch(err => console.log("Note: States/Districts list will load dynamically."));

    function updateDistrictsDatalist() {
        if (!districtInput || !districtsDatalist) return;
        const selectedState = (stateInput ? stateInput.value : "").trim().toUpperCase();
        const dists = statesDistrictsMap[selectedState] || [];
        districtsDatalist.innerHTML = dists.map(d => `<option value="${d}">`).join('');
    }

    if (stateInput) {
        stateInput.addEventListener('input', updateDistrictsDatalist);
        stateInput.addEventListener('change', updateDistrictsDatalist);
    }

    // --- Mode Toggle Logic (Model 3 District Advisor vs 7-Param Scientific) ---
    const modeModel3Btn = document.getElementById('mode-model3');
    const modeNpkBtn = document.getElementById('mode-npk');
    const formModel3 = document.getElementById('model3-form');
    const formNpk = document.getElementById('prediction-form');
    const terminalOutput = document.getElementById('terminal-output');

    if (modeModel3Btn && modeNpkBtn) {
        modeModel3Btn.addEventListener('click', () => {
            modeModel3Btn.classList.add('active');
            modeNpkBtn.classList.remove('active');
            if (formModel3) formModel3.style.display = 'block';
            if (formNpk) formNpk.style.display = 'none';
            if (terminalOutput) terminalOutput.classList.add('hidden');
        });

        modeNpkBtn.addEventListener('click', () => {
            modeNpkBtn.classList.add('active');
            modeModel3Btn.classList.remove('active');
            if (formNpk) formNpk.style.display = 'block';
            if (formModel3) formModel3.style.display = 'none';
            if (terminalOutput) terminalOutput.classList.add('hidden');
        });
    }

    // --- Form Submissions ---
    if (formModel3) {
        formModel3.addEventListener('submit', (e) => {
            e.preventDefault();
            executeModel3Inference();
        });
    }

    if (formNpk) {
        formNpk.addEventListener('submit', (e) => {
            e.preventDefault();
            executeInference();
        });
    }
});

function executeInference() {
    // UI Elements
    const terminalOutput = document.getElementById('terminal-output');
    const submitBtn = document.getElementById('submit-btn');

    // Get 7 Parameters
    const data = {
        nitrogen: Number(document.getElementById("nitrogen").value),
        phosphorous: Number(document.getElementById("phosphorous").value),
        potassium: Number(document.getElementById("potassium").value),
        temperature: Number(document.getElementById("temperature").value),
        humidity: Number(document.getElementById("humidity").value),
        ph: Number(document.getElementById("ph").value),
        rainfall: Number(document.getElementById("rainfall").value)
    };

    // UI Loading State
    terminalOutput.classList.remove('hidden');
    terminalOutput.innerHTML = `> EXECUTING NPK INFERENCE PIPELINE...<br>> TRANSMITTING DATA ARRAY... <span class="blink">_</span>`;
    if(submitBtn) submitBtn.disabled = true;

    fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify(data),
    })
    .then((response) => {
        if (!response.ok) return response.json().then(err => { throw new Error(err.detail || "Server exception") });
        return response.json();
    })
    .then((responseData) => {
        const predictedCrops = responseData["result"];
        const cropsHtml = predictedCrops.map((c, i) => `#${i+1} ${c.crop.toUpperCase()} ${c.confidence}%`).join('<br>');
        setTimeout(() => {
            terminalOutput.innerHTML = `
<div style="color: #8b949e; margin-bottom: 5px;">7-PARAMETER INPUT</div>
N: ${data.nitrogen}<br>
P: ${data.phosphorous}<br>
K: ${data.potassium}<br>
Temperature: ${data.temperature}°C<br>
Humidity: ${data.humidity}%<br>
pH: ${data.ph}<br>
Rainfall: ${data.rainfall} mm<br><br>

<div style="color: #8b949e; margin-bottom: 5px;">RANDOM FOREST OUTPUT</div>
<strong style="color: #4ade80; font-size: 1.2rem;">${cropsHtml}</strong><br><br>
<div id="ai-explanation" style="color: #ffbd2e; font-style: italic;">> [AI ASSISTANT GENERATING EXPLANATION...]</div>
<div id="xai-plots" style="color: #ffbd2e; font-style: italic;">> [GENERATING EXPLAINABLE AI PLOTS...]</div>
            `;
            
            // Fetch AI Explanation
            fetch("http://127.0.0.1:8000/explain/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: predictedCrops[0].crop,
                    nitrogen: data.nitrogen, phosphorous: data.phosphorous, potassium: data.potassium,
                    temperature: data.temperature, humidity: data.humidity, ph: data.ph, rainfall: data.rainfall
                })
            })
            .then(res => res.json())
            .then(explainData => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) {
                    aiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">GEMINI AI ASSISTANT</div><span style="color: #58a6ff;">${explainData.explanation}</span>`;
                }
            })
            .catch(err => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) aiDiv.innerHTML = `<span style="color: #f85149;">> [AI EXPLANATION FAILED]</span>`;
            });
            
            // Fetch XAI Plots
            fetch("http://127.0.0.1:8000/explain_plots/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: predictedCrops[0].crop,
                    nitrogen: data.nitrogen, phosphorous: data.phosphorous, potassium: data.potassium,
                    temperature: data.temperature, humidity: data.humidity, ph: data.ph, rainfall: data.rainfall
                })
            })
            .then(res => res.json())
            .then(plotData => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv && plotData.shap_base64) {
                    xaiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">EXPLAINABLE AI (SHAP)</div>
<img src="data:image/png;base64,${plotData.shap_base64}" style="max-width: 100%; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px;">`;
                } else if(xaiDiv) {
                    xaiDiv.innerHTML = "";
                }
            })
            .catch(err => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv) xaiDiv.innerHTML = `<span style="color: #f85149;">> [XAI PLOTS FAILED]</span>`;
            });
            
        }, 800);
    })
    .catch((error) => {
        terminalOutput.innerHTML = `> EXCEPTION RAISED.<br>> ERR_DESC: ${error.message}`;
    })
    .finally(() => {
        setTimeout(() => { if(submitBtn) submitBtn.disabled = false; }, 800);
    });
}

function executeModel3Inference() {
    const terminalOutput = document.getElementById('terminal-output');
    const submitBtn = document.getElementById('submit-m3-btn');

    const stateVal = (document.getElementById("m3-state")?.value || "").trim();
    const districtVal = (document.getElementById("m3-district")?.value || "").trim();
    const seasonVal = (document.getElementById("m3-season")?.value || "Kharif").trim();

    const nVal = parseFloat(document.getElementById("m3-n")?.value) || null;
    const pVal = parseFloat(document.getElementById("m3-p")?.value) || null;
    const kVal = parseFloat(document.getElementById("m3-k")?.value) || null;
    const phVal = parseFloat(document.getElementById("m3-ph")?.value) || null;

    if (!stateVal || !districtVal) {
        alert("Please enter both State and District.");
        return;
    }

    terminalOutput.classList.remove('hidden');
    terminalOutput.innerHTML = `
        <div style="color: #58a6ff;">> INITIALIZING AGRISENSE MODEL 3 PIPELINE...</div>
        <div style="color: #8b949e;">> FETCHING LIVE WEATHER & 7-DAY PROSPECTIVE FORECAST VIA OPEN-METEO API...</div>
        <div style="color: #8b949e;">> INGESTING 50-YEAR IMD CLIMATE NORMALS & ICAR SOIL HEALTH DATABASE...</div>
        <div style="color: #e3b341;">> EVALUATING PHYSIOLOGICAL BOUNDS & MULTI-LABEL DECISION GATE... <span class="blink">_</span></div>
    `;

    if (submitBtn) submitBtn.disabled = true;

    fetch("http://127.0.0.1:8000/api/v3/recommend_district_crops", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            state: stateVal,
            district: districtVal,
            season: seasonVal,
            nitrogen: nVal,
            phosphorous: pVal,
            potassium: kVal,
            ph: phVal
        })
    })
    .then(res => {
        if (!res.ok) return res.json().then(err => { throw new Error(err.detail || "Server error") });
        return res.json();
    })
    .then(data => {
        const weather = data.weather_report || {};
        const curr = weather.current || {};
        const forecast = weather.upcoming_forecast || {};
        const normals = weather.historical_normals || {};
        const soil = data.soil_profile || {};
        const status = data.status;

        let weatherHtml = `
            <div style="background: rgba(13, 17, 23, 0.7); border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin: 12px 0;">
                <div style="color: #58a6ff; font-weight: 600; margin-bottom: 8px; font-size: 0.9rem;">
                    🌤️ MULTI-HORIZON WEATHER INTELLIGENCE (${data.district}, ${data.state})
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 0.82rem;">
                    <div>
                        <span style="color: #8b949e;">Live Ambient Temp:</span> <strong style="color: #fff;">${curr.temperature ?? 'N/A'}°C</strong> (${curr.weather_condition || 'Normal'})<br>
                        <span style="color: #8b949e;">Relative Humidity:</span> <strong style="color: #fff;">${curr.relative_humidity ?? 'N/A'}%</strong><br>
                        <span style="color: #8b949e;">Current Wind:</span> <strong style="color: #fff;">${curr.wind_speed ?? 'N/A'} km/h</strong>
                    </div>
                    <div>
                        <span style="color: #8b949e;">Upcoming 7-Day Rain:</span> <strong style="color: #4ade80;">${forecast.expected_rainfall_sum_mm ?? 0} mm</strong><br>
                        <span style="color: #8b949e;">Peak Rain Probability:</span> <strong style="color: #fff;">${forecast.max_rain_probability ?? 0}%</strong><br>
                        <span style="color: #8b949e;">Diurnal Range (High/Low):</span> <strong style="color: #fff;">${forecast.mean_max_temp ?? 30}°C / ${forecast.mean_min_temp ?? 20}°C</strong>
                    </div>
                    <div>
                        <span style="color: #8b949e;">50-Yr Seasonal Normal:</span> <strong style="color: #fff;">${normals.seasonal_rainfall_normal_mm ?? 'N/A'} mm</strong><br>
                        <span style="color: #8b949e;">Annual Normal Rain:</span> <strong style="color: #fff;">${normals.annual_rainfall_normal_mm ?? 'N/A'} mm</strong><br>
                        <span style="color: #8b949e;">Precipitation Regime:</span> <span style="color: #ffbd2e; font-weight: 600;">${normals.rainfall_status || 'Normal'}</span>
                    </div>
                </div>
            </div>

            <div style="background: rgba(13, 17, 23, 0.7); border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; margin-bottom: 15px; font-size: 0.82rem;">
                <span style="color: #8b949e;">🌱 SOIL TELEMETRY (${soil.source}):</span> &nbsp;
                <strong>N:</strong> ${soil.nitrogen} kg/ha | 
                <strong>P:</strong> ${soil.phosphorous} kg/ha | 
                <strong>K:</strong> ${soil.potassium} kg/ha | 
                <strong>pH:</strong> ${soil.ph}
            </div>
        `;

        if (status === "NO_CROPS_POSSIBLE") {
            terminalOutput.innerHTML = `
                ${weatherHtml}
                <div style="background: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; border-radius: 6px; padding: 15px; color: #f85149; font-weight: 600;">
                    <div style="font-size: 1.1rem; margin-bottom: 6px;">⚠️ ZERO-CROP INFEASIBILITY DETECTED: NO CROPS POSSIBLE</div>
                    <div style="font-size: 0.88rem; color: #ff7b72; font-weight: 400; margin-bottom: 8px;">
                        ${data.reason || "Cultivation is unviable under the specified environmental and seasonal constraints."}
                    </div>
                    <div style="font-size: 0.8rem; color: #8b949e; font-weight: 400;">
                        Recommendation: Review season selection (e.g. switch to active Kharif/Rabi cycle) or await favorable climatic temperature window.
                    </div>
                </div>
            `;
            return;
        }

        // CROPS AVAILABLE
        const crops = data.crops || [];
        let cropsCardsHtml = crops.map((c, i) => {
            const catBadgeColor = c.category === 'CEREAL' ? '#3b82f6' : (c.category === 'PULSE' ? '#10b981' : (c.category === 'OILSEED' ? '#f59e0b' : '#8b5cf6'));
            const waterColor = c.water_requirement === 'HIGH' ? '#38bdf8' : (c.water_requirement === 'LOW' ? '#fbbf24' : '#a3e635');
            const validBadge = c.historically_validated ? `<span style="background: rgba(74, 222, 128, 0.2); color: #4ade80; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem;">✓ HISTORICAL</span>` : '';

            return `
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #fff; font-size: 1rem;">#${i+1} ${c.crop}</strong>
                        <span style="color: #4ade80; font-weight: bold; font-size: 0.95rem;">${c.confidence}%</span>
                    </div>
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px;">
                        <span style="background: rgba(255,255,255,0.08); color: ${catBadgeColor}; padding: 1px 6px; border-radius: 3px; font-size: 0.72rem; font-weight: 600;">${c.category}</span>
                        <span style="background: rgba(255,255,255,0.08); color: ${waterColor}; padding: 1px 6px; border-radius: 3px; font-size: 0.72rem;">WATER: ${c.water_requirement}</span>
                        ${validBadge}
                    </div>
                    <div style="background: #21262d; border-radius: 3px; height: 4px; width: 100%; margin-top: 6px; overflow: hidden;">
                        <div style="background: #4ade80; height: 100%; width: ${c.confidence}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        terminalOutput.innerHTML = `
            ${weatherHtml}
            <div style="color: #4ade80; font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <span>🌾 ALL VIABLE CROPS FOR ${data.season.toUpperCase()} (${data.total_crops_possible} Varieties Found)</span>
                <span style="font-size: 0.8rem; color: #8b949e; font-weight: 400;">Ranked by Suitability Probability</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; max-height: 400px; overflow-y: auto; padding-right: 5px;">
                ${cropsCardsHtml}
            </div>
        `;
    })
    .catch(err => {
        terminalOutput.innerHTML = `<span style="color: #f85149;">> EXCEPTION RAISED: ${err.message}</span>`;
    })
    .finally(() => {
        if (submitBtn) submitBtn.disabled = false;
    });
}

function executeRegionalInference() {
    const terminalOutput = document.getElementById('terminal-output');
    const submitBtn = document.getElementById('submit-regional-btn');

    const data = {
        state: document.getElementById("state").value,
        district: document.getElementById("district").value
    };

    terminalOutput.classList.remove('hidden');
    terminalOutput.innerHTML = `> EXECUTING REGIONAL HYBRID PIPELINE...<br>> CALLING OPEN-METEO API FOR LIVE WEATHER... <span class="blink">_</span>`;
    if(submitBtn) submitBtn.disabled = true;

    fetch("http://127.0.0.1:8000/predict_hybrid/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify(data),
    })
    .then((response) => {
        if (!response.ok) return response.json().then(err => { throw new Error(err.detail || "Server exception") });
        return response.json();
    })
    .then((responseData) => {
        const crops = responseData["crop"];
        const validated = responseData["validated"];
        const temp = responseData["live_temp"];
        const hum = responseData["live_humidity"];
        const soil = responseData["soil_data"];
        
        setTimeout(() => {
            let validationMsg = validated ? 
                `<span style="color: #4ade80;">> [REGION VALIDATED - GROWN HISTORICALLY]</span>` : 
                `<span style="color: #ffbd2e;">> [MODEL PREDICTION - MAY NOT BE LOCALLY TRADITIONAL]</span>`;

            const cropsHtml = crops.map((c, i) => `#${i+1} ${c.crop.toUpperCase()} ${c.confidence}%`).join('<br>');

            terminalOutput.innerHTML = `
<div style="color: #8b949e; margin-bottom: 5px;">LOCATION</div>
<strong style="color: #fff;">${data.district.toUpperCase()}, ${data.state.toUpperCase()}</strong><br><br>

<div style="color: #8b949e; margin-bottom: 5px;">7-PARAMETER INPUT</div>
N: ${soil.N}<br>
P: ${soil.P}<br>
K: ${soil.K}<br>
Temperature: ${temp}°C<br>
Humidity: ${hum}%<br>
pH: ${soil.pH}<br>
Rainfall: ${soil.Rainfall} mm<br><br>

<div style="color: #8b949e; margin-bottom: 5px;">RANDOM FOREST OUTPUT</div>
<strong style="color: #4ade80; font-size: 1.2rem;">${cropsHtml}</strong><br><br>
${validationMsg}<br><br>
<div id="ai-explanation" style="color: #ffbd2e; font-style: italic;">> [AI ASSISTANT GENERATING EXPLANATION...]</div>
<div id="xai-plots" style="color: #ffbd2e; font-style: italic;">> [GENERATING EXPLAINABLE AI PLOTS...]</div>
            `;
            
            // Fetch AI Explanation
            fetch("http://127.0.0.1:8000/explain/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: crops[0].crop,
                    nitrogen: soil.N, phosphorous: soil.P, potassium: soil.K,
                    temperature: temp, humidity: hum, ph: soil.pH, rainfall: soil.Rainfall
                })
            })
            .then(res => res.json())
            .then(explainData => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) {
                    aiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">GEMINI AI ASSISTANT</div><span style="color: #58a6ff;">${explainData.explanation}</span>`;
                }
            })
            .catch(err => {
                const aiDiv = document.getElementById("ai-explanation");
                if(aiDiv) aiDiv.innerHTML = `<span style="color: #f85149;">> [AI EXPLANATION FAILED]</span>`;
            });
            
            // Fetch XAI Plots
            fetch("http://127.0.0.1:8000/explain_plots/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    crop: crops[0].crop,
                    nitrogen: soil.N, phosphorous: soil.P, potassium: soil.K,
                    temperature: temp, humidity: hum, ph: soil.pH, rainfall: soil.Rainfall
                })
            })
            .then(res => res.json())
            .then(plotData => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv && plotData.shap_base64) {
                    xaiDiv.innerHTML = `<div style="color: #8b949e; margin-bottom: 5px; margin-top: 10px;">EXPLAINABLE AI (SHAP)</div>
<img src="data:image/png;base64,${plotData.shap_base64}" style="max-width: 100%; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px;">`;
                } else if(xaiDiv) {
                    xaiDiv.innerHTML = "";
                }
            })
            .catch(err => {
                const xaiDiv = document.getElementById("xai-plots");
                if(xaiDiv) xaiDiv.innerHTML = `<span style="color: #f85149;">> [XAI PLOTS FAILED]</span>`;
            });

        }, 800);
    })
    .catch((error) => {
        terminalOutput.innerHTML = `> EXCEPTION RAISED.<br>> ERR_DESC: ${error.message}`;
    })
    .finally(() => {
        setTimeout(() => { if(submitBtn) submitBtn.disabled = false; }, 800);
    });
}
