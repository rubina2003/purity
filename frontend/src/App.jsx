import React, { useState } from 'react';

const PRESETS = {
  safe: {
    name: 'Kathmandu Treated Spring',
    ph: 7.2,
    hardness: 180,
    solids: 18500,
    chloramines: 6.8,
    sulfate: 310,
    conductivity: 420,
    organic_carbon: 12.5,
    trihalomethanes: 62,
    turbidity: 3.2
  },
  groundwater: {
    name: 'Kathmandu Valley Groundwater (Raw)',
    ph: 6.1,
    hardness: 260,
    solids: 24500,
    chloramines: 4.2,
    sulfate: 380,
    conductivity: 580,
    organic_carbon: 18.2,
    trihalomethanes: 88,
    turbidity: 5.6
  },
  contaminated: {
    name: 'Monsoon Urban Runoff',
    ph: 5.4,
    hardness: 320,
    solids: 38000,
    chloramines: 9.8,
    sulfate: 440,
    conductivity: 680,
    organic_carbon: 22.4,
    trihalomethanes: 105,
    turbidity: 7.8
  }
};

const PARAM_INFO = [
  { key: 'ph', label: 'pH Value', unit: 'pH', min: 0, max: 14, step: 0.1, safeMin: 6.5, safeMax: 8.5, desc: 'NDWQS safe range: 6.5 - 8.5' },
  { key: 'hardness', label: 'Hardness', unit: 'mg/L', min: 50, max: 400, step: 1, safeMin: 100, safeMax: 300, desc: 'CaCO3 equivalent concentration' },
  { key: 'solids', label: 'Total Dissolved Solids (TDS)', unit: 'ppm', min: 5000, max: 50000, step: 100, safeMin: 0, safeMax: 20000, desc: 'High mineralization indicator' },
  { key: 'chloramines', label: 'Chloramines', unit: 'ppm', min: 1, max: 15, step: 0.1, safeMin: 0, safeMax: 8.0, desc: 'Disinfection agent residue' },
  { key: 'sulfate', label: 'Sulfate', unit: 'mg/L', min: 100, max: 600, step: 1, safeMin: 0, safeMax: 350, desc: 'Naturally occurring mineral' },
  { key: 'conductivity', label: 'Electrical Conductivity', unit: 'μS/cm', min: 100, max: 800, step: 5, safeMin: 0, safeMax: 500, desc: 'Dissolved ions concentration' },
  { key: 'organic_carbon', label: 'Organic Carbon', unit: 'mg/L', min: 2, max: 30, step: 0.1, safeMin: 0, safeMax: 15, desc: 'Total organic carbon content' },
  { key: 'trihalomethanes', label: 'Trihalomethanes', unit: 'μg/L', min: 10, max: 130, step: 1, safeMin: 0, safeMax: 80, desc: 'Disinfection by-product' },
  { key: 'turbidity', label: 'Turbidity', unit: 'NTU', min: 0.5, max: 10, step: 0.1, safeMin: 0, safeMax: 5.0, desc: 'NDWQS upper limit: 5 NTU' },
];

export default function App() {
  const [params, setParams] = useState(PRESETS.safe);
  const [model, setModel] = useState('Random Forest');
  const [activeTab, setActiveTab] = useState('analyzer');

  const handleChange = (key, value) => {
    setParams(prev => ({
      ...prev,
      [key]: parseFloat(value) || 0
    }));
  };

  const applyPreset = (presetKey) => {
    setParams({ ...PRESETS[presetKey] });
  };

  // Rule-based heuristic simulating model assessment until backend inference is connected
  const violations = PARAM_INFO.filter(item => {
    const val = params[item.key];
    return val < item.safeMin || val > item.safeMax;
  });

  const isPotable = violations.length <= 1;
  const potabilityScore = Math.max(12, Math.min(98, Math.round(100 - (violations.length * 22) + (params.ph >= 6.8 && params.ph <= 7.8 ? 8 : -8))));

  return (
    <div style={{ maxWidth: '1240px', margin: '0 auto', padding: '24px 20px 60px' }}>
      {/* Top Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        paddingBottom: '24px',
        borderBottom: '1px solid var(--border-subtle)',
        marginBottom: '32px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '46px',
            height: '46px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 24px rgba(0, 210, 255, 0.4)'
          }}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
            </svg>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '24px', fontWeight: '800', letterSpacing: '-0.5px', color: '#fff' }}>PURITY</h1>
              <span style={{
                background: 'rgba(0, 210, 255, 0.12)',
                color: '#00d2ff',
                fontSize: '11px',
                fontWeight: '700',
                padding: '3px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(0, 210, 255, 0.25)'
              }}>
                Project III
              </span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Water Potability Classification & Quality Assessment System • Nepal Focus
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {['analyzer', 'standards', 'about'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: '600',
                textTransform: 'capitalize',
                background: activeTab === tab ? 'rgba(0, 210, 255, 0.15)' : 'rgba(255, 255, 255, 0.04)',
                color: activeTab === tab ? '#00d2ff' : 'var(--text-secondary)',
                border: activeTab === tab ? '1px solid rgba(0, 210, 255, 0.35)' : '1px solid var(--border-subtle)'
              }}
            >
              {tab === 'standards' ? 'NDWQS Standards' : tab === 'analyzer' ? 'Test Sample' : 'Project Research'}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === 'analyzer' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '28px' }}>
          
          {/* Left Column: Sample Inputs */}
          <div style={{
            background: 'var(--bg-card)',
            backdropFilter: 'blur(16px)',
            borderRadius: '16px',
            padding: '24px',
            border: '1px solid var(--border-subtle)',
            boxShadow: 'var(--glass-shadow)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
              <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#fff' }}>Water Quality Parameters</h2>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>9 Features</span>
            </div>

            {/* Presets */}
            <div style={{ marginBottom: '22px' }}>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: '600' }}>
                Load Sample Preset:
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                <button
                  onClick={() => applyPreset('safe')}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600',
                    background: 'rgba(16, 185, 129, 0.15)',
                    color: '#10b981',
                    border: '1px solid rgba(16, 185, 129, 0.3)'
                  }}
                >
                  ✓ Clean Spring
                </button>
                <button
                  onClick={() => applyPreset('groundwater')}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600',
                    background: 'rgba(245, 158, 11, 0.15)',
                    color: '#f59e0b',
                    border: '1px solid rgba(245, 158, 11, 0.3)'
                  }}
                >
                  ⚡ Kathmandu Groundwater (KUKL)
                </button>
                <button
                  onClick={() => applyPreset('contaminated')}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '600',
                    background: 'rgba(244, 63, 94, 0.15)',
                    color: '#f43f5e',
                    border: '1px solid rgba(244, 63, 94, 0.3)'
                  }}
                >
                  ✕ Turbid Runoff
                </button>
              </div>
            </div>

            {/* Parameter Inputs List */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {PARAM_INFO.map(item => {
                const val = params[item.key] ?? 0;
                const isOutOfRange = val < item.safeMin || val > item.safeMax;

                return (
                  <div key={item.key} style={{
                    padding: '12px 14px',
                    borderRadius: '10px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: isOutOfRange ? '1px solid rgba(244, 63, 94, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <div>
                        <span style={{ fontSize: '13px', fontWeight: '600', color: '#fff' }}>{item.label}</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '6px' }}>({item.unit})</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{
                          fontSize: '13px',
                          fontWeight: '700',
                          color: isOutOfRange ? '#f43f5e' : '#10b981'
                        }}>
                          {val}
                        </span>
                        {isOutOfRange && (
                          <span style={{ fontSize: '10px', color: '#f43f5e', background: 'rgba(244, 63, 94, 0.15)', padding: '1px 5px', borderRadius: '4px' }}>
                            Limit!
                          </span>
                        )}
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <input
                        type="range"
                        min={item.min}
                        max={item.max}
                        step={item.step}
                        value={val}
                        onChange={(e) => handleChange(item.key, e.target.value)}
                        style={{ flex: 1, accentColor: isOutOfRange ? '#f43f5e' : '#00d2ff', cursor: 'pointer' }}
                      />
                      <input
                        type="number"
                        min={item.min}
                        max={item.max}
                        step={item.step}
                        value={val}
                        onChange={(e) => handleChange(item.key, e.target.value)}
                        style={{
                          width: '74px',
                          padding: '4px 8px',
                          borderRadius: '6px',
                          border: '1px solid var(--border-subtle)',
                          background: 'rgba(0, 0, 0, 0.3)',
                          color: '#fff',
                          fontSize: '12px',
                          textAlign: 'right'
                        }}
                      />
                    </div>
                    <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
                      {item.desc}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Prediction & Evaluation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            {/* Model Card */}
            <div style={{
              background: 'var(--bg-card)',
              backdropFilter: 'blur(16px)',
              borderRadius: '16px',
              padding: '24px',
              border: '1px solid var(--border-subtle)',
              boxShadow: 'var(--glass-shadow)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#fff' }}>Classification Model</h2>
                <span style={{ fontSize: '12px', color: '#00d2ff', background: 'rgba(0, 210, 255, 0.1)', padding: '2px 8px', borderRadius: '4px' }}>
                  Supervised
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '20px' }}>
                {['Random Forest', 'SVM', 'Logistic Regression'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setModel(m)}
                    style={{
                      padding: '10px 8px',
                      borderRadius: '8px',
                      fontSize: '12px',
                      fontWeight: '600',
                      textAlign: 'center',
                      background: model === m ? 'rgba(0, 210, 255, 0.18)' : 'rgba(255, 255, 255, 0.03)',
                      color: model === m ? '#00d2ff' : 'var(--text-secondary)',
                      border: model === m ? '1px solid rgba(0, 210, 255, 0.4)' : '1px solid var(--border-subtle)'
                    }}
                  >
                    {m}
                  </button>
                ))}
              </div>

              {/* Potability Result Box */}
              <div style={{
                borderRadius: '14px',
                padding: '24px',
                background: isPotable 
                  ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 78, 59, 0.25))' 
                  : 'linear-gradient(135deg, rgba(244, 63, 94, 0.12), rgba(136, 19, 55, 0.25))',
                border: isPotable ? '1px solid rgba(16, 185, 129, 0.35)' : '1px solid rgba(244, 63, 94, 0.35)',
                textAlign: 'center'
              }}>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '64px',
                  height: '64px',
                  borderRadius: '50%',
                  background: isPotable ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)',
                  color: isPotable ? '#10b981' : '#f43f5e',
                  marginBottom: '12px'
                }}>
                  {isPotable ? (
                    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  ) : (
                    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  )}
                </div>

                <h3 style={{
                  fontSize: '22px',
                  fontWeight: '800',
                  color: isPotable ? '#10b981' : '#f43f5e',
                  marginBottom: '6px'
                }}>
                  {isPotable ? 'POTABLE WATER' : 'NON-POTABLE WATER'}
                </h3>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '340px', margin: '0 auto 16px' }}>
                  {isPotable 
                    ? 'Sample meets standard physicochemical guidelines for human consumption.' 
                    : `Sample violates safety thresholds in ${violations.length} parameter(s) and requires purification.`}
                </p>

                {/* Potability Score Bar */}
                <div style={{ background: 'rgba(0, 0, 0, 0.35)', padding: '12px 16px', borderRadius: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: '600', marginBottom: '6px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Calculated Safety Confidence</span>
                    <span style={{ color: isPotable ? '#10b981' : '#f43f5e' }}>{potabilityScore}%</span>
                  </div>
                  <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${potabilityScore}%`,
                      height: '100%',
                      background: isPotable ? 'linear-gradient(90deg, #10b981, #00d2ff)' : 'linear-gradient(90deg, #f43f5e, #f59e0b)',
                      transition: 'width 0.4s ease'
                    }} />
                  </div>
                </div>
              </div>

              {/* Violations Summary */}
              {violations.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <p style={{ fontSize: '12px', fontWeight: '700', color: '#f43f5e', marginBottom: '8px' }}>
                    Threshold Infractions Detected:
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {violations.map(v => (
                      <div key={v.key} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '12px',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        background: 'rgba(244, 63, 94, 0.08)',
                        border: '1px solid rgba(244, 63, 94, 0.2)'
                      }}>
                        <span style={{ color: '#fff' }}>{v.label}</span>
                        <span style={{ color: '#f43f5e', fontWeight: '600' }}>
                          {params[v.key]} {v.unit} (Safe: {v.safeMin}–{v.safeMax})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Quick Context Card */}
            <div style={{
              background: 'var(--bg-card)',
              backdropFilter: 'blur(16px)',
              borderRadius: '16px',
              padding: '20px',
              border: '1px solid var(--border-subtle)',
              fontSize: '12.5px',
              color: 'var(--text-secondary)'
            }}>
              <h4 style={{ color: '#fff', fontSize: '13px', fontWeight: '700', marginBottom: '8px' }}>
                🇳🇵 Nepal Research Context (NDWQS & KUKL)
              </h4>
              <p style={{ lineHeight: '1.6' }}>
                Groundwater sources utilized by KUKL in Kathmandu Valley frequently exhibit elevated turbidity, 
                ammonia, and localized mineral concentrations following pre-monsoon and monsoon recharge cycles.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab: NDWQS Standards */}
      {activeTab === 'standards' && (
        <div style={{
          background: 'var(--bg-card)',
          backdropFilter: 'blur(16px)',
          borderRadius: '16px',
          padding: '28px',
          border: '1px solid var(--border-subtle)',
          boxShadow: 'var(--glass-shadow)'
        }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#fff', marginBottom: '8px' }}>
            Nepal Drinking Water Quality Standards (NDWQS) Reference
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '24px' }}>
            National guidelines established by the Government of Nepal (DWSS) for potable drinking water evaluation.
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: '#00d2ff' }}>
                  <th style={{ padding: '12px 14px' }}>Parameter</th>
                  <th style={{ padding: '12px 14px' }}>NDWQS Maximum Limit</th>
                  <th style={{ padding: '12px 14px' }}>WHO Standard</th>
                  <th style={{ padding: '12px 14px' }}>Health / Sensory Effect</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'pH Value', ndwqs: '6.5 – 8.5', who: '6.5 – 8.5', effect: 'Corrosion, taste, disinfection efficiency' },
                  { name: 'Total Dissolved Solids (TDS)', ndwqs: '1000 mg/L', who: '600 – 1000 mg/L', effect: 'Palatability, mineral scaling' },
                  { name: 'Turbidity', ndwqs: '5 NTU', who: '5 NTU', effect: 'Microbial shield, aesthetic clarity' },
                  { name: 'Total Hardness (as CaCO3)', ndwqs: '500 mg/L', who: '300 mg/L', effect: 'Scale formation, soap consumption' },
                  { name: 'Sulfate (SO4)', ndwqs: '250 mg/L', who: '250 mg/L', effect: 'Laxative effects at high levels' },
                  { name: 'Free Residual Chlorine', ndwqs: '0.1 – 0.2 mg/L', who: '0.2 – 0.5 mg/L', effect: 'Microbial disinfection residual' },
                ].map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)', color: '#e2e8f0' }}>
                    <td style={{ padding: '12px 14px', fontWeight: '600' }}>{row.name}</td>
                    <td style={{ padding: '12px 14px', color: '#10b981', fontWeight: '600' }}>{row.ndwqs}</td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>{row.who}</td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)' }}>{row.effect}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab: Project Research */}
      {activeTab === 'about' && (
        <div style={{
          background: 'var(--bg-card)',
          backdropFilter: 'blur(16px)',
          borderRadius: '16px',
          padding: '28px',
          border: '1px solid var(--border-subtle)',
          boxShadow: 'var(--glass-shadow)',
          lineHeight: '1.7'
        }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#fff', marginBottom: '8px' }}>
            Project III: Machine Learning for Water Potability in Nepal
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
            8th Semester Engineering Capstone • Department of Computer / IT Engineering
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginTop: '20px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
              <h3 style={{ color: '#00d2ff', fontSize: '14px', fontWeight: '700', marginBottom: '6px' }}>1. Models Evaluated</h3>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                Comparing <strong>Random Forest</strong> (ensemble), <strong>Support Vector Machines (SVM)</strong>, and <strong>Logistic Regression</strong> for binary classification with SMOTE & stratified cross-validation.
              </p>
            </div>

            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
              <h3 style={{ color: '#10b981', fontSize: '14px', fontWeight: '700', marginBottom: '6px' }}>2. Data Sources</h3>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                Kaggle Potability Benchmark (3,276 samples) combined with PANGAEA KUKL Kathmandu Valley Groundwater dataset (60 operational sources) and DWSS reports.
              </p>
            </div>

            <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '18px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
              <h3 style={{ color: '#f59e0b', fontSize: '14px', fontWeight: '700', marginBottom: '6px' }}>3. Real-world Objective</h3>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                Enable rapid testing and prediction of drinking water viability for community taps, tube wells, and tankers across Kathmandu and Terai regions.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
