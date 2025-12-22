'use client';

import { useState, useRef, useEffect } from 'react';
import { LoadingOverlay, Badge } from '@/components';
import { analyzeQuantLab } from '@/lib/quantLabApi';
import type { 
  QuantLabResponse, 
  FactorName,
  FACTOR_CONFIG 
} from '@/lib/quantLabTypes';
import './quant-lab.css';

// Factor configuration with labels and icons
const factorConfig: Record<FactorName, { label: string; icon: string; description: string }> = {
  momentum: {
    label: 'Momentum',
    icon: '📈',
    description: 'Price momentum and relative strength indicators',
  },
  value: {
    label: 'Value',
    icon: '💰',
    description: 'Valuation metrics like P/E, P/B, and PEG ratios',
  },
  quality: {
    label: 'Quality',
    icon: '⭐',
    description: 'Profitability, margins, and financial health',
  },
  growth: {
    label: 'Growth',
    icon: '🌱',
    description: 'Revenue and earnings growth rates',
  },
  volatility: {
    label: 'Volatility',
    icon: '📊',
    description: 'Risk-adjusted metrics and volatility measures',
  },
  technical: {
    label: 'Technical',
    icon: '📉',
    description: 'Moving averages, MACD, and chart signals',
  },
};

// Helper functions
const formatScore = (score: number | null): string => {
  if (score === null) return 'N/A';
  return score.toFixed(1);
};

const formatPercent = (value: number | null, includeSign = false): string => {
  if (value === null) return 'N/A';
  const formatted = `${value.toFixed(2)}%`;
  return includeSign && value > 0 ? `+${formatted}` : formatted;
};

const formatCurrency = (value: number | null): string => {
  if (value === null) return 'N/A';
  return `$${value.toFixed(2)}`;
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'positive': return 'var(--positive)';
    case 'negative': return 'var(--negative)';
    case 'warning': return 'var(--warning)';
    default: return 'var(--text-secondary)';
  }
};

// Alpha Score Gauge Component
function AlphaGauge({ score, status }: { score: number | null; status: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const dpr = window.devicePixelRatio || 1;
    const size = 200;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = 80;
    const lineWidth = 12;
    
    // Clear
    ctx.clearRect(0, 0, size, size);
    
    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 0.25 * Math.PI);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Value arc
    if (score !== null) {
      const percentage = score / 100;
      const endAngle = 0.75 * Math.PI + (1.5 * Math.PI * percentage);
      
      // Gradient
      const gradient = ctx.createLinearGradient(0, size, size, 0);
      if (status === 'positive') {
        gradient.addColorStop(0, '#00a88a');
        gradient.addColorStop(1, '#00e5a0');
      } else if (status === 'negative') {
        gradient.addColorStop(0, '#ff4f4f');
        gradient.addColorStop(1, '#ff6b6b');
      } else {
        gradient.addColorStop(0, '#f59e0b');
        gradient.addColorStop(1, '#fbbf24');
      }
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, endAngle);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();
      
      // Glow effect
      ctx.shadowColor = status === 'positive' ? '#00e5a0' : status === 'negative' ? '#ff6b6b' : '#fbbf24';
      ctx.shadowBlur = 20;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    
    // Center text
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 48px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(score !== null ? Math.round(score).toString() : '—', centerX, centerY - 10);
    
    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '14px Plus Jakarta Sans, sans-serif';
    ctx.fillText('Alpha Score', centerX, centerY + 30);
    
  }, [score, status]);
  
  return (
    <canvas
      ref={canvasRef}
      style={{ width: '200px', height: '200px', display: 'block' }}
    />
  );
}

// Factor Score Card Component
function FactorCard({ 
  name, 
  score, 
  status, 
  factors,
  isExpanded,
  onToggle 
}: { 
  name: FactorName;
  score: number | null;
  status: string;
  factors: Record<string, any>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const config = factorConfig[name];
  
  return (
    <div 
      className={`quant-factor-card ${status}`}
      style={{ cursor: 'pointer' }}
      onClick={onToggle}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '24px' }}>{config.icon}</span>
          <div>
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {config.label}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {config.description}
            </div>
          </div>
        </div>
        <div style={{ 
          fontSize: '1.5rem', 
          fontWeight: 700, 
          fontFamily: "'JetBrains Mono', monospace",
          color: getStatusColor(status)
        }}>
          {formatScore(score)}
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="quant-progress-bar">
        <div 
          className={`quant-progress-fill ${status}`}
          style={{ width: score !== null ? `${score}%` : '0%' }}
        />
      </div>
      
      {/* Expanded Details */}
      {isExpanded && Object.keys(factors).length > 0 && (
        <div style={{ 
          marginTop: '16px', 
          paddingTop: '16px', 
          borderTop: '1px solid var(--border)',
          animation: 'quant-slide-up 0.3s ease-out'
        }}>
          <div style={{ fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Factor Details
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
            {Object.entries(factors).slice(0, 8).map(([key, value]) => (
              <div key={key} style={{ fontSize: '12px' }}>
                <span style={{ color: 'var(--text-muted)' }}>{key.replace(/_/g, ' ')}: </span>
                <span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>
                  {typeof value === 'number' ? value.toFixed(2) : 
                   typeof value === 'boolean' ? (value ? 'Yes' : 'No') : 
                   value ?? 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Expand indicator */}
      <div style={{ 
        textAlign: 'center', 
        marginTop: '8px',
        color: 'var(--text-muted)',
        fontSize: '12px'
      }}>
        {isExpanded ? '▲ Less' : '▼ More'}
      </div>
    </div>
  );
}

// Price Forecast Component
function ForecastSection({ forecast }: { forecast: QuantLabResponse['price_forecast'] }) {
  if (!forecast.has_forecast) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
        Insufficient data for price forecast
      </div>
    );
  }
  
  const periods = ['30d', '90d', '1y'] as const;
  
  return (
    <div>
      {/* Forecast Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
        {periods.map(period => {
          const data = forecast.forecasts[period];
          if (!data) return null;
          
          return (
            <div key={period} className="quant-forecast-card">
              <div className="quant-forecast-label">{period === '1y' ? '1 Year' : period} Target</div>
              <div className="quant-forecast-value" style={{ color: 'var(--text-primary)' }}>
                {formatCurrency(data.median)}
              </div>
              <div 
                className="quant-forecast-change"
                style={{ color: data.expected_return >= 0 ? 'var(--positive)' : 'var(--negative)' }}
              >
                {formatPercent(data.expected_return, true)}
              </div>
              <div style={{ 
                fontSize: '11px', 
                color: 'var(--text-muted)', 
                marginTop: '8px',
                borderTop: '1px solid var(--border)',
                paddingTop: '8px'
              }}>
                Range: {formatCurrency(data.p25)} - {formatCurrency(data.p75)}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Scenarios */}
      {forecast.scenarios && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          {(['bull', 'base', 'bear'] as const).map(scenario => {
            const data = forecast.scenarios[scenario];
            const colors = {
              bull: { bg: 'var(--positive-light)', color: 'var(--positive)' },
              base: { bg: 'var(--neutral-light)', color: 'var(--text-secondary)' },
              bear: { bg: 'var(--negative-light)', color: 'var(--negative)' },
            };
            
            return (
              <div 
                key={scenario}
                style={{
                  background: colors[scenario].bg,
                  borderRadius: '12px',
                  padding: '16px',
                  border: `1px solid ${colors[scenario].color}33`,
                }}
              >
                <div style={{ 
                  fontSize: '13px', 
                  fontWeight: 600, 
                  color: colors[scenario].color,
                  marginBottom: '8px'
                }}>
                  {data.label} ({data.probability})
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                  {data.description}
                </div>
                <div style={{ fontSize: '12px' }}>
                  <div style={{ marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>30d: </span>
                    <span style={{ fontWeight: 600 }}>{formatCurrency(data.price_30d)}</span>
                  </div>
                  <div style={{ marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>90d: </span>
                    <span style={{ fontWeight: 600 }}>{formatCurrency(data.price_90d)}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>1y: </span>
                    <span style={{ fontWeight: 600 }}>{formatCurrency(data.price_1y)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Methodology note */}
      {forecast.methodology && (
        <div style={{ 
          marginTop: '16px', 
          fontSize: '11px', 
          color: 'var(--text-muted)',
          fontStyle: 'italic',
          textAlign: 'center'
        }}>
          {forecast.methodology}
        </div>
      )}
    </div>
  );
}

// Risk Analysis Component
function RiskSection({ risk }: { risk: QuantLabResponse['risk_analysis'] }) {
  if (!risk.has_risk_data) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
        Insufficient data for risk analysis
      </div>
    );
  }
  
  return (
    <div>
      {/* Risk Score Header */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '24px',
        padding: '20px',
        background: `${getStatusColor(risk.risk_status)}15`,
        borderRadius: '12px',
        border: `1px solid ${getStatusColor(risk.risk_status)}33`
      }}>
        <div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Risk Level</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: getStatusColor(risk.risk_status) }}>
            {risk.risk_level}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>Risk Score</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
            {risk.risk_score.toFixed(1)}/100
          </div>
        </div>
      </div>
      
      {/* Volatility Metrics */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Volatility
        </div>
        <div className="quant-risk-grid">
          <div className="quant-risk-item">
            <div className="quant-risk-label">Annual Vol</div>
            <div className="quant-risk-value">{risk.volatility.annual.toFixed(1)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">Daily Vol</div>
            <div className="quant-risk-value">{risk.volatility.daily.toFixed(2)}%</div>
          </div>
          {risk.volatility.upside && (
            <div className="quant-risk-item">
              <div className="quant-risk-label">Upside Vol</div>
              <div className="quant-risk-value" style={{ color: 'var(--positive)' }}>{risk.volatility.upside.toFixed(1)}%</div>
            </div>
          )}
          {risk.volatility.downside && (
            <div className="quant-risk-item">
              <div className="quant-risk-label">Downside Vol</div>
              <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.volatility.downside.toFixed(1)}%</div>
            </div>
          )}
        </div>
      </div>
      
      {/* VaR Metrics */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Value at Risk (VaR)
        </div>
        <div className="quant-risk-grid">
          <div className="quant-risk-item">
            <div className="quant-risk-label">VaR 95% (1d)</div>
            <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.var.var_95_1d.toFixed(2)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">VaR 99% (1d)</div>
            <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.var.var_99_1d.toFixed(2)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">VaR 95% (30d)</div>
            <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.var.var_95_30d.toFixed(2)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">CVaR 95%</div>
            <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.var.cvar_95.toFixed(2)}%</div>
          </div>
        </div>
      </div>
      
      {/* Drawdown */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Drawdown Analysis
        </div>
        <div className="quant-risk-grid">
          <div className="quant-risk-item">
            <div className="quant-risk-label">Max Drawdown</div>
            <div className="quant-risk-value" style={{ color: 'var(--negative)' }}>{risk.drawdown.max_drawdown.toFixed(2)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">Current DD</div>
            <div className="quant-risk-value">{risk.drawdown.current_drawdown.toFixed(2)}%</div>
          </div>
          <div className="quant-risk-item">
            <div className="quant-risk-label">Days in DD</div>
            <div className="quant-risk-value">{risk.drawdown.days_in_drawdown}</div>
          </div>
          {risk.beta && (
            <div className="quant-risk-item">
              <div className="quant-risk-label">Beta</div>
              <div className="quant-risk-value">{risk.beta.toFixed(2)}</div>
            </div>
          )}
        </div>
      </div>
      
      {/* Distribution */}
      <div>
        <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Return Distribution
        </div>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
          <div style={{ fontSize: '13px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Skewness: </span>
            <span style={{ fontWeight: 600 }}>{risk.distribution.skewness.toFixed(3)} ({risk.distribution.skew_direction})</span>
          </div>
          <div style={{ fontSize: '13px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Kurtosis: </span>
            <span style={{ fontWeight: 600 }}>{risk.distribution.kurtosis.toFixed(3)}</span>
          </div>
          {risk.distribution.fat_tails && (
            <Badge variant="warning">Fat Tails Detected</Badge>
          )}
        </div>
      </div>
    </div>
  );
}

// Signal Section Component
function SignalSection({ signal, company }: { signal: QuantLabResponse['signal']; company: QuantLabResponse['company'] }) {
  return (
    <div className={`quant-signal-card ${signal.action_status}`}>
      {/* Action & Confidence */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
        <div className={`quant-action-badge ${signal.action_status}`}>
          {signal.action}
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Confidence</div>
          <Badge variant={signal.confidence === 'High' ? 'success' : signal.confidence === 'Medium' ? 'warning' : 'default'}>
            {signal.confidence}
          </Badge>
        </div>
      </div>
      
      {/* Position Guidance */}
      {signal.position_guidance && (
        <div style={{ 
          background: 'var(--bg-secondary)', 
          borderRadius: '12px', 
          padding: '16px',
          marginBottom: '24px'
        }}>
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Position Guidance
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Stop Loss</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--negative)', fontFamily: "'JetBrains Mono', monospace" }}>
                {formatCurrency(signal.position_guidance.stop_loss)}
              </div>
            </div>
            {signal.position_guidance.target_short && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Target (30d)</div>
                <div style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--positive)', fontFamily: "'JetBrains Mono', monospace" }}>
                  {formatCurrency(signal.position_guidance.target_short)}
                </div>
              </div>
            )}
            {signal.position_guidance.target_medium && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Target (90d)</div>
                <div style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--positive)', fontFamily: "'JetBrains Mono', monospace" }}>
                  {formatCurrency(signal.position_guidance.target_medium)}
                </div>
              </div>
            )}
            {signal.position_guidance.risk_reward_ratio && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Risk/Reward</div>
                <div style={{ fontSize: '1.125rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                  {signal.position_guidance.risk_reward_ratio.toFixed(2)}x
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Bull & Bear Cases */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Bull Case */}
        <div>
          <div style={{ 
            fontSize: '13px', 
            fontWeight: 600, 
            color: 'var(--positive)',
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span>🐂</span> Bull Case
          </div>
          <ul className="quant-thesis-list">
            {signal.bull_case.map((point, idx) => (
              <li key={idx} className="quant-thesis-item">
                <div className="quant-thesis-bullet bull" />
                <div className="quant-thesis-text">{point}</div>
              </li>
            ))}
            {signal.bull_case.length === 0 && (
              <li className="quant-thesis-item">
                <div className="quant-thesis-text" style={{ color: 'var(--text-muted)' }}>No strong bullish signals detected</div>
              </li>
            )}
          </ul>
        </div>
        
        {/* Bear Case */}
        <div>
          <div style={{ 
            fontSize: '13px', 
            fontWeight: 600, 
            color: 'var(--negative)',
            marginBottom: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span>🐻</span> Bear Case
          </div>
          <ul className="quant-thesis-list">
            {signal.bear_case.map((point, idx) => (
              <li key={idx} className="quant-thesis-item">
                <div className="quant-thesis-bullet bear" />
                <div className="quant-thesis-text">{point}</div>
              </li>
            ))}
            {signal.bear_case.length === 0 && (
              <li className="quant-thesis-item">
                <div className="quant-thesis-text" style={{ color: 'var(--text-muted)' }}>No strong bearish signals detected</div>
              </li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}

// Technical Patterns Section
function TechnicalSection({ technical }: { technical: QuantLabResponse['technical_analysis'] }) {
  const { patterns, support_resistance } = technical;
  
  return (
    <div>
      {/* Patterns */}
      {patterns.length > 0 && (
        <div style={{ marginBottom: '24px' }}>
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Detected Patterns
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            {patterns.map((pattern, idx) => (
              <div 
                key={idx}
                className={`quant-pattern-badge ${pattern.status}`}
                title={pattern.description}
              >
                {pattern.name}
                <span style={{ opacity: 0.7, fontSize: '11px' }}>({pattern.confidence}%)</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Support & Resistance */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Resistance Levels
          </div>
          {support_resistance.resistance_levels.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {support_resistance.resistance_levels.map((level, idx) => (
                <div 
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'var(--negative-light)',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 107, 107, 0.2)',
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>R{idx + 1}</span>
                  <span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: 'var(--negative)' }}>
                    ${level.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No resistance levels detected</div>
          )}
          {support_resistance.nearest_resistance && support_resistance.distance_to_resistance && (
            <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
              Nearest resistance {formatPercent(support_resistance.distance_to_resistance, true)} away
            </div>
          )}
        </div>
        
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Support Levels
          </div>
          {support_resistance.support_levels.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {support_resistance.support_levels.map((level, idx) => (
                <div 
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'var(--positive-light)',
                    borderRadius: '8px',
                    border: '1px solid rgba(0, 229, 160, 0.2)',
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>S{idx + 1}</span>
                  <span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: 'var(--positive)' }}>
                    ${level.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No support levels detected</div>
          )}
          {support_resistance.nearest_support && support_resistance.distance_to_support && (
            <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
              Nearest support {formatPercent(support_resistance.distance_to_support)} away
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function QuantLabPage() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<QuantLabResponse | null>(null);
  const [expandedFactors, setExpandedFactors] = useState<Set<FactorName>>(new Set());
  
  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await analyzeQuantLab(ticker);
      setData(result);
      setExpandedFactors(new Set());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setData(null);
    } finally {
      setLoading(false);
    }
  };
  
  const toggleFactor = (factor: FactorName) => {
    setExpandedFactors(prev => {
      const next = new Set(prev);
      if (next.has(factor)) {
        next.delete(factor);
      } else {
        next.add(factor);
      }
      return next;
    });
  };
  
  return (
    <main style={{ padding: '100px 24px 60px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '40px', textAlign: 'center' }}>
        <h1 style={{ 
          fontSize: '2.5rem', 
          fontWeight: 800, 
          marginBottom: '12px',
          background: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Quant Lab
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.125rem', maxWidth: '600px', margin: '0 auto' }}>
          Multi-factor intelligence engine with Monte Carlo forecasting, risk analysis, and pattern recognition
        </p>
      </div>
      
      {/* Search Form */}
      <form onSubmit={handleAnalyze} style={{ marginBottom: '40px' }}>
        <div style={{ 
          display: 'flex', 
          gap: '12px', 
          maxWidth: '500px', 
          margin: '0 auto' 
        }}>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter ticker (e.g., AAPL)"
            style={{
              flex: 1,
              padding: '14px 20px',
              fontSize: '1rem',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: '12px',
              color: 'var(--text-primary)',
              outline: 'none',
              transition: 'border-color 0.2s ease',
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--accent)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--border)'}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: '14px 32px',
              fontSize: '1rem',
              fontWeight: 600,
              background: 'linear-gradient(135deg, var(--accent) 0%, #00a3ff 100%)',
              border: 'none',
              borderRadius: '12px',
              color: '#000',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1,
              transition: 'all 0.2s ease',
            }}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </form>
      
      {/* Error Message */}
      {error && (
        <div style={{
          maxWidth: '500px',
          margin: '0 auto 24px',
          padding: '16px',
          background: 'var(--negative-light)',
          border: '1px solid rgba(255, 107, 107, 0.3)',
          borderRadius: '12px',
          color: 'var(--negative)',
          textAlign: 'center',
        }}>
          {error}
        </div>
      )}
      
      {/* Loading Overlay */}
      {loading && <LoadingOverlay message="Running multi-factor analysis..." />}
      
      {/* Results */}
      {data && !loading && (
        <div className="quant-animate-slide-up">
          {/* Company Header */}
          <div style={{
            background: 'var(--bg-card)',
            borderRadius: '20px',
            padding: '24px 32px',
            marginBottom: '24px',
            border: '1px solid var(--border)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '20px',
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0 }}>{data.company.name}</h2>
                <Badge variant="default">{data.ticker}</Badge>
              </div>
              <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                {data.company.sector} • {data.company.industry}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '2rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                {data.company.price_display}
              </div>
              {data.company.change_percent !== null && (
                <div style={{ 
                  fontSize: '1rem', 
                  fontWeight: 600,
                  color: data.company.change_percent >= 0 ? 'var(--positive)' : 'var(--negative)'
                }}>
                  {formatPercent(data.company.change_percent, true)}
                </div>
              )}
              <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                {data.company.market_cap_display}
              </div>
            </div>
          </div>
          
          {/* Alpha Score & Signal Row */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'minmax(280px, 1fr) 2fr', 
            gap: '24px',
            marginBottom: '24px'
          }}>
            {/* Alpha Score */}
            <div style={{
              background: 'var(--bg-card)',
              borderRadius: '20px',
              padding: '32px',
              border: '1px solid var(--border)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <AlphaGauge score={data.alpha_score.score} status={data.alpha_score.status} />
              <div style={{ marginTop: '16px', textAlign: 'center' }}>
                <Badge variant={data.alpha_score.status === 'positive' ? 'success' : data.alpha_score.status === 'negative' ? 'error' : 'warning'}>
                  {data.alpha_score.status === 'positive' ? 'Bullish' : data.alpha_score.status === 'negative' ? 'Bearish' : 'Neutral'}
                </Badge>
              </div>
            </div>
            
            {/* Signal */}
            <SignalSection signal={data.signal} company={data.company} />
          </div>
          
          {/* Factor Scores Grid */}
          <div style={{
            background: 'var(--bg-card)',
            borderRadius: '20px',
            padding: '24px',
            marginBottom: '24px',
            border: '1px solid var(--border)',
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '20px' }}>Factor Analysis</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
              {(Object.keys(factorConfig) as FactorName[]).map((factor) => {
                const factorData = data.factor_scores[factor];
                return (
                  <FactorCard
                    key={factor}
                    name={factor}
                    score={factorData.score}
                    status={factorData.status}
                    factors={factorData.factors}
                    isExpanded={expandedFactors.has(factor)}
                    onToggle={() => toggleFactor(factor)}
                  />
                );
              })}
            </div>
          </div>
          
          {/* Price Forecast */}
          <div style={{
            background: 'var(--bg-card)',
            borderRadius: '20px',
            padding: '24px',
            marginBottom: '24px',
            border: '1px solid var(--border)',
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '20px' }}>Price Forecast</h3>
            <ForecastSection forecast={data.price_forecast} />
          </div>
          
          {/* Risk Analysis & Technical in two columns */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
            {/* Risk Analysis */}
            <div style={{
              background: 'var(--bg-card)',
              borderRadius: '20px',
              padding: '24px',
              border: '1px solid var(--border)',
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '20px' }}>Risk Analysis</h3>
              <RiskSection risk={data.risk_analysis} />
            </div>
            
            {/* Technical Analysis */}
            <div style={{
              background: 'var(--bg-card)',
              borderRadius: '20px',
              padding: '24px',
              border: '1px solid var(--border)',
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '20px' }}>Technical Analysis</h3>
              <TechnicalSection technical={data.technical_analysis} />
            </div>
          </div>
          
          {/* Disclaimer */}
          <div style={{
            padding: '16px 24px',
            background: 'var(--bg-secondary)',
            borderRadius: '12px',
            border: '1px solid var(--border)',
            fontSize: '12px',
            color: 'var(--text-muted)',
            textAlign: 'center',
          }}>
            {data.disclaimer}
          </div>
        </div>
      )}
    </main>
  );
}
