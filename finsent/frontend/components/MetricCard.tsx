'use client';

import { ReactNode } from 'react';

interface MetricCardProps {
  name: string;
  value: string;
  status?: 'positive' | 'negative' | 'neutral';
  description?: string;
  icon?: ReactNode;
  compact?: boolean;
  className?: string;
}

export default function MetricCard({ name, value, status = 'neutral', description, icon, compact = false, className = '' }: MetricCardProps) {
  const statusColors = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    neutral: 'var(--text-primary)',
  };

  return (
    <div
      className={className}
      style={{
        background: 'var(--bg-secondary)',
        borderRadius: compact ? '12px' : '16px',
        padding: compact ? '14px' : '20px',
        position: 'relative',
        overflow: 'hidden',
        border: '1px solid var(--border)',
        transition: 'all 0.3s var(--ease-out-expo)',
      }}
    >
      {/* Status indicator bar */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '3px',
          background: statusColors[status],
        }}
      />
      
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--text-tertiary)',
              marginBottom: '8px',
            }}
          >
            {name}
          </div>
          <div
            style={{
              fontSize: compact ? '1.125rem' : '1.375rem',
              fontWeight: 700,
              fontFamily: "'JetBrains Mono', monospace",
              color: statusColors[status],
            }}
          >
            {value}
          </div>
          {description && (
            <div
              style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                marginTop: '6px',
              }}
            >
              {description}
            </div>
          )}
        </div>
        {icon && (
          <div
            style={{
              color: 'var(--text-muted)',
              opacity: 0.5,
            }}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string;
  status?: 'positive' | 'negative' | 'neutral';
  suffix?: string;
}

export function MetricRow({ label, value, status, suffix }: MetricRowProps) {
  const statusColors = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    neutral: 'var(--text-primary)',
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 0',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{label}</span>
      <span
        style={{
          fontSize: '14px',
          fontWeight: 600,
          fontFamily: "'JetBrains Mono', monospace",
          color: status ? statusColors[status] : 'var(--text-primary)',
        }}
      >
        {value}
        {suffix && <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: '4px' }}>{suffix}</span>}
      </span>
    </div>
  );
}

interface MetricGridProps {
  metrics: Array<{
    name: string;
    value: string;
    status?: 'positive' | 'negative' | 'neutral';
  }>;
  columns?: 2 | 3 | 4;
  compact?: boolean;
}

export function MetricGrid({ metrics, columns = 4, compact = false }: MetricGridProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: compact ? '12px' : '16px',
      }}
    >
      {metrics.map((metric, i) => (
        <MetricCard
          key={i}
          name={metric.name}
          value={metric.value}
          status={metric.status}
          compact={compact}
        />
      ))}
    </div>
  );
}
